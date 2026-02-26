import sys
import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List

import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
from torchvision import transforms as T
from torch.utils.data import DataLoader

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.data import DatasetDataFrame
from masscls.models.multihead import MultiHeadCNNForClassification, MultiHeadCNNConfig
from masscls.utils import (
    compute_comprehensive_metrics,
    prepare_dataset,
    print_metrics_summary,
    load_checkpoint,
)

random.seed(43876)
np.random.seed(43876)
torch.manual_seed(43876)
torch.cuda.manual_seed(43876)
torch.cuda.manual_seed_all(43876)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

BASE_DIR = Path(__file__).resolve().parent

VALID_ON = [
    "cbis",
    "cdd-cesm",
    # "vida",
]


@torch.no_grad()
def basic_validation(
    dataloader: DataLoader,
    model: nn.Module,
    device: str,
    epoch: int,
    use_amp: bool,
    class_names: Dict[str, Dict[str, int]],
) -> Tuple:
    model.eval()

    # Storage for predictions
    all_targets = defaultdict(list)
    all_preds = defaultdict(list)
    all_probs = defaultdict(list)

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for inputs in pbar:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        # Forward pass
        with torch.autocast(device_type=device, enabled=use_amp):
            outputs = model(inputs["pixel_values"])

        # Store predictions
        for head_name, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            targets = torch.argmax(inputs[head_name], dim=1)

            all_probs[head_name].append(probs.cpu().numpy())
            all_preds[head_name].append(preds.cpu().numpy())
            all_targets[head_name].append(targets.cpu().numpy())

        # if "birads" in outputs and "pathology" not in outputs:
        #     probs = torch.softmax(outputs["cls"], dim=1)
        #     probs = torch.stack(
        #         [
        #             probs[:, [0, 3, 4]].sum(dim=1),
        #             probs[:, [1, 2]].sum(dim=1),
        #         ],
        #         dim=1,
        #     )
        #     preds = torch.argmax(probs, dim=1)

        #     mapping = torch.tensor([0, 1, 1, 0, 0]).to(targets.device)
        #     targets = torch.argmax(inputs["pathology"], dim=1)
        #     targets = mapping[targets]

        #     all_probs["pathology"].append(probs.cpu().numpy())
        #     all_preds["pathology"].append(preds.cpu().numpy())
        #     all_targets["pathology"].append(targets.cpu().numpy())

    # Compute metrics
    metrics = compute_comprehensive_metrics(
        all_targets,
        all_preds,
        all_probs,
        class_names,
    )

    return metrics, all_targets, all_probs


@torch.no_grad()
def tta_validation(
    dataloader: DataLoader,
    model: MultiHeadCNNForClassification,
    device: str,
    epoch: int,
    use_amp: bool,
    class_names: Dict[str, Dict[str, int]],
    weights: Optional[List[float]] = None,
) -> Tuple:
    if weights is not None:
        tta_weights = torch.tensor(weights, dtype=torch.float32)
        tta_weights = tta_weights / tta_weights.sum()
        tta_weights = tta_weights.view(-1, 1, 1).to(device)
    else:
        tta_weights = torch.zeros((1,), dtype=torch.float32).to(device)

    model.eval()

    all_targets = defaultdict(list)
    all_preds = defaultdict(list)
    all_probs = defaultdict(list)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val TTA]", leave=False)

    for inputs in pbar:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.autocast(device_type=device, enabled=use_amp):
            tta_outputs = model.predict_tta(inputs["pixel_values"])

        aggregated = {}
        for key in tta_outputs[0].keys():
            tensors = [out[key] for out in tta_outputs]
            aggregated[key] = torch.mean(torch.stack(tensors) * tta_weights, dim=0)

        for head_name, logits in aggregated.items():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            targets = torch.argmax(inputs[head_name], dim=1)

            all_probs[head_name].append(probs.cpu().numpy())
            all_preds[head_name].append(preds.cpu().numpy())
            all_targets[head_name].append(targets.cpu().numpy())

        # if "birads" in aggregated and "pathology" not in aggregated:
        #     probs = torch.softmax(aggregated["cls"], dim=1)
        #     probs = torch.stack(
        #         [
        #             probs[:, [0, 3, 4]].sum(dim=1),
        #             probs[:, [1, 2]].sum(dim=1),
        #         ],
        #         dim=1,
        #     )
        #     preds = torch.argmax(probs, dim=1)

        #     mapping = torch.tensor([0, 1, 1, 0, 0]).to(targets.device)
        #     targets = torch.argmax(inputs["pathology"], dim=1)
        #     targets = mapping[targets]

        #     all_probs["pathology"].append(probs.cpu().numpy())
        #     all_preds["pathology"].append(preds.cpu().numpy())
        #     all_targets["pathology"].append(targets.cpu().numpy())

    metrics = compute_comprehensive_metrics(
        all_targets,
        all_preds,
        all_probs,
        class_names,
    )

    return metrics, all_targets, all_probs


def export_metrics():
    for experiment in sorted(
        BASE_DIR.glob("output/experiment-*"), key=lambda p: int(p.name.split("-")[1])
    ):
        print(experiment.name.upper())

        config: Dict[str, Any] = json.load(open(experiment / "config.json", "r"))

        os.environ["ALBUMENTATIONS_RESIZE"] = config["RESIZE_BACKBONE"]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        class_names = config["LABEL2ID"]
        head_names = list(class_names.keys())
        head_classes = [len(class_names[name]) for name in head_names]

        if "pathology" not in class_names:
            class_names.update({"pathology": {"MALIGNANT": 0, "BENIGN": 1}})

        model_config = MultiHeadCNNConfig.from_dict(config["MODELCONFIG"])
        model = MultiHeadCNNForClassification(model_config)
        model.to(device)

        load_checkpoint(
            Path(config["OUTPUT_DIR"]) / "best_f1.pth",
            model=model,
        )

        all_datasets = json.load(open("config/datasets.json", "r"))
        df_test = prepare_dataset(
            {k: v for k, v in all_datasets.items() if k in VALID_ON},
            "temp",
        )["test"]
        transform_val = A.Compose(
            [
                A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ]
        )
        dataloader_test = DataLoader(
            DatasetDataFrame(
                df_test,
                map=config["LABEL2ID"],
                transform=transform_val,
                preprocess=config["PREPROCESS"],
            ),
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

        test_metrics, test_targets, test_probs = basic_validation(
            dataloader_test,
            model,
            device,
            -1,
            config["AMP"],
            class_names,
        )

        print_metrics_summary(
            test_metrics,
            output=experiment,
            filename="summary-test.json",
        )


def to_csv(filename: str = "summary-test.json", outputname: str = "results.csv"):
    df = {
        "Experiment": [],
        "Train Datasets": [],
        "Frozen Heads": [],
        "Head": [],
        "Checkpoint": [],
        "F1 Score": [],
        "Accuracy": [],
        "ROC-AUC": [],
    }
    valid_heads = ["MARGIN", "SHAPE", "BIRADS", "PATHOLOGY", "MALIGNANCY"]
    for experiment in sorted(
        BASE_DIR.glob("output/experiment-*"), key=lambda p: int(p.name.split("-")[1])
    ):
        print(experiment.name.upper())

        config = json.load(open(experiment / "config.json"))
        results = json.load(open(experiment / filename))

        for head in valid_heads:
            if head in results:
                df["Head"].append(head)
                df["F1 Score"].append(results[head]["Overall Metrics"]["Weighted F1"])
                df["Accuracy"].append(results[head]["Overall Metrics"]["Accuracy"])
                df["ROC-AUC"].append(results[head]["Overall Metrics"]["ROC-AUC"])
                df["Experiment"].append(experiment.name)
                df["Train Datasets"].append("+".join(list(config["DATASETS"].keys())))
                df["Frozen Heads"].append(
                    "+".join(list(config.get("FREEZE_HEADS", [])))
                )
                df["Checkpoint"].append(
                    False if config.get("CHECKPOINT", None) is not None else True
                )
            else:
                df["Head"].append(head)
                df["F1 Score"].append(None)
                df["Accuracy"].append(None)
                df["ROC-AUC"].append(None)
                df["Experiment"].append(experiment.name)
                df["Train Datasets"].append("+".join(list(config["DATASETS"].keys())))
                df["Frozen Heads"].append(
                    "+".join(list(config.get("FREEZE_HEADS", [])))
                )
                df["Checkpoint"].append(
                    False if config.get("CHECKPOINT", "None") == "None" else True
                )

    pd.DataFrame(df).to_csv(outputname)


def export_tta_metrics():
    for experiment in sorted(
        BASE_DIR.glob("output/experiment-*"), key=lambda p: int(p.name.split("-")[1])
    ):
        print(experiment.name.upper())

        config: Dict[str, Any] = json.load(open(experiment / "config.json", "r"))

        os.environ["ALBUMENTATIONS_RESIZE"] = config["RESIZE_BACKBONE"]

        device = "cuda" if torch.cuda.is_available() else "cpu"

        class_names = config["LABEL2ID"]

        if "pathology" not in class_names:
            class_names.update({"pathology": {"MALIGNANT": 0, "BENIGN": 1}})

        model_config = MultiHeadCNNConfig.from_dict(config["MODELCONFIG"])
        model = MultiHeadCNNForClassification(model_config)
        model.to(device)
        load_checkpoint(
            Path(config["OUTPUT_DIR"]) / "best_f1.pth",
            model=model,
        )

        all_datasets = json.load(open("config/datasets.json", "r"))
        df_test = prepare_dataset(
            {k: v for k, v in all_datasets.items() if k in VALID_ON},
            "temp",
        )["test"]
        transform_val = A.Compose(
            [
                A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ]
        )
        dataloader_test = DataLoader(
            DatasetDataFrame(
                df_test,
                transform=transform_val,
                preprocess=config["PREPROCESS"],
                map=config["LABEL2ID"],
            ),
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

        tta_transforms = [
            T.Compose(
                [
                    T.Lambda(lambda x: x),
                ]
            ),
            T.Compose(
                [
                    T.RandomVerticalFlip(p=1.0),
                ]
            ),
            T.Compose(
                [
                    T.RandomHorizontalFlip(p=1.0),
                ]
            ),
        ]
        setattr(model, "tta_transform", tta_transforms)
        test_metrics, test_targets, test_probs = tta_validation(
            dataloader_test,
            model,
            device,
            -1,
            config["AMP"],
            class_names,
            weights=[2, 1, 1],
        )

        print_metrics_summary(
            test_metrics,
            output=experiment,
            filename="summary-test-tta.json",
        )


if __name__ == "__main__":
    export_metrics()
    to_csv()

    export_tta_metrics()
    to_csv("summary-test-tta.json", "results-tta.csv")

    df_test = pd.read_csv("results.csv")
    df_test_tta = pd.read_csv("results-tta.csv")

    delta = (
        df_test_tta[["F1 Score", "Accuracy", "ROC-AUC"]]
        - df_test[["F1 Score", "Accuracy", "ROC-AUC"]]
    )
    delta.to_csv("delta.csv")
