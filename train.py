import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict
from typing import Tuple, Dict, Any, List, Optional, Literal

import cv2
import torch
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
from torch.utils.data import DataLoader

sys.path.append("/run/media/adminteam/kde/vida/src")

from masscls.data import DatasetDataFrame
from masscls.loss import MultiTaskLoss, TaskSpecificFocalLoss
from masscls.models import MultiHeadCNNForClassification, MultiHeadCNNConfig
from masscls.utils import (
    prepare_dataset,
    compute_comprehensive_metrics,
    plot_training_curves,
    save_checkpoint,
    save_calibration_plots,
    save_confusion_matrices,
    save_roc_curves,
    load_checkpoint,
    print_metrics_summary,
    compute_all_class_weights,
)

random.seed(43876)
np.random.seed(43876)
torch.manual_seed(43876)
torch.cuda.manual_seed(43876)
torch.cuda.manual_seed_all(43876)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

BASE_DIR = Path(__file__).resolve().parent

idx = 1
OUTPUT_DIR = BASE_DIR / "output" / f"experiment-{idx}"
while True:
    if OUTPUT_DIR.exists():
        idx += 1
        OUTPUT_DIR = BASE_DIR / "output" / f"experiment-{idx}"
    else:
        OUTPUT_DIR.mkdir(parents=True)
        break

RESIZE_BACKBONE = "pillow"  # opencv
os.environ["ALBUMENTATIONS_RESIZE"] = RESIZE_BACKBONE

matplotlib.use("Agg")


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""

    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_lr_by_name(optimizer, name):
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return group["lr"]
    raise ValueError(f"No param group named {name}")


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_function: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    scaler: torch.GradScaler,
    use_amp: bool,
    gradient_steps: int,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    optimizer.zero_grad()

    for batch_idx, inputs in enumerate(pbar):
        # Move to device
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.autocast(device_type=device, enabled=use_amp):
            outputs = model(inputs["pixel_values"])

            losses = loss_function(logits=outputs, targets=inputs)
            loss = losses / gradient_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_steps == 0:
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_steps
        pbar.set_postfix({"loss": f"{loss.item() * gradient_steps:.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    use_amp: bool,
    label2id: Dict[str, Dict[str, int]],
) -> Tuple:
    """Validate for one epoch"""

    model.eval()
    total_loss = 0.0

    all_targets = defaultdict(list)
    all_preds = defaultdict(list)
    all_probs = defaultdict(list)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for inputs in pbar:
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.amp.autocast(device_type=device, enabled=use_amp):  # type: ignore
            logits = model(inputs["pixel_values"])
            losses = loss_fn(logits=logits, targets=inputs)

        total_loss += losses.item()

        for head_name, logits in logits.items():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            targets = torch.argmax(inputs[head_name], dim=1)

            all_probs[head_name].append(probs.cpu().numpy())
            all_preds[head_name].append(preds.cpu().numpy())
            all_targets[head_name].append(targets.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_comprehensive_metrics(all_targets, all_preds, all_probs, label2id)

    return avg_loss, metrics, all_targets, all_probs


def train(
    epochs,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    model: MultiHeadCNNForClassification,
    loss_function: MultiTaskLoss,
    device: str,
    scaler: torch.GradScaler,
    use_amp: bool,
    label2id: Dict[str, Dict[str, int]],
    patience: int,
):
    early_stopping = EarlyStopping(patience=patience, mode="min")

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": {k["name"]: [] for k in optimizer.param_groups},
        "metrics": [],
    }
    best_val_loss = float("inf")
    best_val_f1 = 0.0

    for epoch in range(epochs):
        for name in history["lr"].keys():
            history["lr"][name].append(get_lr_by_name(optimizer, name))

        train_loss = train_epoch(
            dataloader=dataloader_train,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
            gradient_steps=4,
            grad_clip_norm=1.0,
        )

        history["train_loss"].append(train_loss)

        val_loss, val_metrics, all_targets, all_probs = validate_epoch(
            dataloader=dataloader_val,
            model=model,
            loss_fn=loss_function,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            label2id=label2id,
        )

        scheduler.step()

        history["val_loss"].append(val_loss)
        history["metrics"].append(val_metrics)

        plot_training_curves(history, OUTPUT_DIR / f"training_curves.png", label2id)

        avg_f1 = np.mean([m["weighted_f1"] for m in val_metrics.values()])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                OUTPUT_DIR / f"best_loss.pth",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
            )

        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            save_checkpoint(
                model,
                OUTPUT_DIR / f"best_f1.pth",
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
            )

        if (epoch + 1) % 50 == 0:
            save_confusion_matrices(val_metrics, epoch, OUTPUT_DIR, label2id)
            save_roc_curves(val_metrics, epoch, OUTPUT_DIR, all_targets, all_probs)
            save_calibration_plots(
                val_metrics, epoch, OUTPUT_DIR, all_targets, all_probs
            )

        if early_stopping(val_loss):
            break

        # if (epoch + 1) % 10 == 0:
        #     save_checkpoint(
        #         model,
        #         optimizer,
        #         scheduler,
        #         epoch,
        #         val_metrics,
        #         OUTPUT_DIR / f"checkpoint_epoch{epoch+1}.pth",
        #     )

    save_checkpoint(
        model,
        OUTPUT_DIR / f"final_model.pth",
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics=val_metrics,
    )

    history_path = OUTPUT_DIR / f"training_history.json"
    with open(history_path, "w") as f:
        history_save = {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "lr": history["lr"],
        }
        json.dump(history_save, f, indent=4)


def evaluate(
    model: MultiHeadCNNForClassification,
    loss_function: MultiTaskLoss,
    dataloader: DataLoader,
    use_amp: bool,
    label2id: Dict[str, Dict[str, int]],
    device: str,
):
    load_checkpoint(
        OUTPUT_DIR / f"best_f1.pth",
        model,
    )

    test_loss, test_metrics, test_targets, test_probs = validate_epoch(
        dataloader=dataloader,
        loss_fn=loss_function,
        model=model,
        epoch=-1,
        device=device,
        use_amp=use_amp,
        label2id=label2id,
    )

    print_metrics_summary(test_metrics, output=OUTPUT_DIR)
    save_confusion_matrices(test_metrics, -2, OUTPUT_DIR, label2id)
    save_roc_curves(test_metrics, -2, OUTPUT_DIR, test_targets, test_probs)
    save_calibration_plots(test_metrics, -2, OUTPUT_DIR, test_targets, test_probs)


def main(
    epochs: int,
    data_source: str,
    datasets: Dict[str, Dict[str, Any]],
    use_amp: bool,
    label2id_json: str,
    loss_args: str,
    freeze_heads: List[str] = [],
    checkpoint: Optional[str] = None,
    preprocess: Literal["iss", "normal"] = "iss",
    batch_size: int = 32,
    num_workers: int = 8,
    lr0: float = 1e-5,
    lrf: float = 1e-8,
    freeze_backbone: bool = True,
    freeze_attention: bool = False,
    patience: int = 100,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dfs = prepare_dataset(
        datasets=datasets,
        output=data_source,
    )

    label2id: Dict[str, Dict[str, int]] = json.load(open(label2id_json, "r"))

    n_heads = len(label2id.keys())

    class_weights = compute_all_class_weights(
        dfs["train"],
        label2id,
        smoothing=0.1,
    )

    lossargs = json.load(open(loss_args, "r"))

    config = MultiHeadCNNConfig(
        head_dims=[[1024, 512, 256, 128]] * n_heads,
        head_names={k: len(label2id[k]) for k in label2id},
        class_weights=class_weights,
        gamma=lossargs["gamma"],
        smoothing=lossargs["smoothing"],
        task_weights={
            **{
                k: v
                for k, v in lossargs["task_weights"].items()
                if k not in freeze_heads
            },
            **{
                k: 0.0 for k, v in lossargs["task_weights"].items() if k in freeze_heads
            },
        },
    )
    model = MultiHeadCNNForClassification(config=config)
    model.to(device)

    if checkpoint is not None:
        load_checkpoint(
            checkpoint,
            model=model,
        )
        print(f"Model Loaded from `{checkpoint}` Successfully!")

    for name in freeze_heads:
        for params in model.heads[name].parameters():
            params.requires_grad = False

    total_params = [
        *[
            {
                "name": head,
                "params": model.heads[head].parameters(),
                "lr": lr0,
            }
            for head in label2id
            if head not in freeze_heads
        ],
    ]

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        total_params.append(
            {
                "name": "backbone",
                "params": model.backbone.parameters(),
                "lr": lr0 * 1e-1,
            }
        )

    if freeze_attention:
        for param in model.attention.parameters():
            param.requires_grad = False
    else:
        total_params.append(
            {
                "name": "attention",
                "params": model.attention.parameters(),
                "lr": lr0 * 1e-1,
            }
        )

    optimizer = torch.optim.AdamW(
        params=total_params,
        lr=lr0,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
    )

    for name in freeze_heads:
        for n, p in model.heads[name].named_parameters():
            assert not p.requires_grad, f"{name}.{n} is still trainable!"

    opt_params = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_params.add(p)

    for name in freeze_heads:
        for p in model.heads[name].parameters():
            assert p not in opt_params, f"{name} is inside optimizer!"

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=lrf,
    )
    scaler = torch.GradScaler(device=device, enabled=use_amp)

    train_dataloader = DataLoader(
        DatasetDataFrame(
            dfs["train"],
            label2id,
            transform=A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(1 - 0.08, 1 + 0.08),
                        translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                        rotate=(-10, 10),
                        shear=(0, 0),
                        interpolation=cv2.INTER_CUBIC,
                        border_mode=cv2.BORDER_CONSTANT,
                        fit_output=False,
                        keep_ratio=True,
                        p=0.3,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.4,
                    ),
                    A.GaussNoise(
                        std_range=(0.05, 0.2),
                        mean_range=(0, 0),
                        p=0.2,
                    ),
                    A.CoarseDropout(num_holes_range=(1, 3), p=0.15),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.ToTensorV2(),
                ]
            ),
            preprocess=preprocess,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        # drop_last=True,
    )
    eval_dataloader = DataLoader(
        DatasetDataFrame(
            dfs["val"],
            map=label2id,
            transform=A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.ToTensorV2(),
                ]
            ),
            preprocess=preprocess,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    loss_function = MultiTaskLoss(
        task_losses={
            key: TaskSpecificFocalLoss(
                alpha=(torch.tensor(config.class_weights[key])),
                gamma=config.gamma[key],
                label_smoothing=config.smoothing[key],
            )
            for key in config.head_names
        },
        task_weights=config.task_weights,
        learnable_weights=False,
    )
    loss_function.to(device)

    json.dump(
        {
            "OUTPUT_DIR": str(OUTPUT_DIR),
            "EPOCHS": epochs,
            "LR0": lr0,
            "LRF": lrf,
            "BATCH_SIZE": batch_size,
            "AMP": use_amp,
            "LABEL2ID": label2id,
            "DATASETS": datasets,
            "MODELCONFIG": config.to_dict(),
            "PREPROCESS": preprocess,
            "RESIZE_BACKBONE": os.environ.get("ALBUMENTATIONS_RESIZE"),
            "FREEZE_HEADS": freeze_heads,
            "CHECKPOINT": checkpoint,
            "FREEZE_ATTENTION": freeze_attention,
            "FREEZE_BACKBONE": freeze_backbone,
            "GAMMA": config.gamma,
            "SMOOTHING": config.smoothing,
            "TASKWEIGHTS": config.task_weights,
        },
        open(OUTPUT_DIR / "config.json", "w"),
        indent=4,
    )

    train(
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_train=train_dataloader,
        dataloader_val=eval_dataloader,
        model=model,
        loss_function=loss_function,
        device=device,
        scaler=scaler,
        use_amp=use_amp,
        label2id=label2id,
        patience=patience,
    )

    test_dataloader = DataLoader(
        DatasetDataFrame(
            dfs["test"],
            map=label2id,
            transform=A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.ToTensorV2(),
                ]
            ),
            preprocess=preprocess,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    evaluate(
        model=model,
        dataloader=test_dataloader,
        loss_function=loss_function,
        use_amp=use_amp,
        label2id=label2id,
        device=device,
    )


if __name__ == "__main__":
    # cbis NOTE: shape, margin, birads, pathology, malignancy
    # cdd-cesm NOTE: shape, margin, birads, pathology, malignancy
    # vida NOTE: shape, margin, birads, malignancy
    # csaw NOTE: pathology
    # vindr NOTE: birads, malignancy

    all_datasets = json.load(open("config/datasets.json", "r"))

    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=460, required=False)
    parser.add_argument("--data", type=str, default="temp", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--label2id", type=str, default="config/label2id.json")

    parser.add_argument("--freeze-heads", required=False, nargs="+", default=[])
    parser.add_argument("--checkpoint", required=False, type=str)
    parser.add_argument("--preprocess", type=str, default="iss", required=False)
    parser.add_argument("--batch-size", type=int, default=32, required=False)
    parser.add_argument("--num-workers", type=int, default=4, required=False)
    parser.add_argument("--only-prepare-data", action="store_true")

    parser.add_argument("--lr0", type=float, default=1e-5, required=False)
    parser.add_argument("--lrf", type=float, default=1e-8, required=False)

    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-attention", action="store_true")

    parser.add_argument("--loss", type=str, default="config/loss.json")
    parser.add_argument("--patience", type=int, default=100)

    args = parser.parse_args()

    start_dt = datetime.now()
    if args.only_prepare_data:
        prepare_dataset(
            {k: v for k, v in all_datasets.items() if k in args.datasets},
            output=args.data,
        )
    else:
        main(
            epochs=args.epochs,
            data_source=args.data,
            datasets={k: v for k, v in all_datasets.items() if k in args.datasets},
            use_amp=args.amp,
            label2id_json=args.label2id,
            freeze_heads=args.freeze_heads,
            checkpoint=args.checkpoint,
            preprocess=args.preprocess,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr0=args.lr0,
            lrf=args.lrf,
            freeze_backbone=args.freeze_backbone,
            freeze_attention=args.freeze_attention,
            loss_args=args.loss,
            patience=args.patience,
        )
    end_dt = datetime.now()

    elapsed = end_dt - start_dt
    total_seconds = int(elapsed.total_seconds())

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    start_str = start_dt.strftime("%a %d %b, From %H:%M:%S")
    end_str = end_dt.strftime("to %H:%M:%S")

    print(f"{start_str} {end_str}, Elapsed time {hours:02}:{minutes:02}:{seconds:02}")
