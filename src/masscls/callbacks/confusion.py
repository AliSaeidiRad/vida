import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TrainerCallback
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from masscls.utils.metrics import (
    compute_clinical_metrics,
    compute_calibration_metrics,
    compute_confidence_stats,
    compute_per_class_metrics,
    print_metrics_summary,
)


class ConfusionMatrixCallback(TrainerCallback):
    def __init__(self, class_names: Dict[str, Dict[str, int]]) -> None:
        super(ConfusionMatrixCallback, self).__init__()
        self.class_names = class_names

    def on_evaluate(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        output_dir = args.output_dir
        device = args.device

        epoch = kwargs.get("epoch")

        model = kwargs.get("model")
        eval_dataloader = kwargs.get("eval_dataloader")
        head_names = model.config.head_names  # type: ignore

        all_targets = defaultdict(list)
        all_preds = defaultdict(list)
        all_probs = defaultdict(list)

        for batch in eval_dataloader:  # type: ignore
            pixel_values = batch["pixel_values"].to(device)
            labels = {
                k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()
            }
            with torch.no_grad():
                outputs = model(pixel_values)["logits"]  # type: ignore

        for head_name, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            targets = torch.argmax(labels[head_name], dim=1)

            all_probs[head_name].append(probs.cpu().detach().numpy())
            all_preds[head_name].append(preds.cpu().detach().numpy())
            all_targets[head_name].append(targets.cpu().detach().numpy())

        metrics = self.compute_comprehensive_metrics(all_targets, all_preds, all_probs)
        self.save_confusion_matrix(
            head_names,
            metrics,
            epoch if epoch is not None else -1,
            output_dir if output_dir is not None else ".",
        )

    def compute_comprehensive_metrics(
        self,
        all_targets: Dict[str, List[np.ndarray]],
        all_preds: Dict[str, List[np.ndarray]],
        all_probs: Dict[str, List[np.ndarray]],
    ) -> Dict:
        """
        Compute comprehensive metrics for all tasks

        Returns nested dictionary with all metrics
        """
        metrics = {}

        for head_name in all_targets.keys():
            # Concatenate all batches
            y_true = np.concatenate(all_targets[head_name])
            y_pred = np.concatenate(all_preds[head_name])
            y_probs = np.concatenate(all_probs[head_name])

            task_metrics = {}

            # Basic classification metrics
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            task_metrics["classification_report"] = report
            task_metrics["accuracy"] = report["accuracy"]  # type: ignore
            task_metrics["macro_f1"] = report["macro avg"]["f1-score"]  # type: ignore
            task_metrics["weighted_f1"] = report["weighted avg"]["f1-score"]  # type: ignore

            # Per-class metrics
            task_metrics["per_class"] = compute_per_class_metrics(
                y_true,
                y_pred,
                y_probs,
                self.class_names[head_name],
            )

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            task_metrics["confusion_matrix"] = cm

            # Clinical metrics (for binary tasks or overall)
            if len(np.unique(y_true)) == 2:
                task_metrics["clinical"] = compute_clinical_metrics(
                    y_true, y_pred, y_probs
                )

            # ROC-AUC
            try:
                if y_probs.shape[1] == 2:
                    # Binary classification
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                    task_metrics["roc_auc"] = auc
                else:
                    # Multi-class (one-vs-rest)
                    auc_ovr = roc_auc_score(
                        y_true, y_probs, multi_class="ovr", average="weighted"
                    )
                    auc_ovo = roc_auc_score(
                        y_true, y_probs, multi_class="ovo", average="weighted"
                    )
                    task_metrics["roc_auc_ovr"] = auc_ovr
                    task_metrics["roc_auc_ovo"] = auc_ovo
                    task_metrics["roc_auc"] = auc_ovr  # Default to OvR
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC for {head_name}: {e}")
                task_metrics["roc_auc"] = 0.0

            # Calibration metrics
            task_metrics["calibration"] = compute_calibration_metrics(y_true, y_probs)

            # Confidence statistics
            task_metrics["confidence"] = compute_confidence_stats(
                y_probs, y_true, y_pred
            )

            metrics[head_name] = task_metrics

        # print_metrics_summary(metrics)
        return metrics

    @staticmethod
    def save_confusion_matrix(
        labels: Dict[str, List[str]],
        metrics: Dict,
        epoch: float,
        output_dir: str,
    ):
        fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 4, 5))
        fig.tight_layout()
        axes = axes.ravel()

        for idx, (head_name, head_metrics) in enumerate(metrics.items()):
            cm = head_metrics["confusion_matrix"]

            label = labels[head_name]

            # Normalize confusion matrix for better visualization
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            ax = sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                ax=axes[idx],
                xticklabels=label,
                yticklabels=label,
                cbar_kws={"label": "Proportion"},
                annot_kws={"size": 6},
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

            # Add counts as text
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[idx].text(
                        j + 0.5,
                        i + 0.7,
                        f"({cm[i, j]})",
                        ha="center",
                        va="center",
                        fontsize=4,
                        color="gray",
                    )

            # Title with metrics
            acc = head_metrics["accuracy"]
            f1 = head_metrics["weighted_f1"]
            axes[idx].set_title(
                f"{head_name.upper()} (Epoch {epoch+1})\n"
                f"Acc: {acc:.3f}, F1: {f1:.3f}",
                fontsize=4,
                fontweight="bold",
            )
            axes[idx].set_ylabel(
                "True Label",
                fontsize=5,
            )
            axes[idx].set_xlabel(
                "Predicted Label",
                fontsize=5,
            )

        # Hide unused subplots
        for idx in range(len(metrics), 4):
            axes[idx].axis("off")

        plt.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"confusion_matrices_epoch_{epoch+1}.png"),
            dpi=400,
        )
        plt.close(fig)
