import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_all_class_weights(
    df: pd.DataFrame,
    class_names: Dict,
    smoothing: float = 0.1,
) -> Dict[str, List[float]]:
    """
    Compute class weights for all tasks
    """
    class_weights = {}

    for task in list(class_names.keys()):
        col_name = {
            "shape": "shape",
            "margin": "margin",
            "birads": "birads",
            "pathology": "pathology",
            "malignancy": "malignancy",
        }[task]

        labels = df[col_name].values

        label_indices = np.array([class_names[task][str(label)] for label in labels])

        # Compute weights
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(list(class_names[task].values())),
            y=label_indices,
        )

        # Apply smoothing to prevent extreme weights
        if smoothing > 0:
            weights = weights * (1 - smoothing) + smoothing

        # Clip weights
        weights = np.clip(weights, 0.3, 5.0)

        class_weights[task] = weights.tolist()

    return class_weights


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: Dict,
    filepath: Path,
):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": {
            # Convert to serializable format
            head: {
                "accuracy": m["accuracy"],
                "weighted_f1": m["weighted_f1"],
                "roc_auc": m.get("roc_auc", 0.0),
            }
            for head, m in metrics.items()
        },
    }

    torch.save(checkpoint, filepath)


def plot_training_curves(
    history: Dict,
    output_path: Path,
    class_names: Dict[str, Dict[str, int]],
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss curves
    axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Learning rate
    for name in history["lr"].keys():
        axes[0, 1].plot(history["lr"][name], label=name, linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Learning Rate")
    axes[0, 1].set_title("Learning Rate Schedule")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # F1 scores per task
    for head_name in list(class_names.keys()):
        f1_scores = [m[head_name]["weighted_f1"] for m in history["metrics"]]
        axes[0, 2].plot(f1_scores, label=head_name, linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Weighted F1")
    axes[0, 2].set_title("F1 Scores per Task")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    # Accuracy per task
    for head_name in list(class_names.keys()):
        accuracies = [m[head_name]["accuracy"] for m in history["metrics"]]
        axes[1, 0].plot(accuracies, label=head_name, linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy per Task")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # ROC-AUC per task
    for head_name in list(class_names.keys()):
        aucs = [m[head_name].get("roc_auc", 0.0) for m in history["metrics"]]
        axes[1, 1].plot(aucs, label=head_name, linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("ROC-AUC")
    axes[1, 1].set_title("ROC-AUC per Task")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    # Calibration ECE per task
    for head_name in list(class_names.keys()):
        eces = [m[head_name]["calibration"]["ece"] for m in history["metrics"]]
        axes[1, 2].plot(eces, label=head_name, linewidth=2)
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("ECE")
    axes[1, 2].set_title("Calibration Error per Task")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint(
    filepath: Union[Path, str],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> int:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]

    return epoch


def compute_comprehensive_metrics(
    all_targets: Dict[str, List[np.ndarray]],
    all_preds: Dict[str, List[np.ndarray]],
    all_probs: Dict[str, List[np.ndarray]],
    class_names: Dict[str, Dict[str, int]],
) -> Dict:
    """
    Compute comprehensive metrics for all tasks

    Returns nested dictionary with all metrics
    """
    metrics = {}

    for head_name in all_targets.keys():
        idxtolabel = np.vectorize({v: k for k, v in class_names[head_name].items()}.get)

        y_true_original = np.concatenate(all_targets[head_name])
        y_pred_original = np.concatenate(all_preds[head_name])
        y_probs_original = np.concatenate(all_probs[head_name])

        y_true = idxtolabel(y_true_original)
        y_pred = idxtolabel(y_pred_original)

        task_metrics = {}

        # Basic classification metrics
        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
            labels=list(class_names[head_name].keys()),
        )
        task_metrics["classification_report"] = report
        task_metrics["accuracy"] = report["accuracy"]  # type: ignore
        task_metrics["macro_f1"] = report["macro avg"]["f1-score"]  # type: ignore
        task_metrics["weighted_f1"] = report["weighted avg"]["f1-score"]  # type: ignore

        # Per-class metrics
        task_metrics["per_class"] = compute_per_class_metrics(
            y_true_original,
            y_pred_original,
            y_probs_original,
            class_names.get(head_name) if class_names else None,
        )

        # Confusion matrix
        cm = confusion_matrix(
            y_true, y_pred, labels=list(class_names[head_name].keys())
        )
        task_metrics["confusion_matrix"] = cm

        # Clinical metrics (for binary tasks or overall)
        if len(np.unique(y_true_original)) == 2:
            task_metrics["clinical"] = compute_clinical_metrics(
                y_true_original, y_pred_original, y_probs_original
            )

        # ROC-AUC
        try:
            if y_probs_original.shape[1] == 2:
                # Binary classification
                auc = roc_auc_score(y_true_original, y_probs_original[:, 1])
                task_metrics["roc_auc"] = auc
            else:
                # Multi-class (one-vs-rest)
                auc_ovr = roc_auc_score(
                    y_true_original,
                    y_probs_original,
                    multi_class="ovr",
                    average="weighted",
                )
                auc_ovo = roc_auc_score(
                    y_true_original,
                    y_probs_original,
                    multi_class="ovo",
                    average="weighted",
                )
                task_metrics["roc_auc_ovr"] = auc_ovr
                task_metrics["roc_auc_ovo"] = auc_ovo
                task_metrics["roc_auc"] = auc_ovr  # Default to OvR
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC for {head_name}: {e}")
            task_metrics["roc_auc"] = 0.0

        # Calibration metrics
        task_metrics["calibration"] = compute_calibration_metrics(
            y_true_original, y_probs_original
        )

        # Confidence statistics
        task_metrics["confidence"] = compute_confidence_stats(
            y_probs_original, y_true_original, y_pred_original
        )

        metrics[head_name] = task_metrics

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: Optional[Dict[str, int]] = None,
) -> Dict:
    """Compute detailed metrics for each class"""
    unique_classes = np.unique(y_true)
    per_class = {}

    class_names = {v: k for k, v in class_names.items()}  # type: ignore
    for cls in unique_classes:
        class_name = class_names[cls] if class_names else f"Class_{cls}"

        # Binary metrics for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        # True/False Positives/Negatives
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Support
        support = np.sum(y_true == cls)

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "support": int(support),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    return per_class


def compute_clinical_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
) -> Dict:
    """
    Compute clinical metrics for binary classification
    - Sensitivity (Recall)
    - Specificity
    - PPV (Precision)
    - NPV
    - Balanced Accuracy
    """
    # Assume positive class is 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": balanced_acc,
    }


def compute_calibration_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
) -> Dict:
    """
    Compute calibration metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier Score
    """
    # Get predicted probabilities for the true class
    max_probs = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    correct = (predictions == y_true).astype(float)

    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get predictions in this bin
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(max_probs[in_bin])

            # ECE: weighted average of |accuracy - confidence|
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            # MCE: maximum |accuracy - confidence|
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

    # Brier score (for multi-class, average over all classes)
    # One-hot encode true labels
    n_classes = y_probs.shape[1]
    y_true_onehot = np.zeros((len(y_true), n_classes))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))

    return {"ece": float(ece), "mce": float(mce), "brier_score": float(brier)}


def compute_confidence_stats(
    y_probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> Dict:
    """Compute statistics about prediction confidence"""
    max_probs = np.max(y_probs, axis=1)
    correct = y_pred == y_true

    return {
        "mean_confidence": float(np.mean(max_probs)),
        "std_confidence": float(np.std(max_probs)),
        "mean_confidence_correct": float(np.mean(max_probs[correct])),
        "mean_confidence_incorrect": (
            float(np.mean(max_probs[~correct])) if np.sum(~correct) > 0 else 0.0
        ),
        "low_confidence_ratio": float(np.mean(max_probs < 0.5)),
        "high_confidence_ratio": float(np.mean(max_probs > 0.9)),
    }


def save_confusion_matrices(
    metrics: Dict,
    epoch: int,
    output_dir: Path,
    class_names: Dict[str, Dict[str, int]],
):
    """Save confusion matrix plots with better formatting"""
    n_tasks = len(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, (head_name, head_metrics) in enumerate(metrics.items()):
        if idx >= 4:
            break

        cm = head_metrics["confusion_matrix"]

        # Get class names if available
        if class_names and head_name in class_names:
            labels = list(class_names[head_name].keys())
        else:
            labels = [f"C{i}" for i in range(cm.shape[0])]

        # Normalize confusion matrix for better visualization
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Proportion"},
        )

        # Add counts as text
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(
                    j + 0.5,
                    i + 0.7,
                    f"({cm[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="gray",
                )

        # Title with metrics
        acc = head_metrics["accuracy"]
        f1 = head_metrics["weighted_f1"]
        axes[idx].set_title(
            f"{head_name.upper()} (Epoch {epoch+1})\n" f"Acc: {acc:.3f}, F1: {f1:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].set_ylabel("True Label", fontsize=10)
        axes[idx].set_xlabel("Predicted Label", fontsize=10)

    # Hide unused subplots
    for idx in range(len(metrics), 4):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"confusion_matrices_epoch_{epoch+1}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_roc_curves(
    metrics: Dict, epoch: int, output_dir: Path, all_targets: Dict, all_probs: Dict
):
    """Save ROC curve plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, (head_name, head_metrics) in enumerate(metrics.items()):
        if idx >= 4:
            break

        y_true = np.concatenate(all_targets[head_name])
        y_probs = np.concatenate(all_probs[head_name])

        ax = axes[idx]

        if y_probs.shape[1] == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            auc = head_metrics.get("roc_auc", 0.0)

            ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
            ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
        else:
            # Multi-class: plot one curve per class
            for cls in range(y_probs.shape[1]):
                y_true_binary = (y_true == cls).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, cls])
                auc = roc_auc_score(y_true_binary, y_probs[:, cls])
                ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {auc:.3f})", linewidth=2)

            ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)

        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(
            f"{head_name.upper()} ROC Curve (Epoch {epoch+1})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(metrics), 4):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"roc_curves_epoch_{epoch+1}.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


def save_calibration_plots(
    metrics: Dict,
    epoch: int,
    output_dir: Path,
    all_targets: Dict,
    all_probs: Dict,
    n_bins: int = 10,
):
    """Save calibration reliability diagrams"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()

    for idx, (head_name, head_metrics) in enumerate(metrics.items()):
        if idx >= 4:
            break

        y_true = np.concatenate(all_targets[head_name])
        y_probs = np.concatenate(all_probs[head_name])

        max_probs = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        correct = (predictions == y_true).astype(float)

        # Bin predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies = []
        confidences = []
        counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(correct[in_bin])
                avg_confidence_in_bin = np.mean(max_probs[in_bin])
                count_in_bin = np.sum(in_bin)

                accuracies.append(accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
                counts.append(count_in_bin)

        ax = axes[idx]

        # Plot reliability diagram
        ax.bar(
            range(len(confidences)),
            accuracies,
            width=0.8,
            alpha=0.7,
            label="Accuracy",
            edgecolor="black",
        )
        ax.plot(
            range(len(confidences)),
            confidences,
            "ro-",
            linewidth=2,
            label="Confidence",
            markersize=8,
        )
        ax.plot(
            [0, len(confidences) - 1],
            [0, 1],
            "k--",
            alpha=0.5,
            label="Perfect Calibration",
        )

        # Annotations
        ece = head_metrics["calibration"]["ece"]
        mce = head_metrics["calibration"]["mce"]

        ax.set_xlabel("Confidence Bin", fontsize=10)
        ax.set_ylabel("Accuracy / Confidence", fontsize=10)
        ax.set_title(
            f"{head_name.upper()} Calibration (Epoch {epoch+1})\n"
            f"ECE: {ece:.3f}, MCE: {mce:.3f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    # Hide unused subplots
    for idx in range(len(metrics), 4):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"calibration_plots_epoch_{epoch+1}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def print_metrics_summary(
    metrics: Dict,
    epoch: Optional[int] = None,
    output: Optional[Path] = None,
    filename: Optional[str] = None,
):
    lines = []
    output_dict = {}

    header = f"\n{'='*80}\n"
    if epoch is not None:
        header += f"EPOCH {epoch+1} - METRICS SUMMARY\n"
    else:
        header += "METRICS SUMMARY\n"
    header += f"{'='*80}\n"

    lines.append(header)

    for head_name, head_metrics in metrics.items():
        lines.append(f"\n{head_name.upper()}:")
        lines.append(f"  Overall Metrics:")
        lines.append(f"    Accuracy:     {head_metrics['accuracy']:.4f}")
        lines.append(f"    Macro F1:     {head_metrics['macro_f1']:.4f}")
        lines.append(f"    Weighted F1:  {head_metrics['weighted_f1']:.4f}")
        lines.append(f"    ROC-AUC:      {head_metrics.get('roc_auc', 0):.4f}")

        if "clinical" in head_metrics:
            lines.append(f"  Clinical Metrics:")
            clinical = head_metrics["clinical"]
            lines.append(f"    Sensitivity:  {clinical['sensitivity']:.4f}")
            lines.append(f"    Specificity:  {clinical['specificity']:.4f}")
            lines.append(f"    PPV:          {clinical['ppv']:.4f}")
            lines.append(f"    NPV:          {clinical['npv']:.4f}")
            clinical_dict = {
                "Clinical Metrics": {
                    "Sensitivity": clinical["sensitivity"],
                    "Specificity": clinical["specificity"],
                    "PPV": clinical["ppv"],
                    "NPV": clinical["npv"],
                }
            }
        else:
            clinical_dict = {}

        lines.append(f"  Calibration:")
        cal = head_metrics["calibration"]
        lines.append(f"    ECE:          {cal['ece']:.4f}")
        lines.append(f"    MCE:          {cal['mce']:.4f}")
        lines.append(f"    Brier Score:  {cal['brier_score']:.4f}")

        lines.append(f"  Confidence:")
        conf = head_metrics["confidence"]
        lines.append(f"    Mean:         {conf['mean_confidence']:.4f}")
        lines.append(f"    Std:          {conf['std_confidence']:.4f}")
        lines.append(f"    Low (<0.5):   {conf['low_confidence_ratio']:.2%}")

        output_dict[head_name.upper()] = {
            "Overall Metrics": {
                "Accuracy": head_metrics["accuracy"],
                "Macro F1": head_metrics["macro_f1"],
                "Weighted F1": head_metrics["weighted_f1"],
                "ROC-AUC": head_metrics.get("roc_auc", 0),
            },
            "Calibration": {
                "ECE": cal["ece"],
                "MCE": cal["mce"],
                "Brier Score": cal["brier_score"],
            },
            "Confidence": {
                "Mean": conf["mean_confidence"],
                "Std": conf["std_confidence"],
                "Low (<0.5)": conf["low_confidence_ratio"],
            },
            **clinical_dict,
        }

    lines.append(f"\n{'='*80}\n")

    summary_text = "\n".join(lines)

    print(summary_text)

    if output is not None:
        if filename is None:
            fname = output / "summary.txt"
        else:
            fname = output / filename

        if fname.suffix == ".txt":
            with fname.open("a", encoding="utf-8") as f:
                f.write(summary_text)
        elif fname.suffix == ".json":
            json.dump(
                output_dict,
                fname.open("w"),
                indent=4,
            )
