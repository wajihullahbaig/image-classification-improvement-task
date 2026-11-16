"""Comprehensive classification evaluation metrics, reporting, and plotting."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class PerClassMetric:
    """Metrics for an individual target class."""

    class_id: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.recall


@dataclass
class ClassificationMetrics:
    """Full suite of multi-class classification evaluation metrics."""

    num_samples: int
    loss: Optional[float]
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    cohen_kappa: float
    matthews_corrcoef: float
    roc_auc_ovr_macro: Optional[float]
    confusion_matrix: List[List[int]]
    per_class: List[PerClassMetric]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return {
            "num_samples": self.num_samples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "cohen_kappa": self.cohen_kappa,
            "matthews_corrcoef": self.matthews_corrcoef,
            "roc_auc_ovr_macro": self.roc_auc_ovr_macro,
            "confusion_matrix": self.confusion_matrix,
            "per_class": [asdict(item) for item in self.per_class],
        }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray] = None,
    total_loss: Optional[float] = None,
    num_classes: int = 4,
    class_names: Optional[Sequence[str]] = None,
) -> ClassificationMetrics:
    """Compute aggregate, per-class, and probabilistic classification metrics."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics on an empty target array.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    labels = list(range(num_classes))
    names = (
        list(class_names)
        if class_names is not None
        else [f"Class {idx}" for idx in labels]
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0
    )

    roc_auc: Optional[float] = None
    if y_prob is not None and len(np.unique(y_true_arr)) == num_classes:
        try:
            roc_auc = float(
                roc_auc_score(
                    y_true_arr, y_prob, multi_class="ovr", average="macro", labels=labels
                )
            )
        except ValueError:
            roc_auc = None

    per_class = [
        PerClassMetric(
            class_id=idx,
            class_name=names[idx],
            precision=float(prec[idx]),
            recall=float(rec[idx]),
            f1=float(f1[idx]),
            support=int(supp[idx]),
            correct=int(cm[idx, idx]),
        )
        for idx in labels
    ]

    avg_loss = (
        float(total_loss / len(y_true_arr)) if total_loss is not None else None
    )

    return ClassificationMetrics(
        num_samples=int(len(y_true_arr)),
        loss=avg_loss,
        accuracy=float(accuracy_score(y_true_arr, y_pred_arr)),
        balanced_accuracy=float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        precision_macro=float(prec_macro),
        recall_macro=float(rec_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(prec_w),
        recall_weighted=float(rec_w),
        f1_weighted=float(f1_w),
        cohen_kappa=float(cohen_kappa_score(y_true_arr, y_pred_arr, labels=labels)),
        matthews_corrcoef=float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        roc_auc_ovr_macro=roc_auc,
        confusion_matrix=cm.tolist(),
        per_class=per_class,
    )


def format_metrics_report(metrics: ClassificationMetrics, title: str = "Evaluation") -> str:
    """Format a human-readable multi-line report of classification metrics."""
    lines = [f"=== {title} (n={metrics.num_samples}) ==="]
    if metrics.loss is not None:
        lines.append(f"Loss              : {metrics.loss:.6f}")
    lines.extend(
        [
            f"Accuracy          : {metrics.accuracy * 100:.2f}%",
            f"Balanced Accuracy : {metrics.balanced_accuracy * 100:.2f}%",
            f"Macro F1          : {metrics.f1_macro:.4f} "
            f"(P={metrics.precision_macro:.4f}, R={metrics.recall_macro:.4f})",
            f"Weighted F1       : {metrics.f1_weighted:.4f} "
            f"(P={metrics.precision_weighted:.4f}, R={metrics.recall_weighted:.4f})",
            f"Cohen's Kappa     : {metrics.cohen_kappa:.4f}",
            f"Matthews CorrCoef : {metrics.matthews_corrcoef:.4f}",
        ]
    )
    if metrics.roc_auc_ovr_macro is not None:
        lines.append(f"ROC-AUC (OvR)     : {metrics.roc_auc_ovr_macro:.4f}")

    lines.append("\nPer-Class Breakdown:")
    lines.append(f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Correct/Supp':>14}")
    for item in metrics.per_class:
        lines.append(
            f"{item.class_name:<10} {item.precision:>8.4f} {item.recall:>8.4f} "
            f"{item.f1:>8.4f} {f'{item.correct}/{item.support}':>14}"
        )
    return "\n".join(lines)


def save_metrics_json(metrics: ClassificationMetrics, output_path: Path) -> None:
    """Persist metrics dictionary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics.to_dict(), handle, indent=2)


def compare_models_table(results: Dict[str, ClassificationMetrics]) -> pd.DataFrame:
    """Build a summary comparison DataFrame across multiple evaluated models."""
    rows = []
    for model_name, metric in results.items():
        rows.append(
            {
                "model": model_name,
                "samples": metric.num_samples,
                "loss": metric.loss,
                "accuracy": round(metric.accuracy * 100.0, 2),
                "balanced_accuracy": round(metric.balanced_accuracy * 100.0, 2),
                "macro_precision": round(metric.precision_macro, 4),
                "macro_recall": round(metric.recall_macro, 4),
                "macro_f1": round(metric.f1_macro, 4),
                "weighted_f1": round(metric.f1_weighted, 4),
                "cohen_kappa": round(metric.cohen_kappa, 4),
                "mcc": round(metric.matthews_corrcoef, 4),
                "roc_auc_ovr": (
                    round(metric.roc_auc_ovr_macro, 4)
                    if metric.roc_auc_ovr_macro is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Render and save a confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics.confusion_matrix, dtype=int)
    labels = [item.class_name for item in metrics.per_class]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_history(
    history: List[Dict[str, float]],
    output_path: Path,
    title: str = "Training & Validation Curves",
) -> None:
    """Plot epoch-level train/eval loss and accuracy/F1 curves."""
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train Loss")
    if "val_loss" in history[0]:
        axes[0].plot(epochs, [row["val_loss"] for row in history], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs, [row["train_accuracy"] * 100.0 for row in history], label="Train Acc (%)"
    )
    if "val_accuracy" in history[0]:
        axes[1].plot(
            epochs, [row["val_accuracy"] * 100.0 for row in history], label="Val Acc (%)"
        )
    if "val_f1_macro" in history[0]:
        axes[1].plot(
            epochs, [row["val_f1_macro"] * 100.0 for row in history], label="Val Macro-F1 (%)"
        )
    axes[1].set_title("Accuracy & Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
