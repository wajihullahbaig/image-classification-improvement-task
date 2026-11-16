"""Training, evaluation, and metrics reporting engine."""

from .metrics import (
    ClassificationMetrics,
    PerClassMetric,
    compare_models_table,
    compute_classification_metrics,
    format_metrics_report,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics_json,
)
from .trainer import TrainResult, evaluate_model, format_model_inputs, get_device, set_seed, train_and_evaluate

__all__ = [
    "ClassificationMetrics",
    "PerClassMetric",
    "TrainResult",
    "compare_models_table",
    "compute_classification_metrics",
    "evaluate_model",
    "format_metrics_report",
    "format_model_inputs",
    "get_device",
    "plot_confusion_matrix",
    "plot_training_history",
    "save_metrics_json",
    "set_seed",
    "train_and_evaluate",
]
