"""Unit tests for classification metrics and leak-free dataset splitting."""

import numpy as np
import pandas as pd

from image_classifier.config import ProjectPaths
from image_classifier.data.dataset import create_train_test_indices, extract_source_group_id
from image_classifier.engine.metrics import (
    compare_models_table,
    compute_classification_metrics,
    format_metrics_report,
)


def test_project_paths_resolves_repo_root():
    paths = ProjectPaths()
    assert (paths.root / "pyproject.toml").exists()
    assert paths.dataset_dir.name == "dataset"


def test_compute_classification_metrics_exact_values():
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 0, 1, 2, 2, 2, 3, 3]
    y_prob = np.eye(4)[y_pred]

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        total_loss=4.0,
        num_classes=4,
    )
    assert metrics.num_samples == 8
    assert metrics.loss == 0.5
    assert metrics.accuracy == 0.875
    assert metrics.balanced_accuracy == 0.875
    assert metrics.confusion_matrix[1][2] == 1
    assert metrics.roc_auc_ovr_macro is not None
    assert "Macro F1" in format_metrics_report(metrics, "UnitTest")

    table = compare_models_table({"test_model": metrics})
    assert list(table["model"]) == ["test_model"]
    assert float(table["accuracy"].iloc[0]) == 87.5


def test_group_stratified_split_prevents_augmentation_leakage():
    assert extract_source_group_id("train_augumented2/dilated_aug_img_012.png") == "img_012.png"
    assert extract_source_group_id("gist_features/eroded_img_005.png.npy") == "img_005.png"

    rows = []
    for idx in range(20):
        label = idx % 4
        base = f"img_{idx:03d}.png"
        for prefix in ["", "dilated_", "eroded_", "translated_"]:
            rows.append({"file_path": f"train_augumented2/{prefix}{base}", "label": label})
    df = pd.DataFrame(rows)

    train_idx, test_idx = create_train_test_indices(
        df, test_size=0.25, random_seed=42, strategy="group_stratified"
    )
    train_groups = set(df.iloc[train_idx]["file_path"].apply(extract_source_group_id))
    test_groups = set(df.iloc[test_idx]["file_path"].apply(extract_source_group_id))
    assert train_groups.isdisjoint(test_groups)
