"""CLI script to train and evaluate MLP and RNN models on Raw and GIST features."""

import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from image_classifier.config import ProjectPaths, TrainConfig
from image_classifier.data import (
    GistTrainDataset,
    RawImageTrainDataset,
    create_train_test_indices,
)
from image_classifier.engine import (
    ClassificationMetrics,
    compare_models_table,
    format_metrics_report,
    get_device,
    plot_confusion_matrix,
    plot_training_history,
    save_metrics_json,
    set_seed,
    train_and_evaluate,
)
from image_classifier.models import build_model, save_checkpoint

EXPERIMENTS = {
    "raw_mlp": ("mlp", "raw", "raw_mlp.pt"),
    "gist_mlp": ("mlp", "gist", "gist_mlp.pt"),
    "raw_rnn": ("rnn", "raw", "raw_rnn.pt"),
    "gist_rnn": ("rnn", "gist", "gist_rnn.pt"),
}


def run_experiment(
    name: str, cfg: TrainConfig, paths: ProjectPaths, device: torch.device
) -> ClassificationMetrics:
    """Train, evaluate, and export artifacts for a single experiment."""
    arch, feature_type, ckpt_name = EXPERIMENTS[name]
    set_seed(cfg.random_seed)

    if feature_type == "gist":
        dataset = GistTrainDataset(paths.gist_train_labels, paths.dataset_dir)
    else:
        tfms = [transforms.ToTensor()]
        if arch == "mlp":
            tfms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        dataset = RawImageTrainDataset(
            paths.augmented_train_labels,
            paths.dataset_dir,
            transforms=transforms.Compose(tfms),
        )

    train_idx, test_idx = create_train_test_indices(
        dataset.data,
        test_size=cfg.test_split,
        random_seed=cfg.random_seed,
        strategy=cfg.split_strategy,
    )
    train_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=SubsetRandomSampler(train_idx)
    )
    test_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=SubsetRandomSampler(test_idx)
    )

    model = build_model(arch, feature_type, cfg)
    print(f"\n>>> Running {name} (split={cfg.split_strategy}, train={len(train_idx)}, test={len(test_idx)})")
    result = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        arch=arch,
        feature_type=feature_type,
        config=cfg,
        device=device,
    )

    print(f"Best checkpoint epoch: {result.best_epoch}")
    print(format_metrics_report(result.test_metrics, title=f"{name} Test Metrics"))

    save_checkpoint(
        result.model,
        paths.models_dir / ckpt_name,
        arch=arch,
        feature_type=feature_type,
        config=cfg,
        metrics=result.test_metrics.to_dict(),
    )
    save_metrics_json(result.test_metrics, paths.metrics_dir / f"{name}_metrics.json")
    plot_confusion_matrix(
        result.test_metrics,
        paths.metrics_dir / f"{name}_confusion_matrix.png",
        title=f"{name} Confusion Matrix ({cfg.split_strategy})",
    )
    plot_training_history(
        result.history,
        paths.metrics_dir / f"{name}_training_curves.png",
        title=f"{name} Training History",
    )
    return result.test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate image classifiers.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--split-strategy",
        choices=["group_stratified", "stratified"],
        default="group_stratified",
    )
    args = parser.parse_args()

    selected = list(EXPERIMENTS.keys()) if "all" in args.models else args.models
    cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        split_strategy=args.split_strategy,
    )
    paths = ProjectPaths()
    device = get_device()
    print(f"PyTorch {torch.__version__} | Device: {device}")

    results: Dict[str, ClassificationMetrics] = {}
    for exp_name in selected:
        results[exp_name] = run_experiment(exp_name, cfg, paths, device)

    summary_df = compare_models_table(results)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = paths.metrics_dir / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nModel Comparison Summary ({summary_path}):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
