"""CLI script for class balancing and dataset augmentation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from image_classifier.config import ProjectPaths, TrainConfig
from image_classifier.data import (
    balance_classes,
    generate_augmented_dataset,
    summarize_class_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance classes and generate augmented images.")
    parser.add_argument("--seed", type=int, default=TrainConfig().random_seed)
    args = parser.parse_args()

    paths = ProjectPaths()
    raw_df = pd.read_csv(paths.raw_train_labels)
    raw_summary = summarize_class_distribution(raw_df)
    print("Original class counts:\n", raw_summary["counts"].to_string())

    balanced_df = balance_classes(
        labels_csv=paths.raw_train_labels,
        dataset_root=paths.dataset_dir,
        output_subdir="train_augumented",
        output_csv=paths.balanced_train_labels,
        random_seed=args.seed,
    )
    bal_summary = summarize_class_distribution(balanced_df)
    print("\nBalanced class counts:\n", bal_summary["counts"].to_string())

    aug_df = generate_augmented_dataset(
        labels_csv=paths.balanced_train_labels,
        dataset_root=paths.dataset_dir,
        output_subdir="train_augumented2",
        output_csv=paths.augmented_train_labels,
        seed=args.seed,
    )
    print(f"\nGenerated {len(aug_df)} augmented training records -> {paths.augmented_train_labels}")


if __name__ == "__main__":
    main()
