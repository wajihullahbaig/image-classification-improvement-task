"""CLI script for extracting GIST descriptors for training and test sets."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import pandas as pd

from image_classifier.config import ProjectPaths, TrainConfig
from image_classifier.data import gist


def extract_train_gist(paths: ProjectPaths, cfg: TrainConfig) -> None:
    """Extract and save GIST features for all rows in the augmented training CSV."""
    labels_df = pd.read_csv(paths.augmented_train_labels)
    gist_df = labels_df.copy()
    paths.gist_train_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in labels_df.iterrows():
        img_path = paths.dataset_dir / str(row["file_path"])
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        feat = gist(
            img,
            image_size=cfg.image_size,
            orientations=cfg.gist_orientations,
            num_blocks=cfg.gist_num_blocks,
        )
        out_name = f"{img_path.name}.npy"
        np.save(paths.gist_train_dir / out_name, feat)
        gist_df.at[idx, "file_path"] = f"gist_features/{out_name}"

    gist_df.to_csv(paths.gist_train_labels, index=False)
    print(f"Saved {len(gist_df)} training GIST descriptors -> {paths.gist_train_labels}")


def extract_test_gist(paths: ProjectPaths, cfg: TrainConfig) -> None:
    """Extract and save GIST features for unlabeled test images."""
    paths.gist_test_dir.mkdir(parents=True, exist_ok=True)
    test_files = sorted(paths.test_images_dir.glob("*.png"))
    for img_path in test_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load test image: {img_path}")
        feat = gist(
            img,
            image_size=cfg.image_size,
            orientations=cfg.gist_orientations,
            num_blocks=cfg.gist_num_blocks,
        )
        np.save(paths.gist_test_dir / f"{img_path.name}.npy", feat)
    print(f"Saved {len(test_files)} test GIST descriptors -> {paths.gist_test_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GIST feature vectors.")
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which split to extract GIST features for.",
    )
    args = parser.parse_args()

    paths = ProjectPaths()
    cfg = TrainConfig()
    if args.split in {"train", "all"}:
        extract_train_gist(paths, cfg)
    if args.split in {"test", "all"}:
        extract_test_gist(paths, cfg)


if __name__ == "__main__":
    main()
