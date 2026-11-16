"""Dataset class-balancing and multi-transform augmentation pipeline."""

import os
import shutil
from pathlib import Path
from typing import Dict

import cv2
import pandas as pd

from . import augmentation as aug


def summarize_class_distribution(labels_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute per-class counts, differences from max class, and proportions."""
    counts = labels_df["label"].value_counts().sort_index()
    diffs = counts.max() - counts
    percentages = counts / counts.sum()
    return {"counts": counts, "diffs": diffs, "percentages": percentages}


def balance_classes(
    labels_csv: Path,
    dataset_root: Path,
    output_subdir: str = "train_augumented",
    output_csv: Path | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Upsample minority classes with vertical flips to equalize class counts."""
    labels_df = pd.read_csv(labels_csv)
    output_dir = dataset_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = labels_df["label"].value_counts()
    diffs = counts.max() - counts

    new_rows = []
    for label_val, deficit in diffs.items():
        if deficit <= 0:
            continue
        candidates = labels_df[labels_df["label"] == label_val]
        replace = deficit > len(candidates)
        samples = candidates.sample(n=int(deficit), replace=replace, random_state=random_seed)
        for _, sample in samples.iterrows():
            src_path = dataset_root / sample["file_path"]
            img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {src_path}")

            flipped = aug.flip(img, vflip=True, hflip=False)
            out_name = f"aug_{Path(sample['file_path']).name}"
            cv2.imwrite(str(output_dir / out_name), flipped)
            new_rows.append({"file_path": f"{output_subdir}/{out_name}", "label": int(sample["label"])})

    # Copy original training images into the balanced directory
    for rel_path in labels_df["file_path"]:
        src = dataset_root / rel_path
        dst = output_dir / Path(rel_path).name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    updated_df = labels_df.copy()
    updated_df["file_path"] = updated_df["file_path"].apply(
        lambda p: f"{output_subdir}/{Path(p).name}"
    )
    if new_rows:
        updated_df = pd.concat([updated_df, pd.DataFrame(new_rows)], ignore_index=True)

    if output_csv is not None:
        updated_df.to_csv(output_csv, index=False)
    return updated_df


def generate_augmented_dataset(
    labels_csv: Path,
    dataset_root: Path,
    output_subdir: str = "train_augumented2",
    output_csv: Path | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Expand a balanced dataset 6x using morphological, spatial, blur, and noise transforms."""
    labels_df = pd.read_csv(labels_csv)
    output_dir = dataset_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, row in labels_df.iterrows():
        src_path = dataset_root / row["file_path"]
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {src_path}")

        base_name = os.path.basename(row["file_path"])
        label = int(row["label"])

        # Copy original image
        cv2.imwrite(str(output_dir / base_name), img)
        rows.append({"file_path": f"{output_subdir}/{base_name}", "label": label})

        transforms = {
            f"dilated_{base_name}": aug.dilation_image(img, kernel_size=1),
            f"eroded_{base_name}": aug.erosion_image(img, kernel_size=1),
            f"translated_{base_name}": aug.translation_image(img, 2, 2),
            f"gausian_blur_{base_name}": aug.gaussian_blur(img, sigma=1.0),
            f"salt_and_pepper_noise_{base_name}": aug.salt_and_pepper_noise(
                img, amount=0.05, seed=seed + int(idx)
            ),
        }
        for out_name, transformed in transforms.items():
            cv2.imwrite(str(output_dir / out_name), transformed)
            rows.append({"file_path": f"{output_subdir}/{out_name}", "label": label})

    augmented_df = pd.DataFrame(rows)
    if output_csv is not None:
        augmented_df.to_csv(output_csv, index=False)
    return augmented_df
