"""PyTorch Dataset classes and leak-free train/test split utilities."""

import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset

_AUG_PREFIX_PATTERN = re.compile(
    r"^(?:dilated_|eroded_|translated_|gausian_blur_|salt_and_pepper_noise_|aug_)+"
)


def extract_source_group_id(file_path: str) -> str:
    """Recover the base image identifier so augmented variants share one group."""
    stem = Path(file_path).name
    if stem.endswith(".npy"):
        stem = stem[:-4]
    return _AUG_PREFIX_PATTERN.sub("", stem)


class GistTrainDataset(Dataset):
    """Dataset loading precomputed 1D GIST feature vectors and integer labels."""

    def __init__(self, labels_file: Path | str, root_dir: Path | str) -> None:
        self.data = pd.read_csv(labels_file)
        self.root_dir = Path(root_dir)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = int(idx.item())
        feature_path = self.root_dir / str(self.data["file_path"].iloc[idx])
        if not feature_path.exists():
            raise FileNotFoundError(f"GIST feature file not found: {feature_path}")
        feature = torch.from_numpy(np.load(feature_path)).float()
        label = int(self.data["label"].iloc[idx])
        return feature, label


class RawImageTrainDataset(Dataset):
    """Dataset loading normalized grayscale images and integer labels."""

    def __init__(
        self,
        labels_file: Path | str,
        root_dir: Path | str,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.data = pd.read_csv(labels_file)
        self.root_dir = Path(root_dir)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor | np.ndarray, int]:
        if torch.is_tensor(idx):
            idx = int(idx.item())
        image_path = self.root_dir / str(self.data["file_path"].iloc[idx])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
        image = np.float32(image / 255.0)
        if self.transforms is not None:
            image = self.transforms(image)
        label = int(self.data["label"].iloc[idx])
        return image, label


class RawImageTestDataset(Dataset):
    """Dataset loading unlabeled test images in deterministic sorted order."""

    def __init__(
        self, pattern: str | Path, transforms: Optional[Callable] = None
    ) -> None:
        path = Path(pattern)
        if path.is_dir():
            self.files = sorted(path.glob("*.png"))
        else:
            self.files = sorted(path.parent.glob(path.name))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor | np.ndarray]:
        if torch.is_tensor(idx):
            idx = int(idx.item())
        image_path = self.files[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")
        image = np.float32(image / 255.0)
        if self.transforms is not None:
            image = self.transforms(image)
        return str(image_path), image


class GistTestDataset(Dataset):
    """Dataset loading unlabeled test GIST feature vectors in sorted order."""

    def __init__(self, pattern: str | Path) -> None:
        path = Path(pattern)
        if path.is_dir():
            self.files = sorted(path.glob("*.npy"))
        else:
            self.files = sorted(path.parent.glob(path.name))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())
        feature_path = self.files[idx]
        feature = torch.from_numpy(np.load(feature_path)).float()
        return str(feature_path), feature


def create_train_test_indices(
    labels_df: pd.DataFrame,
    test_size: float = 0.3,
    random_seed: int = 42,
    strategy: str = "group_stratified",
) -> Tuple[List[int], List[int]]:
    """Split dataset indices using stratified or group-stratified splitting.

    Args:
        labels_df: DataFrame containing ``file_path`` and ``label`` columns.
        test_size: Target fraction of samples in the test split.
        random_seed: Random seed for reproducibility.
        strategy: ``"group_stratified"`` prevents augmented siblings from leaking
            across train and test splits; ``"stratified"`` splits at sample level.
    """
    indices = np.arange(len(labels_df))
    labels = labels_df["label"].astype(int).to_numpy()

    if strategy == "stratified":
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels,
        )
        return train_idx.tolist(), test_idx.tolist()

    if strategy == "group_stratified":
        groups = labels_df["file_path"].astype(str).apply(extract_source_group_id).to_numpy()
        n_splits = max(2, int(round(1.0 / test_size)))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_seed
        )
        train_idx, test_idx = next(splitter.split(indices, labels, groups))
        return train_idx.tolist(), test_idx.tolist()

    raise ValueError(f"Unknown split strategy: {strategy!r}")
