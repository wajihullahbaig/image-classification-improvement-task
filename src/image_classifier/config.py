"""Centralized configuration, paths, and training hyperparameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ProjectPaths:
    """Repository directory and file layout."""

    root: Path = Path(__file__).resolve().parents[2]

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def raw_train_labels(self) -> Path:
        return self.dataset_dir / "train_labels.csv"

    @property
    def balanced_train_labels(self) -> Path:
        return self.dataset_dir / "train_augumented_labels.csv"

    @property
    def augmented_train_labels(self) -> Path:
        return self.dataset_dir / "train_augumented2_labels.csv"

    @property
    def gist_train_labels(self) -> Path:
        return self.dataset_dir / "train_gist_augumented2_labels.csv"

    @property
    def test_images_dir(self) -> Path:
        return self.dataset_dir / "test"

    @property
    def gist_train_dir(self) -> Path:
        return self.dataset_dir / "gist_features"

    @property
    def gist_test_dir(self) -> Path:
        return self.dataset_dir / "gist_test_features"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def predictions_dir(self) -> Path:
        return self.outputs_dir / "predictions"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"


@dataclass
class TrainConfig:
    """Hyperparameters and feature extraction settings."""

    num_classes: int = 4
    class_names: Tuple[str, ...] = ("Class 0", "Class 1", "Class 2", "Class 3")
    image_size: int = 28
    gist_orientations: Tuple[int, ...] = (8, 8, 4)
    gist_num_blocks: int = 4
    batch_size: int = 32
    test_split: float = 0.3
    random_seed: int = 42
    epochs: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.5
    mlp_hidden_dim: int = 512
    mlp_dropout: float = 0.2
    rnn_hidden_size: int = 128
    rnn_num_layers: int = 2
    split_strategy: str = "group_stratified"  # "group_stratified" or "stratified"
    extra_model_kwargs: dict = field(default_factory=dict)

    @property
    def gist_feature_dim(self) -> int:
        return sum(self.gist_orientations) * (self.gist_num_blocks ** 2)

    @property
    def raw_feature_dim(self) -> int:
        return self.image_size * self.image_size
