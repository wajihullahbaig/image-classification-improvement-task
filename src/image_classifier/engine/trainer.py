"""Unified training and evaluation engine for MLP and RNN classifiers."""

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import TrainConfig
from .metrics import ClassificationMetrics, compute_classification_metrics


def set_seed(seed: int) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_model_inputs(x: torch.Tensor, arch: str, feature_type: str) -> torch.Tensor:
    """Reshape batch tensors to match the expected input shape for MLP or RNN."""
    arch = arch.lower()
    feature_type = feature_type.lower()

    if arch == "mlp":
        return x.view(x.size(0), -1)

    if arch == "rnn":
        if feature_type == "raw":
            # (B, 1, H, W) -> (B, H, W)
            return x.squeeze(1) if x.ndim == 4 else x
        if feature_type == "gist":
            # (B, D) -> (B, 1, D)
            return x.unsqueeze(1) if x.ndim == 2 else x

    raise ValueError(f"Unsupported arch={arch!r}, feature_type={feature_type!r}")


@dataclass
class TrainResult:
    """Container for trained model, epoch history, and final evaluation metrics."""

    model: nn.Module
    history: List[Dict[str, float]]
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    best_epoch: int


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    arch: str,
    feature_type: str,
    config: TrainConfig,
) -> ClassificationMetrics:
    """Evaluate a model with torch.no_grad() and compute full classification metrics."""
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for data, target in loader:
            inputs = format_model_inputs(data.to(device), arch, feature_type)
            targets = target.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.append(probs.cpu().numpy())

    prob_matrix = np.concatenate(all_probs, axis=0) if all_probs else None
    return compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_prob=prob_matrix,
        total_loss=total_loss,
        num_classes=config.num_classes,
        class_names=config.class_names,
    )


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    arch: str,
    feature_type: str,
    config: TrainConfig,
    device: torch.device,
    verbose: bool = True,
) -> TrainResult:
    """Train a classifier and return best-epoch model alongside rich metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=config.momentum
    )

    history: List[Dict[str, float]] = []
    best_val_f1 = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        train_targets: List[int] = []
        train_preds: List[int] = []

        for data, target in train_loader:
            inputs = format_model_inputs(data.to(device), arch, feature_type)
            targets = target.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * targets.size(0)
            preds = torch.argmax(logits, dim=1)
            train_targets.extend(targets.cpu().tolist())
            train_preds.extend(preds.cpu().tolist())

        sample_count = max(len(train_targets), 1)
        epoch_train_loss = running_loss / sample_count
        epoch_train_acc = float(
            np.mean(np.asarray(train_targets) == np.asarray(train_preds))
        )

        val_metrics = evaluate_model(
            model, test_loader, criterion, device, arch, feature_type, config
        )

        history_row = {
            "epoch": float(epoch),
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": float(val_metrics.loss or 0.0),
            "val_accuracy": val_metrics.accuracy,
            "val_f1_macro": val_metrics.f1_macro,
        }
        history.append(history_row)

        if val_metrics.f1_macro >= best_val_f1:
            best_val_f1 = val_metrics.f1_macro
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == config.epochs):
            print(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:5.2f}% | "
                f"Val Loss: {val_metrics.loss:.4f} Acc: {val_metrics.accuracy*100:5.2f}% "
                f"Macro-F1: {val_metrics.f1_macro:.4f}"
            )

    model.load_state_dict(best_state)
    final_train_metrics = evaluate_model(
        model, train_loader, criterion, device, arch, feature_type, config
    )
    final_test_metrics = evaluate_model(
        model, test_loader, criterion, device, arch, feature_type, config
    )
    return TrainResult(
        model=model,
        history=history,
        train_metrics=final_train_metrics,
        test_metrics=final_test_metrics,
        best_epoch=best_epoch,
    )
