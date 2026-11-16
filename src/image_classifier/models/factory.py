"""Model factory and checkpoint persistence helpers."""

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ..config import TrainConfig
from .mlp import MLP
from .rnn import RNN


def build_model(
    arch: str,
    feature_type: str,
    config: TrainConfig,
    input_length: int | None = None,
) -> nn.Module:
    """Instantiate an MLP or RNN for either raw pixels or GIST vectors."""
    arch = arch.lower()
    feature_type = feature_type.lower()

    if feature_type not in {"raw", "gist"}:
        raise ValueError(f"Unsupported feature_type: {feature_type!r}")

    if arch == "mlp":
        dim = input_length or (
            config.raw_feature_dim if feature_type == "raw" else config.gist_feature_dim
        )
        return MLP(
            input_length=dim,
            hidden_dim=config.mlp_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.mlp_dropout,
        )

    if arch == "rnn":
        if feature_type == "raw":
            input_size = config.image_size
            sequence_length = config.image_size
        else:
            input_size = input_length or config.gist_feature_dim
            sequence_length = 1
        return RNN(
            input_size=input_size,
            hidden_size=config.rnn_hidden_size,
            sequence_length=sequence_length,
            num_layers=config.rnn_num_layers,
            num_classes=config.num_classes,
        )

    raise ValueError(f"Unsupported model architecture: {arch!r}")


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    arch: str,
    feature_type: str,
    config: TrainConfig,
    metrics: Dict[str, Any] | None = None,
) -> None:
    """Save model state_dict and metadata cleanly without pickling module paths."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "feature_type": feature_type,
        "state_dict": model.state_dict(),
        "config": {
            "num_classes": config.num_classes,
            "image_size": config.image_size,
            "gist_feature_dim": config.gist_feature_dim,
            "raw_feature_dim": config.raw_feature_dim,
            "mlp_hidden_dim": config.mlp_hidden_dim,
            "mlp_dropout": config.mlp_dropout,
            "rnn_hidden_size": config.rnn_hidden_size,
            "rnn_num_layers": config.rnn_num_layers,
        },
        "metrics": metrics or {},
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a saved checkpoint (state_dict dict) onto the requested device."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(payload, nn.Module):
        payload.to(device)
        payload.eval()
        return payload, {}

    cfg_dict = payload["config"]
    cfg = TrainConfig(
        num_classes=cfg_dict["num_classes"],
        image_size=cfg_dict["image_size"],
        mlp_hidden_dim=cfg_dict["mlp_hidden_dim"],
        mlp_dropout=cfg_dict["mlp_dropout"],
        rnn_hidden_size=cfg_dict["rnn_hidden_size"],
        rnn_num_layers=cfg_dict["rnn_num_layers"],
    )
    input_length = (
        cfg_dict["raw_feature_dim"]
        if payload["feature_type"] == "raw"
        else cfg_dict["gist_feature_dim"]
    )
    model = build_model(
        payload["arch"],
        payload["feature_type"],
        cfg,
        input_length=input_length,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
