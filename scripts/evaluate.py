"""CLI script for running inference on the unlabeled test image directory."""

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from image_classifier.config import ProjectPaths, TrainConfig
from image_classifier.data import GistTestDataset, RawImageTestDataset
from image_classifier.engine import format_model_inputs, get_device
from image_classifier.models import load_checkpoint

CHECKPOINT_MAP = {
    "raw_mlp": ("mlp", "raw", "raw_mlp.pt", "raw_mlp_test_labels.csv"),
    "gist_mlp": ("mlp", "gist", "gist_mlp.pt", "gist_mlp_test_labels.csv"),
    "raw_rnn": ("rnn", "raw", "raw_rnn.pt", "raw_rnn_test_labels.csv"),
    "gist_rnn": ("rnn", "gist", "gist_rnn.pt", "gist_rnn_test_labels.csv"),
}


def predict_test_set(name: str, paths: ProjectPaths, cfg: TrainConfig, device: torch.device) -> Path:
    """Run batched inference over the test dataset and export ordered predictions."""
    arch, feature_type, ckpt_name, out_csv = CHECKPOINT_MAP[name]
    ckpt_path = paths.models_dir / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {ckpt_path} not found. Run scripts/train.py first."
        )

    model, _ = load_checkpoint(ckpt_path, device)
    if feature_type == "gist":
        dataset = GistTestDataset(paths.gist_test_dir / "*.npy")
    else:
        tfms = [transforms.ToTensor()]
        if arch == "mlp":
            tfms.append(transforms.Lambda(lambda x: torch.flatten(x)))
        dataset = RawImageTestDataset(
            paths.test_images_dir / "*.png", transforms=transforms.Compose(tfms)
        )

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    rows: List[dict] = []

    with torch.no_grad():
        for file_paths, data in loader:
            inputs = format_model_inputs(data.to(device), arch, feature_type)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for raw_path, pred_label, conf in zip(
                file_paths, preds.cpu().tolist(), confs.cpu().tolist()
            ):
                clean_name = Path(raw_path).name.removesuffix(".npy")
                rows.append(
                    {
                        "file": clean_name,
                        "label": int(pred_label),
                        "confidence": round(float(conf), 4),
                    }
                )

    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.predictions_dir / out_csv
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[{name}] Wrote {len(rows)} predictions -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models on unlabeled test images.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CHECKPOINT_MAP.keys()) + ["all"],
        default=["all"],
    )
    args = parser.parse_args()

    selected = list(CHECKPOINT_MAP.keys()) if "all" in args.models else args.models
    paths = ProjectPaths()
    cfg = TrainConfig()
    device = get_device()

    for name in selected:
        predict_test_set(name, paths, cfg, device)


if __name__ == "__main__":
    main()
