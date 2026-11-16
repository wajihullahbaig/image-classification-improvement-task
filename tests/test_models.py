"""Unit tests for MLP and RNN models and checkpoint serialization."""

import torch

from image_classifier.config import TrainConfig
from image_classifier.models import MLP, RNN, build_model, load_checkpoint, save_checkpoint


def test_mlp_outputs_four_class_logits():
    model = MLP(input_length=320, hidden_dim=64, num_classes=4, dropout=0.2)
    x = torch.randn(5, 320)
    out = model(x)
    assert out.shape == (5, 4)


def test_rnn_raw_and_gist_forward_shapes():
    cfg = TrainConfig()
    raw_rnn = build_model("rnn", "raw", cfg)
    gist_rnn = build_model("rnn", "gist", cfg)

    assert raw_rnn(torch.randn(6, 28, 28)).shape == (6, 4)
    assert gist_rnn(torch.randn(6, 1, 320)).shape == (6, 4)


def test_single_sample_batch_works():
    cfg = TrainConfig()
    mlp = build_model("mlp", "raw", cfg)
    mlp.eval()
    assert mlp(torch.randn(1, 784)).shape == (1, 4)


def test_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig()
    model = build_model("mlp", "gist", cfg)
    ckpt = tmp_path / "test_mlp.pt"
    save_checkpoint(model, ckpt, "mlp", "gist", cfg, metrics={"accuracy": 0.9})

    loaded, payload = load_checkpoint(ckpt, torch.device("cpu"))
    assert isinstance(loaded, MLP)
    assert payload["metrics"]["accuracy"] == 0.9

    sample = torch.randn(2, 320)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(sample), loaded(sample))
