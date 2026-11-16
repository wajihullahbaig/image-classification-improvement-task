"""Unit tests for GIST feature descriptor extraction."""

import numpy as np
import pytest

from image_classifier.data.gist import create_gabor, gist


def test_gist_descriptor_shape_and_finiteness():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(28, 28), dtype=np.uint8)
    feat = gist(img, image_size=28, orientations=(8, 8, 4), num_blocks=4)
    assert feat.shape == (320,)
    assert np.all(np.isfinite(feat))


def test_gist_accepts_color_image():
    rgb = np.full((32, 32, 3), 100, dtype=np.uint8)
    feat = gist(rgb, image_size=28)
    assert feat.shape == (320,)


def test_create_gabor_is_cached():
    g1 = create_gabor((8, 8, 4), 28)
    g2 = create_gabor((8, 8, 4), 28)
    assert g1 is g2
    assert g1.shape == (28, 28, 20)


def test_gist_invalid_inputs():
    with pytest.raises(ValueError):
        gist(None, image_size=28)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        gist(np.zeros((28, 28), dtype=np.uint8), image_size=27)
