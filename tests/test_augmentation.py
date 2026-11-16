"""Unit tests for image augmentation utilities."""

import numpy as np
import pytest

from image_classifier.data.augmentation import (
    dilation_image,
    erosion_image,
    flip,
    gaussian_blur,
    rotate,
    salt_and_pepper_noise,
    translation_image,
)


def test_flip_and_rotate_preserve_shape():
    img = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)
    assert flip(img, vflip=True).shape == (28, 28)
    assert flip(img, hflip=True, vflip=True).shape == (28, 28)
    assert np.array_equal(flip(img, vflip=False, hflip=False), img)
    assert rotate(img, angle=45.0).shape == (28, 28)


def test_translation_works_for_grayscale_and_rgb():
    gray = np.zeros((28, 28), dtype=np.uint8)
    rgb = np.zeros((28, 28, 3), dtype=np.uint8)
    assert translation_image(gray, 2, 2).shape == (28, 28)
    assert translation_image(rgb, 2, 2).shape == (28, 28, 3)


def test_morphological_blur_and_noise():
    img = np.full((28, 28), 128, dtype=np.uint8)
    assert erosion_image(img, 3).shape == (28, 28)
    assert dilation_image(img, 3).shape == (28, 28)
    assert gaussian_blur(img, sigma=1.0).shape == (28, 28)
    noisy = salt_and_pepper_noise(img, amount=0.1, seed=42)
    assert noisy.dtype == np.uint8
    assert noisy.shape == (28, 28)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        flip(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        erosion_image(np.zeros((8, 8), dtype=np.uint8), kernel_size=0)
    with pytest.raises(ValueError):
        gaussian_blur(np.zeros((8, 8), dtype=np.uint8), kernel_size=4)
