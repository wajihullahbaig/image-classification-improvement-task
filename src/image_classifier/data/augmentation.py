"""Image augmentation utilities for grayscale and RGB images."""

from typing import Optional

import cv2
import numpy as np
from skimage.util import random_noise


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate that the input image is a non-empty NumPy array."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Expected a non-empty numpy.ndarray image.")
    return image


def flip(image: np.ndarray, vflip: bool = False, hflip: bool = False) -> np.ndarray:
    """Flip an image vertically, horizontally, or both."""
    image = _validate_image(image)
    if not (hflip or vflip):
        return image.copy()
    if hflip and vflip:
        flip_code = -1
    else:
        flip_code = 0 if vflip else 1
    return cv2.flip(image, flipCode=flip_code)


def rotate(image: np.ndarray, angle: float = 90.0, scale: float = 1.0) -> np.ndarray:
    """Rotate an image counter-clockwise around its center."""
    image = _validate_image(image)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    return cv2.warpAffine(image, matrix, (width, height))


def gaussian_blur(image: np.ndarray, sigma: float = 1.0, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    image = _validate_image(image)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def averaging_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply normalized box-filter blur to an image."""
    image = _validate_image(image)
    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")
    return cv2.blur(image, (kernel_size, kernel_size))


def median_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median blur to an image."""
    image = _validate_image(image)
    if kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than 1.")
    return cv2.medianBlur(image, kernel_size)


def bilateral_blur(
    image: np.ndarray, diameter: int = 5, sigma_color: float = 75.0, sigma_space: float = 75.0
) -> np.ndarray:
    """Apply bilateral filtering to smooth while preserving edges."""
    image = _validate_image(image)
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def erosion_image(image: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    """Apply morphological erosion with a square kernel."""
    image = _validate_image(image)
    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def dilation_image(image: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    """Apply morphological dilation with a square kernel."""
    image = _validate_image(image)
    if kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def translation_image(image: np.ndarray, x: float, y: float) -> np.ndarray:
    """Translate an image by (x, y) pixels."""
    image = _validate_image(image)
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image, matrix, (cols, rows))


def salt_and_pepper_noise(
    image: np.ndarray, amount: float = 0.05, seed: Optional[int] = None
) -> np.ndarray:
    """Add salt-and-pepper noise to an image and return uint8 in [0, 255]."""
    image = _validate_image(image)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be in [0.0, 1.0].")
    rng = np.random.default_rng(seed)
    noisy = random_noise(image, mode="s&p", amount=amount, rng=rng)
    return np.clip(np.round(255.0 * noisy), 0, 255).astype(np.uint8)
