"""GIST global image descriptor extraction (Oliva & Torralba style)."""

from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np


@lru_cache(maxsize=16)
def create_gabor(orientations: Tuple[int, ...], image_size: int) -> np.ndarray:
    """Compute Gabor filter transfer functions in the frequency domain."""
    n_scales = len(orientations)
    num_filters = sum(orientations)

    params = []
    for scale_idx in range(n_scales):
        n_orient = orientations[scale_idx]
        for orient_idx in range(n_orient):
            params.append(
                [
                    0.35,
                    0.3 / (1.85 ** scale_idx),
                    16.0 * (n_orient ** 2) / (32.0 ** 2),
                    np.pi / n_orient * orient_idx,
                ]
            )
    param_arr = np.asarray(params, dtype=np.float64)

    half = image_size / 2.0
    fx, fy = np.meshgrid(np.arange(-half, half), np.arange(-half, half))
    fr = np.fft.fftshift(np.sqrt(fx ** 2 + fy ** 2))
    theta = np.fft.fftshift(np.angle(fx + 1j * fy))

    gabors = np.zeros((image_size, image_size, num_filters), dtype=np.float64)
    for idx in range(num_filters):
        tr = theta + param_arr[idx, 3]
        tr = tr + 2.0 * np.pi * (tr < -np.pi) - 2.0 * np.pi * (tr > np.pi)
        gabors[:, :, idx] = np.exp(
            -10.0 * param_arr[idx, 0] * (fr / image_size / param_arr[idx, 1] - 1.0) ** 2
            - 2.0 * param_arr[idx, 2] * np.pi * (tr ** 2)
        )
    return gabors


def _prefilter(img: np.ndarray, fc: float = 4.0, pad_width: int = 5) -> np.ndarray:
    """Apply log transform, FFT whitening, and local contrast normalization."""
    s1 = fc / np.sqrt(np.log(2.0))

    img = np.log1p(np.maximum(img, 0.0))
    img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width)), mode="symmetric")
    sn, sm = img.shape
    n = max(sn, sm)
    n = n + (n % 2)
    img = np.pad(img, ((0, n - sn), (0, n - sm)), mode="symmetric")

    half = n / 2.0
    fx, fy = np.meshgrid(np.arange(-half, half), np.arange(-half, half))
    gf = np.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (s1 ** 2)))

    output = img - np.real(np.fft.ifft2(np.fft.fft2(img) * gf))
    local_var = np.fft.ifft2(np.fft.fft2(output ** 2) * gf)
    local_std = np.sqrt(np.abs(local_var))
    output /= 0.2 + local_std

    return output[pad_width : sn - pad_width, pad_width : sm - pad_width]


def _block_average(feature_map: np.ndarray, num_blocks: int) -> np.ndarray:
    """Average a 2D response map over a (num_blocks x num_blocks) spatial grid."""
    nx = np.fix(np.linspace(0, feature_map.shape[0], num_blocks + 1)).astype(int)
    ny = np.fix(np.linspace(0, feature_map.shape[1], num_blocks + 1)).astype(int)
    pooled = np.zeros((num_blocks, num_blocks), dtype=np.float64)
    for xx in range(num_blocks):
        for yy in range(num_blocks):
            pooled[yy, xx] = np.mean(feature_map[nx[xx] : nx[xx + 1], ny[yy] : ny[yy + 1]])
    return pooled


def _extract_gabor_features(
    prefiltered: np.ndarray, num_blocks: int, gabors: np.ndarray
) -> np.ndarray:
    """Convolve prefiltered image with Gabor bank and pool spatial blocks."""
    _, _, num_filters = gabors.shape
    block_area = num_blocks * num_blocks
    descriptor = np.zeros(block_area * num_filters, dtype=np.float64)

    img_fft = np.fft.fft2(prefiltered)
    offset = 0
    for filter_idx in range(num_filters):
        response = np.abs(np.fft.ifft2(img_fft * gabors[:, :, filter_idx]))
        pooled = _block_average(response, num_blocks)
        descriptor[offset : offset + block_area] = pooled.reshape(block_area)
        offset += block_area
    return descriptor


def gist(
    img: np.ndarray,
    image_size: int = 128,
    orientations: Tuple[int, ...] = (8, 8, 4),
    num_blocks: int = 4,
) -> np.ndarray:
    """Compute a 1D GIST descriptor vector for a grayscale or BGR image."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Expected a non-empty numpy.ndarray image.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size <= 0 or image_size % 2 != 0:
        raise ValueError("image_size must be a positive even integer.")

    resized = cv2.resize(img, (image_size, image_size))
    gabors = create_gabor(tuple(orientations), image_size)
    prefiltered = _prefilter(resized.astype(np.float64))
    return _extract_gabor_features(prefiltered, num_blocks, gabors)
