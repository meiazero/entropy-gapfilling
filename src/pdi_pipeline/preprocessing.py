"""Shared preprocessing utilities for datasets.

Keeps normalization, NaN handling, and mask shaping consistent across
classic and DL pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_SPAN = 1e-8


@dataclass(frozen=True)
class NormalizeRange:
    """Range used for per-sample normalization."""

    vmin: float
    vmax: float


def compute_normalize_range(clean: np.ndarray) -> NormalizeRange:
    """Compute a stable [vmin, vmax] range from the clean image."""
    vmin = float(clean.min())
    vmax = float(clean.max())
    if vmax - vmin < _MIN_SPAN:
        vmax = vmin + 1.0
    return NormalizeRange(vmin=vmin, vmax=vmax)


def normalize_image(
    arr: np.ndarray,
    norm_range: NormalizeRange,
) -> np.ndarray:
    """Normalize an array to [0, 1] using a fixed range."""
    span = max(norm_range.vmax - norm_range.vmin, _MIN_SPAN)
    out = (arr - norm_range.vmin) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def replace_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with 0.0, preserving dtype when possible."""
    return np.nan_to_num(arr, nan=0.0, copy=False)


def ensure_mask_2d(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is 2D and float32 without thresholding."""
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    return mask_arr


def threshold_mask(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert a mask to 2D float32 with values in {0, 1}."""
    mask_arr = ensure_mask_2d(mask)
    return (mask_arr > threshold).astype(np.float32)
