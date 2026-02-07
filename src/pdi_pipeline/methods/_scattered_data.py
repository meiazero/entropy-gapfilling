"""Shared point-selection utilities for scattered-data interpolators."""

from __future__ import annotations

import numpy as np


def select_local_points(
    valid_y: np.ndarray,
    valid_x: np.ndarray,
    gap_y: np.ndarray,
    gap_x: np.ndarray,
    h: int,
    w: int,
    *,
    kernel_size: int | None = None,
    max_points: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Select valid points near the gap region, with fallback and downsampling.

    Args:
        valid_y: Row indices of valid (non-gap) pixels.
        valid_x: Column indices of valid pixels.
        gap_y: Row indices of gap pixels.
        gap_x: Column indices of gap pixels.
        h: Image height.
        w: Image width.
        kernel_size: Search window size. ``None`` uses the whole image.
        max_points: Maximum number of points to return.

    Returns:
        ``(local_valid_y, local_valid_x)`` coordinate arrays.
    """
    radius = max(h, w) if kernel_size is None else max(1, kernel_size // 2)

    y_min = max(0, int(gap_y.min()) - radius)
    y_max = min(h - 1, int(gap_y.max()) + radius)
    x_min = max(0, int(gap_x.min()) - radius)
    x_max = min(w - 1, int(gap_x.max()) + radius)

    inside = (
        (valid_y >= y_min)
        & (valid_y <= y_max)
        & (valid_x >= x_min)
        & (valid_x <= x_max)
    )
    local_y = valid_y[inside]
    local_x = valid_x[inside]

    if len(local_y) < 4:
        local_y = valid_y
        local_x = valid_x

    if len(local_y) > max_points:
        indices = np.linspace(0, len(local_y) - 1, max_points, dtype=int)
        local_y = local_y[indices]
        local_x = local_x[indices]

    return local_y, local_x
