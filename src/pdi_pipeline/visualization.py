"""Shared visualization utilities for array-to-image conversion.

Provides helpers used by both the experiment runner (PNG export of
reconstructions) and the figure generation scripts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def to_display_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a multi-band array to a displayable RGB image in [0, 1].

    Args:
        arr: 2D (H, W) or 3D (H, W, C) array. If C >= 3 the first
            three bands are taken as R, G, B; extra bands (e.g. NIR)
            are dropped.

    Returns:
        Float array in [0, 1] with shape (H, W, 3) or (H, W) for
        single-band inputs. Constant arrays return zeros.
    """
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.float64)

    result = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.nan_to_num(result, nan=0.0)


def save_array_as_png(
    arr: np.ndarray,
    path: str | Path,
    dpi: int = 150,
) -> None:
    """Save a numpy array as a borderless PNG image.

    Handles both 2D grayscale and 3D multi-channel arrays by
    delegating to :func:`to_display_rgb` for normalization and
    band selection.

    Args:
        arr: 2D or 3D numpy array.
        path: Output file path (parent directories created
            automatically).
        dpi: Resolution for the saved image.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = to_display_rgb(arr)

    if rgb.ndim == 2:
        plt.imsave(str(path), rgb, cmap="gray", dpi=dpi)
    else:
        plt.imsave(str(path), rgb, dpi=dpi)
