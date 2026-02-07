"""Shared griddata interpolation with nearest-neighbour fallback."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


def griddata_fill(
    degraded: np.ndarray,
    mask_2d: np.ndarray,
    method: str,
    apply_channelwise: Callable[
        [
            np.ndarray,
            np.ndarray,
            Callable[[np.ndarray, np.ndarray], np.ndarray],
        ],
        np.ndarray,
    ],
) -> np.ndarray | None:
    """Interpolate gaps via scipy griddata with nearest-neighbour fallback.

    Args:
        degraded: Input image array (H, W) or (H, W, C).
        mask_2d: 2D boolean mask (True = gap pixel).
        method: Interpolation method for scipy.interpolate.griddata
                (e.g. "linear", "cubic").
        apply_channelwise: The BaseMethod._apply_channelwise static method.

    Returns:
        Filled array, or None if there are no gaps or no valid pixels.
    """
    gap_y, gap_x = np.where(mask_2d)
    if len(gap_y) == 0:
        return None

    valid_mask = ~mask_2d
    valid_y, valid_x = np.where(valid_mask)
    if len(valid_y) == 0:
        return None

    valid_coords = np.column_stack([valid_y, valid_x]).astype(np.float64)
    gap_coords = np.column_stack([gap_y, gap_x]).astype(np.float64)

    _, nn_indices = distance_transform_edt(
        ~valid_mask,
        return_distances=True,
        return_indices=True,
    )

    def _fill_channel(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        values = ch[valid_y, valid_x].astype(np.float64)
        filled = griddata(valid_coords, values, gap_coords, method=method)

        nan_mask = np.isnan(filled)
        if np.any(nan_mask):
            fallback_y = nn_indices[0, gap_y[nan_mask], gap_x[nan_mask]]
            fallback_x = nn_indices[1, gap_y[nan_mask], gap_x[nan_mask]]
            filled[nan_mask] = ch[fallback_y, fallback_x]

        result = ch.copy()
        result[gap_y, gap_x] = filled
        return result

    return apply_channelwise(degraded, mask_2d, _fill_channel)
