"""Nearest Neighbor interpolation for image reconstruction.

Nearest-neighbor interpolation selects the value of the nearest point and does
not consider the values of neighboring points at all, yielding a piecewise-constant
interpolant. This method is simple but can produce blocky artifacts.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


class NearestInterpolator(BaseMethod):
    r"""Nearest Neighbor interpolation for image reconstruction.

    Nearest-neighbor interpolation selects the value of the nearest point and does
    not consider the values of neighboring points at all, yielding a piecewise-constant
    interpolant. This method is simple but can produce blocky artifacts.

    Mathematical Formulation:
        For a point (x, y), find the nearest known point (x_i, y_i):

        $$f(x, y) = f(x_i, y_i)$$

        where (x_i, y_i) is the closest known point to (x, y) in Euclidean distance.

    Citation: Wikipedia contributors. "Nearest-neighbor interpolation." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
    """

    name = "nearest"

    def __init__(self, kernel_size: int | None = None) -> None:
        """Initialize the nearest neighbor interpolator.

        Args:
            kernel_size: Search window size (max distance in pixels). If None,
                uses the full image extent.
        """
        self.kernel_size = kernel_size

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply nearest neighbor interpolation to recover missing pixels.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
                Shape ``(H, W)`` or broadcastable ``(H, W, C)``.
            meta: Optional metadata (CRS, transform, band names, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        # Create a copy to avoid modifying the input
        result = degraded.copy()

        # Replace NaN with 0 in input
        result = np.nan_to_num(result, nan=0.0)

        # Invert mask: True for valid pixels, False for gaps
        valid_mask = ~mask_2d

        # Handle multi-channel images
        is_multichannel = degraded.ndim == 3

        distances, indices = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )

        height, width = degraded.shape[:2]

        radius = (
            self.kernel_size
            if self.kernel_size is not None
            else max(height, width)
        )
        within_radius = distances <= radius
        gap_pixels_y, gap_pixels_x = np.where(mask_2d & within_radius)

        if len(gap_pixels_y) == 0:
            logger.debug(
                "No gap pixels within radius %d; returning as-is.", radius
            )
            return self._finalize(result)

        nearest_y = indices[0, gap_pixels_y, gap_pixels_x]
        nearest_x = indices[1, gap_pixels_y, gap_pixels_x]

        if is_multichannel:
            channels = degraded.shape[2]
            logger.debug(
                "Filling %d gap pixels across %d channels.",
                len(gap_pixels_y),
                channels,
            )
            for channel_idx in range(channels):
                channel_data = degraded[:, :, channel_idx]
                result[gap_pixels_y, gap_pixels_x, channel_idx] = channel_data[
                    nearest_y, nearest_x
                ]
        else:
            logger.debug(
                "Filling %d gap pixels (single channel).", len(gap_pixels_y)
            )
            result[gap_pixels_y, gap_pixels_x] = degraded[nearest_y, nearest_x]

        return self._finalize(result)
