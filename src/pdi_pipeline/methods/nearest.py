"""Nearest Neighbor interpolation for image reconstruction.

Nearest-neighbor interpolation selects the value of the nearest point and does
not consider the values of neighboring points at all, yielding a piecewise-constant
interpolant. This method is simple but can produce blocky artifacts.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.methods.base import BaseMethod


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
        """Initialize the interpolator.

        Args:
            kernel_size: Search window size. If None, uses entire image.
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
            degraded: Array with missing data (e.g., NaN or masked pixels).
            mask: Binary mask where 1 indicates missing pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed array with same shape as degraded.
        """
        mask_2d = self._normalize_mask(mask)

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
            return self._finalize(result)

        nearest_y = indices[0, gap_pixels_y, gap_pixels_x]
        nearest_x = indices[1, gap_pixels_y, gap_pixels_x]

        if is_multichannel:
            channels = degraded.shape[2]
            for channel_idx in range(channels):
                channel_data = degraded[:, :, channel_idx]
                result[gap_pixels_y, gap_pixels_x, channel_idx] = channel_data[
                    nearest_y, nearest_x
                ]
        else:
            result[gap_pixels_y, gap_pixels_x] = degraded[nearest_y, nearest_x]

        return self._finalize(result)


def build() -> NearestInterpolator:
    """Build a NearestInterpolator instance."""
    return NearestInterpolator()
