"""Bilinear interpolation for image reconstruction.

Bilinear interpolation is a method for interpolating functions of two variables
using repeated linear interpolation. It is performed using linear interpolation
first in one direction, and then again in another direction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.methods.base import BaseMethod


class BilinearInterpolator(BaseMethod):
    r"""Bilinear interpolation for image reconstruction.

    Bilinear interpolation is a method for interpolating functions of two variables
    using repeated linear interpolation. It is performed using linear interpolation
    first in one direction, and then again in another direction.

    Mathematical Formulation:
        Bilinear interpolation uses a 2x2 pixel neighborhood. For a point at position $(x, y)$,
        where $x = x_0 + \Delta x$ and $y = y_0 + \Delta y$ (with $0 \leq \Delta x, \Delta y < 1$),
        the interpolated value is:

        $$f(x, y) = (1-\Delta x)(1-\Delta y)f(x_0, y_0) + \Delta x(1-\Delta y)f(x_0+1, y_0) +$$
        $$(1-\Delta x)\Delta y f(x_0, y_0+1) + \Delta x \Delta y f(x_0+1, y_0+1)$$

        This is equivalent to linear interpolation in the x-direction followed by linear
        interpolation in the y-direction. The kernel has compact support with radius 1.

    Note:
        This interpolator assumes input data is normalized to [0, 1] range.
        For satellite imagery (Sentinel-2, Landsat, MODIS), ensure reflectance
        values are scaled appropriately before interpolation. Multi-band images
        (e.g., 4 bands: B2, B3, B4, B8) are processed independently per channel.
        Output is clipped to [0, 1] range.

    Citation: Wikipedia contributors. "Bilinear interpolation." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Bilinear_interpolation
    """

    name = "bilinear"

    @staticmethod
    def _tent_kernel(distance: np.ndarray) -> np.ndarray:
        r"""Tent (triangle) kernel function for bilinear interpolation.

        Mathematical formula:
        $$\omega(t) = \max(0, 1 - |t|)$$

        Args:
            distance: Normalized distance values

        Returns:
            Kernel weights
        """
        return np.maximum(0.0, 1.0 - np.abs(distance))

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply bilinear interpolation to recover missing pixels.

        Uses canonical 2x2 pixel neighborhood as per Wikipedia definition.
        Implements exactly: f(x,y) = (1-dx)(1-dy)f00 + dx(1-dy)f10 + (1-dx)dy f01 + dx*dy*f11
        If valid pixels are not available in the 2x2 neighborhood, falls back to nearest valid pixel.

        Args:
            degraded: Array with missing data (e.g., NaN or masked pixels).
            mask: Binary mask where 1 indicates missing pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed array with same shape as degraded.
        """
        mask_2d = self._normalize_mask(mask)
        height, width = degraded.shape[:2]
        result = degraded.copy()
        gap_y, gap_x = np.where(mask_2d)

        # Precompute nearest valid pixel for fallback using distance transform
        valid_mask = ~mask_2d
        if not np.any(valid_mask):
            return self._finalize(result)

        _, nearest_indices = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )

        for y, x in zip(gap_y, gap_x):
            # Use canonical 2x2 neighborhood (radius=1)
            # Need pixels at: (y-1,x-1), (y-1,x), (y,x-1), (y,x) which form a unit square
            # But missing pixel is at (y,x), so we look at neighboring positions
            y0 = max(0, min(height - 2, y))  # Ensure we have room for y0+1
            x0 = max(0, min(width - 2, x))  # Ensure we have room for x0+1

            # Calculate fractional position within the [y0, y0+1] x [x0, x0+1] cell
            dy = float(y - y0)  # 0 <= dy <= 1
            dx = float(x - x0)  # 0 <= dx <= 1

            # Get the 4 corner values: f00, f10, f01, f11
            # f00 = image[y0, x0], f10 = image[y0, x0+1]
            # f01 = image[y0+1, x0], f11 = image[y0+1, x0+1]
            try:
                corners_mask = [
                    mask_2d[y0, x0],  # f00
                    mask_2d[y0, x0 + 1],  # f10
                    mask_2d[y0 + 1, x0],  # f01
                    mask_2d[y0 + 1, x0 + 1],  # f11
                ]
            except IndexError:
                # Edge case: fallback to nearest
                nearest_y = nearest_indices[0, y, x]
                nearest_x = nearest_indices[1, y, x]
                result[y, x] = degraded[nearest_y, nearest_x]
                continue

            # If any of the 4 corners is missing, fallback to weighted average of available pixels
            if any(corners_mask):
                # Fallback: use available pixels in 2x2 neighborhood with tent kernel weights
                y_min, y_max = max(0, y - 1), min(height, y + 2)
                x_min, x_max = max(0, x - 1), min(width, x + 2)

                local_mask = mask_2d[y_min:y_max, x_min:x_max]
                local_valid = ~local_mask
                local_y_indices, local_x_indices = np.where(local_valid)

                if len(local_y_indices) == 0:
                    nearest_y = nearest_indices[0, y, x]
                    nearest_x = nearest_indices[1, y, x]
                    result[y, x] = degraded[nearest_y, nearest_x]
                    continue

                abs_y = local_y_indices + y_min
                abs_x = local_x_indices + x_min
                values = degraded[abs_y, abs_x]

                delta_y = np.abs(abs_y - y).astype(np.float32)
                delta_x = np.abs(abs_x - x).astype(np.float32)
                weight_y = self._tent_kernel(delta_y)
                weight_x = self._tent_kernel(delta_x)
                weights = weight_y * weight_x

                total_weight = weights.sum()
                if total_weight < 1e-10:
                    distances = delta_y + delta_x
                    nearest_idx = np.argmin(distances)
                    result[y, x] = values[nearest_idx]
                else:
                    if values.ndim > 1:
                        weighted_sum = (values * weights[:, np.newaxis]).sum(
                            axis=0
                        )
                    else:
                        weighted_sum = (values * weights).sum()
                    result[y, x] = weighted_sum / total_weight
            else:
                # All 4 corners are valid: use exact bilinear formula
                # f(x,y) = (1-dx)(1-dy)f00 + dx(1-dy)f10 + (1-dx)dy f01 + dx*dy f11
                f00 = degraded[y0, x0]
                f10 = degraded[y0, x0 + 1]
                f01 = degraded[y0 + 1, x0]
                f11 = degraded[y0 + 1, x0 + 1]

                result[y, x] = (
                    (1 - dx) * (1 - dy) * f00
                    + dx * (1 - dy) * f10
                    + (1 - dx) * dy * f01
                    + dx * dy * f11
                )

        return self._finalize(result)


def build() -> BilinearInterpolator:
    """Build a BilinearInterpolator instance."""
    return BilinearInterpolator()
