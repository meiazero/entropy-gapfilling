"""Bicubic interpolation for image reconstruction.

Bicubic interpolation uses a 4x4 neighborhood of pixels to fit a bicubic polynomial
surface. It provides smoother results than bilinear interpolation by considering
both function values and derivatives.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.methods.base import BaseMethod


class BicubicInterpolator(BaseMethod):
    r"""Bicubic interpolation for image reconstruction.

    Bicubic interpolation uses a 4x4 neighborhood of pixels to fit a bicubic polynomial
    surface. It provides smoother results than bilinear interpolation by considering
    both function values and derivatives.

    Mathematical Formulation:
        Bicubic interpolation uses a 4x4 pixel neighborhood (16 pixels). For a point at
        position $(x, y)$ with integer part $(x_0, y_0)$ and fractional part $(\Delta x, \Delta y)$,
        the interpolated value uses the separable Keys cubic kernel $W(t)$:

        $$f(x, y) = \sum_{i=-1}^{2} \sum_{j=-1}^{2} W(\Delta x - i) W(\Delta y - j) f(x_0+i, y_0+j)$$

        The Keys cubic kernel with parameter $\alpha$ (typically -0.5) is:

        $$W(t) = \begin{cases}
            (\alpha+2)|t|^3 - (\alpha+3)|t|^2 + 1 & \text{if } |t| \leq 1 \\
            \alpha|t|^3 - 5\alpha|t|^2 + 8\alpha|t| - 4\alpha & \text{if } 1 < |t| \leq 2 \\
            0 & \text{otherwise}
        \end{cases}$$

        The kernel has compact support with radius 2, making the effective neighborhood 4x4 pixels.

    Citation: Wikipedia contributors. "Bicubic interpolation." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    name = "bicubic"

    def __init__(self, alpha: float = -0.5) -> None:
        """Initialize bicubic interpolator.

        Args:
            alpha: Parameter for the cubic kernel. Common values:
                   -0.5 (Keys kernel, default - sharper)
                   -0.75 (softer)
                   -1.0 (even softer)
        """
        self.alpha = alpha

    @staticmethod
    def _bicubic_kernel(
        distance: np.ndarray, alpha: float = -0.5
    ) -> np.ndarray:
        """Compute bicubic kernel weights using Keys kernel.

        Args:
            distance: Normalized distance from pixel (absolute value)
            alpha: Kernel parameter (typically -0.5 for Keys kernel)

        Returns:
            Kernel weights
        """
        abs_distance = np.abs(distance)

        # Initialize weights array
        weights = np.zeros_like(abs_distance)

        # For |x| <= 1: w(x) = (alpha + 2)|x|^3 - (alpha + 3)|x|^2 + 1
        mask1 = abs_distance <= 1.0
        weights[mask1] = (
            (alpha + 2) * np.power(abs_distance[mask1], 3)
            - (alpha + 3) * np.power(abs_distance[mask1], 2)
            + 1
        )

        # For 1 < |x| <= 2: w(x) = alpha|x|^3 - 5*alpha|x|^2 + 8*alpha|x| - 4*alpha
        mask2 = (abs_distance > 1.0) & (abs_distance <= 2.0)
        weights[mask2] = (
            alpha * np.power(abs_distance[mask2], 3)
            - 5 * alpha * np.power(abs_distance[mask2], 2)
            + 8 * alpha * abs_distance[mask2]
            - 4 * alpha
        )

        # For |x| > 2: w(x) = 0 (already initialized to 0)

        return weights

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply bicubic interpolation to recover missing pixels.

        Uses canonical 4x4 pixel neighborhood (radius=2) as per Wikipedia definition.
        If valid pixels are not available in the 4x4 neighborhood, falls back to
        nearest valid pixel.

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
        result = np.nan_to_num(result, nan=0.0)
        gap_y, gap_x = np.where(mask_2d)

        # Precompute nearest valid pixel for fallback
        valid_mask = ~mask_2d
        if not np.any(valid_mask):
            return self._finalize(result)

        _, nearest_indices = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )

        for y, x in zip(gap_y, gap_x):
            # Use canonical 4x4 neighborhood (radius=2)
            y_min, y_max = max(0, y - 2), min(height, y + 3)
            x_min, x_max = max(0, x - 2), min(width, x + 3)

            local_mask = mask_2d[y_min:y_max, x_min:x_max]
            local_valid = ~local_mask
            local_y_indices, local_x_indices = np.where(local_valid)

            if len(local_y_indices) == 0:
                # Fallback to nearest valid pixel
                nearest_y = nearest_indices[0, y, x]
                nearest_x = nearest_indices[1, y, x]
                result[y, x] = degraded[nearest_y, nearest_x]
                continue

            abs_y = local_y_indices + y_min
            abs_x = local_x_indices + x_min
            values = degraded[abs_y, abs_x]

            # Calculate distances (unnormalized, kernel expects pixel units)
            dy = (abs_y - y).astype(np.float32)
            dx = (abs_x - x).astype(np.float32)

            # Compute bicubic weights using Keys kernel
            wy = self._bicubic_kernel(dy, self.alpha)
            wx = self._bicubic_kernel(dx, self.alpha)
            weights = wy * wx

            total_weight = weights.sum()
            if total_weight < 1e-10:
                # Use nearest valid pixel from the local neighborhood
                distances = np.abs(dy) + np.abs(dx)
                nearest_idx = np.argmin(distances)
                result[y, x] = values[nearest_idx]
            else:
                if values.ndim > 1:
                    weighted_sum = (values * weights[:, np.newaxis]).sum(axis=0)
                else:
                    weighted_sum = (values * weights).sum()
                result[y, x] = weighted_sum / total_weight

        return self._finalize(result)


def build() -> BicubicInterpolator:
    """Build a BicubicInterpolator instance."""
    return BicubicInterpolator()
