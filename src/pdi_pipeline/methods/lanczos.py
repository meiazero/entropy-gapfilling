"""Lanczos interpolation for high-quality image reconstruction.

Implements Lanczos resampling using a windowed sinc function. This method
provides superior quality compared to bilinear and bicubic interpolation,
particularly for images with fine detail.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.methods.base import BaseMethod


class LanczosInterpolator(BaseMethod):
    r"""Lanczos interpolation for high-quality image reconstruction.

    Implements Lanczos resampling using a windowed sinc function. This method
    provides superior quality compared to bilinear and bicubic interpolation,
    particularly for images with fine detail.

    Mathematical Formulation:
        Lanczos interpolation uses a windowed sinc function with window size parameter $a$.
        For a point at position $(x, y)$ with integer part $(x_0, y_0)$ and fractional part
        $(\Delta x, \Delta y)$, the interpolated value is:

        $$f(x, y) = \sum_{i=\lfloor x \rfloor - a + 1}^{\lfloor x \rfloor + a} \sum_{j=\lfloor y \rfloor - a + 1}^{\lfloor y \rfloor + a} f(i, j) \cdot L(x - i) \cdot L(y - j)$$

        The Lanczos kernel with parameter $a$ is:
        $$L(t) = \begin{cases}
            \text{sinc}(t) \cdot \text{sinc}(t/a) & \text{if } |t| < a \\
            0 & \text{otherwise}
        \end{cases}$$

        where $\text{sinc}(t) = \frac{\sin(\pi t)}{\pi t}$ (with $\text{sinc}(0) = 1$).

        The kernel is separable and has compact support with radius $a$. For $a=3$ (default),
        this gives a 6x6 pixel neighborhood.

    Citation: Wikipedia contributors. "Lanczos resampling." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Lanczos_resampling
    """

    name = "lanczos"

    def __init__(self, a: int = 3) -> None:
        """Initialize Lanczos interpolator.

        Args:
            a: Lanczos kernel parameter (window size). Common values are 2 or 3.
               Larger values provide better quality but slower computation.
        """
        if a < 1:
            raise ValueError("Lanczos parameter 'a' must be >= 1")
        self.a = a

    @staticmethod
    def _sinc(x: np.ndarray) -> np.ndarray:
        """Compute normalized sinc function: sinc(x) = sin(πx)/(πx).

        Args:
            x: Input values

        Returns:
            Sinc function values
        """
        # Avoid division by zero
        result = np.ones_like(x)
        nonzero = np.abs(x) > 1e-10
        result[nonzero] = np.sin(np.pi * x[nonzero]) / (np.pi * x[nonzero])
        return result

    def _lanczos_kernel(self, distance: np.ndarray) -> np.ndarray:
        """Compute Lanczos kernel weights.

        Args:
            distance: Normalized distance values

        Returns:
            Kernel weights
        """
        abs_distance = np.abs(distance)
        weights = np.zeros_like(distance)

        # L(x) = sinc(x) * sinc(x/a) for |x| < a
        mask = abs_distance < self.a
        dist_masked = np.asarray(distance[mask], dtype=np.float32)
        weights[mask] = self._sinc(dist_masked) * self._sinc(
            dist_masked / np.float32(self.a)
        )

        return weights

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Lanczos interpolation to recover missing pixels.

        Uses canonical 2ax2a pixel neighborhood (e.g., 6x6 for a=3) as per Wikipedia definition.
        If valid pixels are not available in the neighborhood, falls back to nearest valid pixel.

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

        # Precompute nearest valid pixel for fallback
        valid_mask = ~mask_2d
        if not np.any(valid_mask):
            return self._finalize(result)

        _, nearest_indices = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )

        # Use canonical Lanczos radius: a
        radius = self.a

        for y, x in zip(gap_y, gap_x):
            # Define canonical Lanczos neighborhood [-a+1, a]
            y_min = max(0, y - radius + 1)
            y_max = min(height, y + radius + 1)
            x_min = max(0, x - radius + 1)
            x_max = min(width, x + radius + 1)

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

            # Calculate distances in pixel units (kernel expects this)
            delta_y = (abs_y - y).astype(np.float32)
            delta_x = (abs_x - x).astype(np.float32)

            # Compute Lanczos kernel weights
            weight_y = self._lanczos_kernel(delta_y)
            weight_x = self._lanczos_kernel(delta_x)
            weights = weight_y * weight_x

            total_weight = weights.sum()
            if total_weight < 1e-10:
                # Use nearest valid pixel from the local neighborhood
                distances = np.abs(delta_y) + np.abs(delta_x)
                nearest_idx = np.argmin(distances)
                result[y, x] = values[nearest_idx]
            else:
                if values.ndim > 1:
                    weighted_sum = (values * weights[:, np.newaxis]).sum(axis=0)
                else:
                    weighted_sum = (values * weights).sum()
                result[y, x] = weighted_sum / total_weight

        return self._finalize(result)


def build() -> LanczosInterpolator:
    """Build a LanczosInterpolator instance."""
    return LanczosInterpolator()
