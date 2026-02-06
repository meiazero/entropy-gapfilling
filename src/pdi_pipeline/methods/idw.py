"""Inverse Distance Weighting (IDW) interpolation.

IDW is a deterministic spatial interpolation method that estimates values
at unmeasured locations using a weighted average of neighboring known values.
Weights are inversely proportional to the distance raised to a power parameter.
"""

from __future__ import annotations

import numpy as np

from pdi_pipeline.methods.base import BaseMethod


class IDWInterpolator(BaseMethod):
    r"""Inverse Distance Weighting (IDW) interpolation.

    IDW is a deterministic spatial interpolation method that estimates values
    at unmeasured locations using a weighted average of neighboring known values.
    Weights are inversely proportional to the distance raised to a power parameter.

    Mathematical Formulation:
        Given a set of sample points {(x_i, u_i)}, the IDW interpolation function u(x) is:

        $$u(x) = \begin{cases}
            \frac{\sum_{i=1}^N w_i(x) u_i}{\sum_{i=1}^N w_i(x)} & \text{if } d(x, x_i) \neq 0 \text{ for all } i \\
            u_i & \text{if } d(x, x_i) = 0 \text{ for some } i
        \end{cases}$$

        where the weight function is:
        $$w_i(x) = \frac{1}{d(x, x_i)^p}$$

        and $p$ is the power parameter (typically 2), $d$ is the Euclidean distance metric.

    Note:
        For satellite imagery with varying texture complexity (entropy), consider
        adjusting the power parameter: lower values (p=1) for high-entropy regions
        to incorporate more neighbors, higher values (p=3) for low-entropy regions
        where nearby pixels are more representative.

    Citation: Wikipedia contributors. "Inverse distance weighting." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Inverse_distance_weighting
    """

    name = "idw"

    def __init__(
        self, power: float = 2.0, kernel_size: int | None = None
    ) -> None:
        """Initialize IDW interpolator.

        Args:
            power: Power parameter for distance weighting (default: 2.0, Shepard's method).
                   p=1: Linear decay, more neighbors contribute.
                   p=2: Standard IDW, balanced weighting (recommended default).
                   p=3+: Stronger locality, closer neighbors dominate.
            kernel_size: Search window size. If None, uses entire image.
        """
        self.power = power
        self.kernel_size = kernel_size

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply IDW interpolation to recover missing pixels.

        Args:
            degraded: Array with missing data (e.g., NaN or masked pixels).
            mask: Binary mask where 1 indicates missing pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed array with same shape as degraded.
        """
        mask_2d = self._normalize_mask(mask)
        result = degraded.copy()
        h, w = degraded.shape[:2]
        is_multichannel = degraded.ndim == 3

        # Calculate kernel radius
        radius = max(h, w) if self.kernel_size is None else self.kernel_size
        radius = max(1, radius)

        gap_y, gap_x = np.where(mask_2d)

        for y, x in zip(gap_y, gap_x):
            # Define local search window
            y_min = max(0, y - radius)
            y_max = min(h, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius + 1)

            # Identify valid pixels in the local window
            local_mask = mask_2d[y_min:y_max, x_min:x_max]
            local_valid = ~local_mask

            local_y_indices, local_x_indices = np.where(local_valid)

            if len(local_y_indices) == 0:
                continue  # No valid neighbors, skip this pixel

            # Get absolute coordinates and values
            abs_y = local_y_indices + y_min
            abs_x = local_x_indices + x_min
            values = degraded[abs_y, abs_x]

            # Calculate Euclidean distances
            distances = np.sqrt((abs_y - y) ** 2 + (abs_x - x) ** 2)

            # Handle zero distances (should not occur if mask is correct)
            distances = np.maximum(distances, 1e-10)

            # Compute IDW weights: w_i = 1 / d_i^p
            weights = 1.0 / np.power(distances, self.power)

            # Compute weighted average
            total_weight = weights.sum()

            if is_multichannel:
                # Multi-channel: weights (N,), values (N, C)
                weighted_sum = (values * weights[:, np.newaxis]).sum(axis=0)
            else:
                # Single-channel
                weighted_sum = (values * weights).sum()

            result[y, x] = weighted_sum / total_weight

        return self._finalize(result)
