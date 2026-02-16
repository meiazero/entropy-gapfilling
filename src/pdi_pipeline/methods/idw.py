"""Inverse Distance Weighting (IDW) interpolation.

IDW is a deterministic spatial interpolation method that estimates values
at unmeasured locations using a weighted average of neighboring known values.
Weights are inversely proportional to the distance raised to a power parameter.
"""

from __future__ import annotations

import logging

import numpy as np

from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DISTANCE_EPSILON = 1e-10
"""Minimum distance to avoid division by zero in weight computation."""

_MIN_KERNEL_RADIUS = 1
"""Minimum kernel radius in pixels."""


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
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        result = degraded.copy()
        h, w = degraded.shape[:2]
        is_multichannel = degraded.ndim == 3

        # Calculate kernel radius
        radius = max(h, w) if self.kernel_size is None else self.kernel_size
        radius = max(_MIN_KERNEL_RADIUS, radius)

        gap_y, gap_x = np.where(mask_2d)
        n_gaps = len(gap_y)
        logger.debug(
            "IDW interpolation: %d gap pixels, power=%.1f, radius=%d",
            n_gaps,
            self.power,
            radius,
        )

        if n_gaps == 0:
            return self._finalize(result)

        # Global (full-image) vectorized path: when no kernel window
        # restriction is needed, we can avoid the per-pixel Python loop
        # entirely by computing a distance matrix between all gap pixels
        # and all valid pixels.
        valid_y, valid_x = np.where(~mask_2d)
        n_valid = len(valid_y)

        if n_valid == 0:
            logger.debug("No valid pixels; returning degraded image as-is")
            return self._finalize(result)

        use_global = self.kernel_size is None or radius >= max(h, w)

        if use_global and n_gaps > 0 and n_valid > 0:
            # Vectorized: distance matrix (n_gaps, n_valid)
            dy = (
                gap_y[:, np.newaxis].astype(np.float64) - valid_y[np.newaxis, :]
            )
            dx = (
                gap_x[:, np.newaxis].astype(np.float64) - valid_x[np.newaxis, :]
            )
            dist = np.sqrt(dy * dy + dx * dx)
            dist = np.maximum(dist, _DISTANCE_EPSILON)

            weights = 1.0 / np.power(dist, self.power)  # (n_gaps, n_valid)
            weight_sums = weights.sum(axis=1)  # (n_gaps,)

            if is_multichannel:
                # values shape: (n_valid, C)
                values = degraded[valid_y, valid_x]
                # weighted sum: (n_gaps, C)
                weighted = weights @ values  # (n_gaps, n_valid) @ (n_valid, C)
                result[gap_y, gap_x] = weighted / weight_sums[:, np.newaxis]
            else:
                values = degraded[valid_y, valid_x]  # (n_valid,)
                weighted = weights @ values  # (n_gaps,)
                result[gap_y, gap_x] = weighted / weight_sums
        else:
            # Windowed per-pixel fallback for local kernel sizes
            for y, x in zip(gap_y, gap_x):
                y_min = max(0, int(y) - radius)
                y_max = min(h, int(y) + radius + 1)
                x_min = max(0, int(x) - radius)
                x_max = min(w, int(x) + radius + 1)

                local_mask = mask_2d[y_min:y_max, x_min:x_max]
                local_valid = ~local_mask
                local_y_indices, local_x_indices = np.where(local_valid)

                if len(local_y_indices) == 0:
                    continue

                abs_y = local_y_indices + y_min
                abs_x = local_x_indices + x_min
                values = degraded[abs_y, abs_x]

                distances = np.sqrt((abs_y - y) ** 2 + (abs_x - x) ** 2).astype(
                    np.float64
                )
                distances = np.maximum(distances, _DISTANCE_EPSILON)

                wt = 1.0 / np.power(distances, self.power)
                total_wt = wt.sum()

                if is_multichannel:
                    weighted_sum = (values * wt[:, np.newaxis]).sum(axis=0)
                else:
                    weighted_sum = (values * wt).sum()

                result[y, x] = weighted_sum / total_wt

        return self._finalize(result)
