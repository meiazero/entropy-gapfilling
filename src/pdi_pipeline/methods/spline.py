r"""Thin Plate Spline interpolation for image reconstruction.

Thin Plate Splines (TPS) are a popular choice for scattered data interpolation
because they minimize bending energy, producing smooth surfaces. TPS is equivalent
to RBF with a specific kernel: \phi(r) = r^2 \log(r).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from pdi_pipeline.methods.base import BaseMethod


class SplineInterpolator(BaseMethod):
    r"""Thin Plate Spline interpolation for image reconstruction.

    Thin Plate Splines (TPS) are a popular choice for scattered data interpolation
    because they minimize bending energy, producing smooth surfaces. TPS is equivalent
    to RBF with a specific kernel: \phi(r) = r^2 \log(r).

    Mathematical Formulation:
        The Thin Plate Spline interpolant f(x,y) minimizes the bending energy:
        $$E[f] = \iint_{\mathbb{R}^2} \left( \left(\frac{\partial^2 f}{\partial x^2}\right)^2 + 2\left(\frac{\partial^2 f}{\partial x \partial y}\right)^2 + \left(\frac{\partial^2 f}{\partial y^2}\right)^2 \right) dx dy$$

        The solution has the form:
        $$f(x,y) = a_0 + a_1 x + a_2 y + \sum_{i=1}^N w_i \phi(\| (x,y) - (x_i,y_i) \|)$$

        where $\phi(r) = r^2 \log(r)$ is the radial basis function.

    Citation: Wikipedia contributors. "Thin plate spline." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Thin_plate_spline
    """

    name = "spline"

    def __init__(
        self,
        smoothing: float = 0.0,
        max_training_points: int = 5000,
        kernel_size: int | None = None,
    ) -> None:
        """Initialize Spline interpolator.

        Args:
            smoothing: Smoothing factor (0 = exact interpolation, >0 = smoother fit)
            max_training_points: Maximum number of training points to use.
            kernel_size: Search window size. If None, uses entire image.
        """
        self.smoothing = smoothing
        self.max_training_points = max_training_points
        self.kernel_size = kernel_size

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Thin Plate Spline interpolation to recover missing pixels.

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

        gap_y, gap_x = np.where(mask_2d)
        if len(gap_y) == 0:
            return self._finalize(result)

        # Get coordinates of valid pixels
        valid_y, valid_x = np.where(~mask_2d)

        if len(valid_y) < 4:
            # Need at least 4 points for spline fitting
            return self._finalize(result)

        # Restrict training points to a neighborhood around the missing region
        radius = self._get_radius(h, w)
        y_min = max(0, int(gap_y.min()) - radius)
        y_max = min(h - 1, int(gap_y.max()) + radius)
        x_min = max(0, int(gap_x.min()) - radius)
        x_max = min(w - 1, int(gap_x.max()) + radius)

        local_selector = (
            (valid_y >= y_min)
            & (valid_y <= y_max)
            & (valid_x >= x_min)
            & (valid_x <= x_max)
        )
        local_valid_y = valid_y[local_selector]
        local_valid_x = valid_x[local_selector]

        if len(local_valid_y) < 4:
            local_valid_y = valid_y
            local_valid_x = valid_x

        # Downsample if too many points
        if len(local_valid_y) > self.max_training_points:
            indices = np.linspace(
                0,
                len(local_valid_y) - 1,
                self.max_training_points,
                dtype=int,
            )
            local_valid_y = local_valid_y[indices]
            local_valid_x = local_valid_x[indices]

        # Process each channel separately
        if is_multichannel:
            channels = degraded.shape[2]
            for channel_idx in range(channels):
                valid_values = degraded[
                    local_valid_y, local_valid_x, channel_idx
                ]

                try:
                    # Create bivariate spline using RBF with thin_plate_spline kernel
                    spline = RBFInterpolator(
                        np.column_stack([local_valid_y, local_valid_x]),
                        valid_values,
                        kernel="thin_plate_spline",
                        smoothing=self.smoothing,
                    )

                    # Interpolate gap pixels
                    interpolated_values = spline(
                        np.column_stack([gap_y, gap_x])
                    )
                    result[gap_y, gap_x, channel_idx] = interpolated_values
                except Exception:
                    # Fallback to mean if spline fitting fails
                    if len(gap_y) > 0:
                        result[gap_y, gap_x, channel_idx] = float(
                            np.mean(valid_values)
                        )
        else:
            valid_values = degraded[local_valid_y, local_valid_x]

            try:
                # Create thin plate spline interpolator
                spline = RBFInterpolator(
                    np.column_stack([local_valid_y, local_valid_x]),
                    valid_values,
                    kernel="thin_plate_spline",
                    smoothing=self.smoothing,
                )

                # Interpolate gap pixels
                interpolated_values = spline(np.column_stack([gap_y, gap_x]))
                result[gap_y, gap_x] = interpolated_values
            except Exception:
                # Fallback to mean if spline fitting fails
                if len(gap_y) > 0:
                    result[gap_y, gap_x] = float(np.mean(valid_values))

        return self._finalize(result)

    def _get_radius(self, h: int, w: int) -> int:
        """Calculate search radius from kernel_size."""
        if self.kernel_size is None:
            return max(h, w)
        return max(1, self.kernel_size // 2)
