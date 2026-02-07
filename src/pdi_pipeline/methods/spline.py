r"""Thin Plate Spline interpolation for image reconstruction.

Thin Plate Splines (TPS) are a popular choice for scattered data interpolation
because they minimize bending energy, producing smooth surfaces. TPS is equivalent
to RBF with a specific kernel: \phi(r) = r^2 \log(r).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator

from pdi_pipeline.methods._scattered_data import select_local_points
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

    def _interpolate_channel(
        self,
        channel_data: np.ndarray,
        local_valid_y: np.ndarray,
        local_valid_x: np.ndarray,
        gap_y: np.ndarray,
        gap_x: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a single 2D channel using thin plate spline."""
        valid_values = channel_data[local_valid_y, local_valid_x]
        result = channel_data.copy()

        try:
            spline = RBFInterpolator(
                np.column_stack([local_valid_y, local_valid_x]),
                valid_values,
                kernel="thin_plate_spline",
                smoothing=self.smoothing,
            )
            result[gap_y, gap_x] = spline(np.column_stack([gap_y, gap_x]))
        except Exception:
            if len(gap_y) > 0:
                result[gap_y, gap_x] = float(np.mean(valid_values))

        return result

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

        gap_y, gap_x = np.where(mask_2d)
        if len(gap_y) == 0:
            return self._finalize(result)

        valid_y, valid_x = np.where(~mask_2d)
        if len(valid_y) < 4:
            return self._finalize(result)

        local_valid_y, local_valid_x = select_local_points(
            valid_y,
            valid_x,
            gap_y,
            gap_x,
            h,
            w,
            kernel_size=self.kernel_size,
            max_points=self.max_training_points,
        )

        def _channel_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            return self._interpolate_channel(
                ch, local_valid_y, local_valid_x, gap_y, gap_x
            )

        result = self._apply_channelwise(degraded, mask_2d, _channel_fn)
        return self._finalize(result)
