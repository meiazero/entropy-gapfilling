"""Radial Basis Function (RBF) interpolation.

RBF interpolation constructs a smooth surface through scattered data points
using radially symmetric basis functions. The method solves a linear system
to find coefficients that satisfy the interpolation conditions.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from scipy.interpolate import RBFInterpolator as SciPyRBFInterpolator

from pdi_pipeline.methods._scattered_data import select_local_points
from pdi_pipeline.methods.base import BaseMethod


def _get_kernel_literal(kernel: str) -> Any:
    """Helper to cast kernel string to SciPy literal type."""
    return cast(
        Literal[
            "thin_plate_spline",
            "linear",
            "cubic",
            "quintic",
            "multiquadric",
            "inverse_multiquadric",
            "inverse_quadratic",
            "gaussian",
        ],
        kernel,
    )


class RBFInterpolator(BaseMethod):
    r"""Radial Basis Function (RBF) interpolation.

    RBF interpolation constructs a smooth surface through scattered data points
    using radially symmetric basis functions. The method solves a linear system
    to find coefficients that satisfy the interpolation conditions.

    Mathematical Formulation:
        Given N sample points {x_k}, the RBF interpolant is:

        $$s(x) = \sum_{k=1}^N w_k \phi(\|x - x_k\|)$$

        where \phi is a radial basis function (e.g., Gaussian, multiquadric, thin-plate spline),
        and the weights w_k are found by solving the linear system:

        $$\begin{bmatrix}
            \phi(\|x_1 - x_1\|) & \phi(\|x_2 - x_1\|) & \cdots & \phi(\|x_N - x_1\|) \\
            \phi(\|x_1 - x_2\|) & \phi(\|x_2 - x_2\|) & \cdots & \phi(\|x_N - x_2\|) \\
            \vdots & \vdots & \ddots & \vdots \\
            \phi(\|x_1 - x_N\|) & \phi(\|x_2 - x_N\|) & \cdots & \phi(\|x_N - x_N\|)
        \end{bmatrix}
        \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_N \end{bmatrix} =
        \begin{bmatrix} f(x_1) \\ f(x_2) \\ \vdots \\ f(x_N) \end{bmatrix}$$

    Citation: Wikipedia contributors. "Radial basis function interpolation." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Radial_basis_function_interpolation
    """

    name = "rbf"

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        epsilon: float = 1.0,
        smoothing: float = 0.0,
        max_training_points: int = 5000,
        kernel_size: int | None = None,
    ) -> None:
        """Initialize RBF interpolator.

        Args:
            kernel: RBF kernel type. Options: 'thin_plate_spline', 'gaussian',
                    'multiquadric', 'inverse_multiquadric', 'linear', 'cubic'.
                    'thin_plate_spline' is recommended for satellite imagery
                    as it provides smooth interpolation without scale parameter.
            epsilon: Shape parameter for RBF (affects smoothness). Only used
                     by 'gaussian', 'multiquadric', and 'inverse_multiquadric'.
            smoothing: Smoothing parameter (0 = exact interpolation). Small
                       positive values (0.01-0.1) can improve numerical stability.
            max_training_points: Maximum number of training points to use.
            kernel_size: Search window size. If None, uses entire image.
        """
        self.kernel = kernel
        self.epsilon = epsilon
        self.smoothing = smoothing
        self.max_training_points = max_training_points
        self.kernel_size = kernel_size

    def _interpolate_channel(
        self,
        channel_data: np.ndarray,
        local_valid_y: np.ndarray,
        local_valid_x: np.ndarray,
        valid_coords: np.ndarray,
        gap_coords: np.ndarray,
        gap_y: np.ndarray,
        gap_x: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a single 2D channel using RBF."""
        valid_values = channel_data[local_valid_y, local_valid_x].astype(
            np.float64
        )
        result = channel_data.copy()

        try:
            kernel_lit = _get_kernel_literal(self.kernel)
            rbf = SciPyRBFInterpolator(
                valid_coords,
                valid_values,
                kernel=kernel_lit,
                epsilon=self.epsilon,
                smoothing=self.smoothing,
            )
            result[gap_y, gap_x] = rbf(gap_coords)
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
        """Apply RBF interpolation to recover missing pixels.

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
        if len(valid_y) == 0:
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

        valid_coords = np.column_stack([local_valid_y, local_valid_x]).astype(
            np.float64
        )
        gap_coords = np.column_stack([gap_y, gap_x]).astype(np.float64)

        def _channel_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            return self._interpolate_channel(
                ch,
                local_valid_y,
                local_valid_x,
                valid_coords,
                gap_coords,
                gap_y,
                gap_x,
            )

        result = self._apply_channelwise(degraded, mask_2d, _channel_fn)
        return self._finalize(result)
