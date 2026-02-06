"""Radial Basis Function (RBF) interpolation.

RBF interpolation constructs a smooth surface through scattered data points
using radially symmetric basis functions. The method solves a linear system
to find coefficients that satisfy the interpolation conditions.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
from scipy.interpolate import RBFInterpolator as SciPyRBFInterpolator

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

        # Fallback to global set if local neighborhood is too sparse
        if len(local_valid_y) < 4:
            local_valid_y = valid_y
            local_valid_x = valid_x

        # Deterministic downsampling to keep RBF tractable
        if len(local_valid_y) > self.max_training_points:
            indices = np.linspace(
                0,
                len(local_valid_y) - 1,
                self.max_training_points,
                dtype=int,
            )
            local_valid_y = local_valid_y[indices]
            local_valid_x = local_valid_x[indices]

        # Prepare coordinates for RBF
        valid_coords = np.column_stack([local_valid_y, local_valid_x]).astype(
            np.float64
        )
        gap_coords = np.column_stack([gap_y, gap_x]).astype(np.float64)

        # Process each channel separately
        kernel_lit = _get_kernel_literal(self.kernel)
        if degraded.ndim == 3:
            channels = degraded.shape[2]
            for channel_idx in range(channels):
                valid_values = degraded[
                    local_valid_y, local_valid_x, channel_idx
                ].astype(np.float64)

                # Create RBF interpolator for this channel
                rbf = SciPyRBFInterpolator(
                    valid_coords,
                    valid_values,
                    kernel=kernel_lit,
                    epsilon=self.epsilon,
                    smoothing=self.smoothing,
                )

                # Interpolate gap pixels
                interpolated_values = rbf(gap_coords)
                result[gap_y, gap_x, channel_idx] = interpolated_values
        else:
            valid_values = degraded[local_valid_y, local_valid_x].astype(
                np.float64
            )

            # Create RBF interpolator
            rbf = SciPyRBFInterpolator(
                valid_coords,
                valid_values,
                kernel=kernel_lit,
                epsilon=self.epsilon,
                smoothing=self.smoothing,
            )

            # Interpolate gap pixels
            interpolated_values = rbf(gap_coords)
            result[gap_y, gap_x] = interpolated_values

        return self._finalize(result)

    def _get_radius(self, h: int, w: int) -> int:
        """Calculate search radius from kernel_size."""
        if self.kernel_size is None:
            return max(h, w)
        return max(1, self.kernel_size // 2)
