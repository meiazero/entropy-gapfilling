"""Radial Basis Function (RBF) interpolation."""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import numpy as np
from scipy.interpolate import RBFInterpolator as SciPyRBFInterpolator

from pdi_pipeline.methods._scattered_data import select_local_points
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


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
    """Radial Basis Function interpolation via ``scipy.interpolate.RBFInterpolator``.

    Solves s(x) = sum_k w_k phi(||x - x_k||) for gap pixels.

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
            kernel: RBF kernel ('thin_plate_spline', 'gaussian', etc.).
            epsilon: Shape parameter (kernel-dependent).
            smoothing: Smoothing factor (0 = exact interpolation).
            max_training_points: Max known pixels to use.
            kernel_size: Search window size. None = full image.
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
        """Interpolate a single 2D channel using RBF.

        Args:
            channel_data: 2D array of pixel values for one channel.
            local_valid_y: Row indices of selected valid pixels.
            local_valid_x: Column indices of selected valid pixels.
            valid_coords: Stacked (y, x) coordinates of valid pixels.
            gap_coords: Stacked (y, x) coordinates of gap pixels.
            gap_y: Row indices of gap pixels.
            gap_x: Column indices of gap pixels.

        Returns:
            Channel array with gap pixels filled.
        """
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
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            logger.warning(
                "RBF interpolation failed (kernel=%s): %s; "
                "falling back to mean fill for %d gap pixels",
                self.kernel,
                exc,
                len(gap_y),
            )
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

        h, w = degraded.shape[:2]

        gap_y, gap_x = np.where(mask_2d)
        valid_y, valid_x = np.where(~mask_2d)

        if len(valid_y) == 0:
            logger.warning(
                "No valid pixels available; returning input unchanged"
            )
            return self._finalize(degraded.copy())

        logger.debug(
            "RBF interpolation: %d gap pixels, %d valid pixels, kernel=%s",
            len(gap_y),
            len(valid_y),
            self.kernel,
        )

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
