"""Thin Plate Spline (TPS) interpolation."""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import RBFInterpolator

from pdi_pipeline.methods._scattered_data import select_local_points
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_MIN_VALID_POINTS = 4
"""Minimum number of valid pixels required for thin plate spline fitting."""


class SplineInterpolator(BaseMethod):
    """Thin Plate Spline (TPS) interpolation - minimizes bending energy.

    Equivalent to RBF with phi(r) = r^2 log(r).

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
            smoothing: Smoothing factor (0 = exact interpolation, >0 = smoother fit).
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
        """Interpolate a single 2D channel using thin plate spline.

        Args:
            channel_data: 2D array of pixel values for one channel.
            local_valid_y: Row indices of selected valid pixels.
            local_valid_x: Column indices of selected valid pixels.
            gap_y: Row indices of gap pixels.
            gap_x: Column indices of gap pixels.

        Returns:
            Channel array with gap pixels filled.
        """
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
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            logger.warning(
                "Thin plate spline interpolation failed: %s; "
                "falling back to mean fill for %d gap pixels",
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
        """Apply Thin Plate Spline interpolation to recover missing pixels.

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

        if len(valid_y) < _MIN_VALID_POINTS:
            logger.warning(
                "Insufficient valid pixels (%d < %d); returning input unchanged",
                len(valid_y),
                _MIN_VALID_POINTS,
            )
            return self._finalize(degraded.copy())

        logger.debug(
            "Spline interpolation: %d gap pixels, %d valid pixels",
            len(gap_y),
            len(valid_y),
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

        def _channel_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            return self._interpolate_channel(
                ch, local_valid_y, local_valid_x, gap_y, gap_x
            )

        result = self._apply_channelwise(degraded, mask_2d, _channel_fn)
        return self._finalize(result)
