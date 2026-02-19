"""Lanczos spectral gap-filling (Papoulis-Gerchberg iteration)."""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.exceptions import (
    ValidationError,
)
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


def _invalid_lanczos_parameter_error() -> ValidationError:
    return ValidationError("Lanczos parameter 'a' must be >= 1")


class LanczosInterpolator(BaseMethod):
    """Lanczos spectral gap-filling via Papoulis-Gerchberg iteration.

    Iteratively applies a Lanczos-windowed low-pass in the FFT domain,
    restoring known pixels after each step until convergence.
    See: Papoulis (1975), IEEE Trans. CAS; Gerchberg (1974), Optica Acta.
    """

    name = "lanczos"

    def __init__(
        self,
        a: int = 3,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
    ) -> None:
        """Initialize the Lanczos spectral interpolator.

        Args:
            a: Lanczos window parameter (controls passband width).
                ``a=2`` gives a narrower passband (smoother result).
                ``a=3`` (default) balances detail preservation and smoothness.
            max_iterations: Maximum Papoulis-Gerchberg iterations.
            tolerance: RMS convergence threshold on gap pixels.

        Raises:
            ValidationError: If ``a`` is less than 1.
        """
        if a < 1:
            raise _invalid_lanczos_parameter_error()
        self.a = a
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def _build_frequency_response(self, height: int, width: int) -> np.ndarray:
        """Build separable 2-D Lanczos frequency response.

        Args:
            height: Spatial height of the image.
            width: Spatial width of the image.

        Returns:
            2-D array of shape ``(height, width)`` with the frequency response.
        """
        fy = np.fft.fftfreq(height)
        fx = np.fft.fftfreq(width)

        a = self.a
        low = (a - 1) / (2 * a)
        high = (a + 1) / (2 * a)

        def _lanczos_1d(freq: np.ndarray) -> np.ndarray:
            f_abs = np.abs(freq)
            response = np.zeros_like(f_abs)
            response[f_abs <= low] = 1.0
            transition = (f_abs > low) & (f_abs <= high)
            if np.any(transition):
                response[transition] = (high - f_abs[transition]) / (high - low)
            return response

        return np.outer(_lanczos_1d(fy), _lanczos_1d(fx))

    def _fill_channel(
        self,
        channel: np.ndarray,
        mask_2d: np.ndarray,
        freq_response: np.ndarray,
        nn_indices: np.ndarray,
    ) -> np.ndarray:
        """Apply Papoulis-Gerchberg iteration to one channel.

        Args:
            channel: Single-channel 2-D array.
            mask_2d: 2-D boolean gap mask.
            freq_response: Pre-computed Lanczos frequency response.
            nn_indices: Nearest-neighbour index arrays from EDT.

        Returns:
            Filled single-channel array as ``float32``.
        """
        valid_mask = ~mask_2d

        result = channel.copy().astype(np.float64)

        # Initialize gaps with nearest-neighbour values
        gap_y, gap_x = np.where(mask_2d)
        nn_y = nn_indices[0, gap_y, gap_x]
        nn_x = nn_indices[1, gap_y, gap_x]
        result[gap_y, gap_x] = channel[nn_y, nn_x]

        for iteration in range(self.max_iterations):
            # Band-limit via FFT
            spectrum = np.fft.fft2(result)
            spectrum *= freq_response
            filtered = np.real(np.fft.ifft2(spectrum))

            # Restore known pixels, keep filtered values for gaps
            old_gaps = result[mask_2d].copy()
            result[mask_2d] = filtered[mask_2d]
            result[valid_mask] = channel[valid_mask]

            # Check convergence (RMS change on gap pixels)
            new_gaps = result[mask_2d]
            if old_gaps.size == 0:
                break
            rms_change = float(np.sqrt(np.mean((new_gaps - old_gaps) ** 2)))
            if rms_change < self.tolerance:
                logger.debug(
                    "Converged at iteration %d (RMS=%.2e).",
                    iteration,
                    rms_change,
                )
                break

        return result.astype(np.float32)

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Lanczos spectral gap-filling via Papoulis-Gerchberg iteration.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
                Shape ``(H, W)`` or broadcastable ``(H, W, C)``.
            meta: Optional metadata (CRS, transform, band names, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        height, width = degraded.shape[:2]

        valid_mask = ~mask_2d
        if not np.any(valid_mask):
            logger.debug("No valid pixels found; returning input copy.")
            return self._finalize(degraded.copy())

        logger.debug(
            "Building Lanczos frequency response (a=%d) for %dx%d image.",
            self.a,
            height,
            width,
        )
        freq_response = self._build_frequency_response(height, width)

        _, nn_indices = distance_transform_edt(
            ~valid_mask,
            return_distances=True,
            return_indices=True,
        )

        def _channel_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            return self._fill_channel(ch, mask_2d, freq_response, nn_indices)

        if degraded.ndim == 3:
            logger.debug("Processing %d channels.", degraded.shape[2])

        result = self._apply_channelwise(degraded, mask_2d, _channel_fn)
        return self._finalize(result)
