"""DINEOF (Data Interpolating EOF) iterative SVD reconstruction."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from pdi_pipeline.exceptions import (
    DimensionError,
)
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DEFAULT_MAX_ITERATIONS = 100
"""Default maximum number of DINEOF iterations."""

_DEFAULT_TOLERANCE = 1e-4
"""Default RMS convergence threshold."""

_MIN_MODES = 1
"""Minimum number of EOF modes."""


class DINEOFInterpolator(BaseMethod):
    """Iterative truncated-SVD (EOF/PCA) reconstruction for time-series data.

    Requires input shape (T, H, W) or (T, H, W, C).

    Citation: Wikipedia contributors. "Empirical orthogonal functions." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Empirical_orthogonal_functions
    """

    name = "dineof"

    def __init__(
        self,
        max_modes: int | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        tolerance: float = _DEFAULT_TOLERANCE,
    ) -> None:
        """Initialize DINEOF interpolator.

        Args:
            max_modes: Maximum number of EOF modes to use. If ``None``,
                automatically determined from the data dimensions.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence threshold (RMS change in reconstructed
                values between successive iterations).
        """
        self.max_modes = max_modes
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply DINEOF interpolation to recover missing pixels.

        Args:
            degraded: Input time series with shape ``(T, H, W)`` or
                ``(T, H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Boolean mask where ``True``/``1`` marks gap pixels.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed time series with gaps filled, dtype ``float32``,
            values clipped to ``[0, 1]``.

        Raises:
            DimensionError: If *degraded* is not 3D or 4D.
        """
        degraded = np.asarray(degraded, dtype=np.float32)
        if degraded.ndim not in (3, 4):
            msg = (
                "DINEOF requires a time series (T, H, W) or (T, H, W, C), "
                f"got ndim={degraded.ndim}"
            )
            raise DimensionError(msg)

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 4:
            mask_3d = np.any(mask_arr, axis=3)
        else:
            mask_3d = mask_arr.astype(bool)

        logger.debug(
            "DINEOF: shape=%s, gap_fraction=%.3f, max_modes=%s, tol=%.1e",
            degraded.shape,
            float(np.mean(mask_3d)),
            self.max_modes,
            self.tolerance,
        )

        result = np.zeros_like(degraded, dtype=np.float32)

        if degraded.ndim == 3:
            result = self._dineof_series(degraded, mask_3d)
        else:
            for ch in range(degraded.shape[3]):
                result[..., ch] = self._dineof_series(
                    degraded[..., ch], mask_3d
                )

        return self._finalize(result)

    def _dineof_series(self, series: NDArray, mask_3d: NDArray) -> NDArray:
        """Apply DINEOF-style iterative EOF reconstruction to a time series.

        Args:
            series: Array with shape ``(T, H, W)``.
            mask_3d: Boolean mask with shape ``(T, H, W)``, where ``True``
                indicates missing.

        Returns:
            Reconstructed series with shape ``(T, H, W)``.
        """
        n_timesteps, height, width = series.shape
        matrix = series.reshape(n_timesteps, height * width)
        missing = mask_3d.reshape(n_timesteps, height * width)

        if not np.any(missing):
            logger.debug("DINEOF: no missing pixels in series; returning as-is")
            return series.astype(np.float32, copy=False)

        observed = ~missing

        if not np.any(observed):
            logger.warning(
                "DINEOF: all pixels are missing; returning series unchanged"
            )
            return series.astype(np.float32, copy=False)

        filled = matrix.copy()

        # Initialize missing pixels with temporal mean for each pixel location
        masked = np.ma.array(matrix, mask=missing)
        pixel_means = (
            np.ma.mean(masked, axis=0)
            .filled(fill_value=float(np.mean(matrix[~missing])))
            .astype(np.float32)
        )

        filled[missing] = pixel_means[np.where(missing)[1]]

        # Determine number of modes
        max_modes = (
            min(n_timesteps, height * width)
            if self.max_modes is None
            else int(min(self.max_modes, n_timesteps, height * width))
        )
        max_modes = max(_MIN_MODES, max_modes)

        last_change = np.inf
        iteration = 0
        for iteration in range(self.max_iterations):
            # Compute temporal mean and center data
            mean_over_time = np.mean(filled, axis=0, keepdims=True)
            centered = filled - mean_over_time

            try:
                # SVD decomposition: X = U Sigma V^T
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                logger.warning(
                    "DINEOF: SVD failed at iteration %d; stopping early",
                    iteration,
                )
                break

            # Reconstruct using truncated EOF: X_rec = U_k Sigma_k V_k^T
            k_modes = int(min(max_modes, S.size))
            reconstructed = (
                U[:, :k_modes] @ (S[:k_modes, None] * Vt[:k_modes, :])
            ) + mean_over_time

            # Update missing values
            old_missing = filled[missing]
            filled[missing] = reconstructed[missing]
            new_missing = filled[missing]

            # Check convergence (RMS change)
            change = float(np.sqrt(np.mean((new_missing - old_missing) ** 2)))
            if change < self.tolerance:
                logger.debug(
                    "DINEOF: converged at iteration %d (change=%.2e)",
                    iteration,
                    change,
                )
                break
            if change >= last_change:
                logger.debug(
                    "DINEOF: change not decreasing at iteration %d "
                    "(%.2e >= %.2e); stopping",
                    iteration,
                    change,
                    last_change,
                )
                break
            last_change = change

        logger.debug(
            "DINEOF: finished after %d iterations, k_modes=%d",
            iteration + 1,
            max_modes,
        )

        return filled.reshape(n_timesteps, height, width).astype(
            np.float32, copy=False
        )
