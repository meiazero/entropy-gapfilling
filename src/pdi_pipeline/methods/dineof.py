"""DINEOF (Data Interpolating Empirical Orthogonal Functions) interpolation.

DINEOF is an iterative method for reconstructing missing data in geophysical
datasets using Empirical Orthogonal Functions (EOF), also known as Principal
Component Analysis (PCA). It is particularly effective for spatiotemporal data
but can be adapted for spatial-only reconstruction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pdi_pipeline.methods.base import BaseMethod


class DINEOFInterpolator(BaseMethod):
    r"""DINEOF (Data Interpolating Empirical Orthogonal Functions) interpolation.

    DINEOF is an iterative method for reconstructing missing data in geophysical
    datasets using Empirical Orthogonal Functions (EOF), also known as Principal
    Component Analysis (PCA). It is particularly effective for spatiotemporal data
    but can be adapted for spatial-only reconstruction.

    Mathematical Formulation:
        The DINEOF algorithm iteratively reconstructs missing data by:

        1. Initializing missing values (e.g., with spatial mean).
        2. Computing EOF decomposition:
           $$X = U \Sigma V^T$$
        3. Reconstructing data using truncated EOF with $k$ modes:
           $$X_{rec} = U_k \Sigma_k V_k^T$$
        4. Updating missing values with reconstructed values.
        5. Repeating until convergence.

        The convergence criterion is based on the RMS change in reconstructed values:
        $$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i \in \text{missing}} (x_i^{(k+1)} - x_i^{(k)})^2}$$

        where $N$ is the number of missing pixels, and $k$ is the iteration number.

    Note:
        DINEOF is a spatiotemporal method designed for time-series data with shape
        (T, H, W, C). It is not intended for single images. For single-image gap
        filling, use spatial methods such as Kriging, RBF, or other interpolators.

    Citation: Wikipedia contributors. "Empirical orthogonal functions." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Empirical_orthogonal_functions
    """

    name = "dineof"

    def __init__(
        self,
        max_modes: int | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> None:
        """Initialize DINEOF interpolator.

        Args:
            max_modes: Maximum number of EOF modes to use. If None, automatically determined.
            max_iterations: Maximum number of iterations
            tolerance: Convergence threshold (RMS change in reconstructed values)
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
            degraded: Input time series with shape (T, H, W) or (T, H, W, C)
            mask: Boolean mask where True indicates missing pixels
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed time series with gaps filled
        """
        if degraded.ndim not in (3, 4):
            raise ValueError("DINEOF requires a time series (T, H, W, [C])")

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 4:
            mask_3d = np.any(mask_arr, axis=3)
        else:
            mask_3d = mask_arr.astype(bool)

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
            series: Array with shape (T, H, W).
            mask_3d: Boolean mask with shape (T, H, W), where True indicates missing.

        Returns:
            Reconstructed series with shape (T, H, W).
        """
        n_timesteps, height, width = series.shape
        matrix = series.reshape(n_timesteps, height * width)
        missing = mask_3d.reshape(n_timesteps, height * width)

        if not np.any(missing):
            return series.astype(np.float32, copy=False)

        observed = ~missing

        if not np.any(observed):
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
        max_modes = max(1, max_modes)

        last_change = np.inf
        for _iteration in range(self.max_iterations):
            # Compute temporal mean and center data
            mean_over_time = np.mean(filled, axis=0, keepdims=True)
            centered = filled - mean_over_time

            try:
                # SVD decomposition: X = U Σ V^T
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                break

            # Reconstruct using truncated EOF: X_rec = U_k Σ_k V_k^T
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
            if change < self.tolerance or change >= last_change:
                break
            last_change = change

        return filled.reshape(n_timesteps, height, width).astype(
            np.float32, copy=False
        )
