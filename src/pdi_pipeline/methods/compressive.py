"""Compressive Sensing with L1 minimization in transform domain.

Compressive Sensing (CS) exploits sparsity of natural images in appropriate
transform domains (wavelet, DCT, etc.) to reconstruct from incomplete measurements.
"""

from __future__ import annotations

import numpy as np
import pywt
from scipy.fftpack import dct, idct

from pdi_pipeline.methods.base import BaseMethod


class L1WaveletInpainting(BaseMethod):
    r"""Compressive Sensing with L1 minimization in transform domain.

    Compressive Sensing (CS) exploits sparsity of natural images in appropriate
    transform domains (wavelet, DCT, etc.) to reconstruct from incomplete measurements.

    Mathematical Formulation:
        Compressive Sensing recovers a signal from incomplete measurements by exploiting sparsity.
        The reconstruction problem is:

        $$\min_{\alpha} \|\alpha\|_1 \quad \text{subject to} \quad \|y - \Phi \Psi \alpha\|_2 \leq \epsilon$$

        where:
        - $\alpha$ is the sparse coefficient vector in the transform domain.
        - $\Psi$ is the sparsifying transform (Wavelet, DCT).
        - $\Phi$ is the measurement matrix (sampling mask).
        - $y$ are the observed pixel values.
        - $\epsilon$ is the noise tolerance.

        This is solved iteratively via soft thresholding (ISTA/FISTA):
        1. Forward transform: $\alpha^{(k)} = \Psi(x^{(k)})$
        2. Soft threshold: $\alpha^{(k+1)} = \mathcal{S}_{\lambda}(\alpha^{(k)})$
        3. Inverse transform: $x^{(k+1)} = \Psi^{-1}(\alpha^{(k+1)})$
        4. Data fidelity projection: $x^{(k+1)}[\text{observed}] = y$

    Citation: Wikipedia contributors. "Compressed sensing." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Compressed_sensing
    """

    name = "cs_wavelet"

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 3,
        lambda_param: float = 0.05,
        max_iterations: int = 100,
    ) -> None:
        """Initialize Compressive Sensing L1 interpolator.

        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'sym4').
                     'db4' provides good sparsity for satellite imagery.
            level: Decomposition level. For 64x64 patches, level=3 is appropriate.
            lambda_param: Regularization parameter balancing sparsity and data fidelity.
                          Lower values (0.01-0.1) preserve more detail. Default reduced
                          from 0.1 to 0.05 for better spectral preservation.
            max_iterations: Maximum iterations for optimization.
        """
        self.wavelet = wavelet
        self.level = level
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Compressive Sensing with L1 minimization using wavelet transform.

        Args:
            degraded: Array with missing data (e.g., NaN or masked pixels).
            mask: Binary mask where 1 indicates missing pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed array with same shape as degraded.
        """
        mask_2d = self._normalize_mask(mask)
        if degraded.ndim == 3 and degraded.shape[2] > 1:
            raise ValueError(
                "CS wavelet inpainting implemented for single-channel only"
            )
        degraded = np.asarray(degraded, dtype=np.float32)
        if degraded.ndim == 3:
            degraded = degraded[..., 0]

        result = self._cs_l1_channel(degraded, mask_2d)
        return self._finalize(result)

    def _cs_l1_channel(
        self, channel_data: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Apply CS L1 minimization to a single channel using iterative soft thresholding.

        Args:
            channel_data: Single channel image data
            mask: Boolean mask where True indicates missing pixels

        Returns:
            Reconstructed channel
        """
        mask_flat = mask.reshape(-1)
        if not np.any(~mask_flat):
            return channel_data

        # Initialize with simple interpolation
        reconstructed = channel_data.copy()
        valid_mask = ~mask
        if valid_mask.sum() > 0:
            mean_val = channel_data[valid_mask].mean()
            reconstructed[mask] = mean_val

        # Iterative refinement using wavelet sparsity (ISTA algorithm)
        for _iteration in range(
            min(self.max_iterations, 20)
        ):  # Limit iterations for efficiency
            # Wavelet decomposition
            coeffs = pywt.wavedec2(
                reconstructed, self.wavelet, level=self.level
            )

            # Flatten coefficients
            coeffs_flat, slices = pywt.coeffs_to_array(coeffs)

            # Soft thresholding (proximal operator for L1 norm)
            threshold = self.lambda_param * np.abs(coeffs_flat).max()
            coeffs_flat_thresh = np.sign(coeffs_flat) * np.maximum(
                np.abs(coeffs_flat) - threshold, 0
            )

            # Reconstruct from thresholded coefficients
            coeffs_thresh = pywt.array_to_coeffs(
                coeffs_flat_thresh, slices, output_format="wavedec2"
            )
            reconstructed_new = pywt.waverec2(coeffs_thresh, self.wavelet)

            # Handle size mismatch
            if reconstructed_new.shape != channel_data.shape:
                reconstructed_new = reconstructed_new[
                    : channel_data.shape[0], : channel_data.shape[1]
                ]

            # Enforce data fidelity on known pixels
            reconstructed_new[valid_mask] = channel_data[valid_mask]

            # Check convergence
            change = np.linalg.norm(
                reconstructed_new[mask] - reconstructed[mask]
            )
            reconstructed = np.asarray(reconstructed_new, dtype=np.float32)

            if change < 1e-4:
                break

        return reconstructed


class L1DCTInpainting(BaseMethod):
    r"""Compressive Sensing with L1 minimization using DCT transform.

    Similar to L1WaveletInpainting but uses DCT as the sparsifying transform.

    Mathematical Formulation:
        Same as L1WaveletInpainting but with $\Psi$ being the DCT transform:

        $$\min_{\alpha} \|\alpha\|_1 \quad \text{subject to} \quad \|y - \Phi \Psi \alpha\|_2 \leq \epsilon$$

        where $\Psi$ is the 2D DCT transform.

    Citation: Wikipedia contributors. "Compressed sensing." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Compressed_sensing
    """

    name = "cs_dct"

    def __init__(
        self, lambda_param: float = 0.05, max_iterations: int = 100
    ) -> None:
        """Initialize CS DCT interpolator.

        Args:
            lambda_param: Regularization parameter balancing sparsity and data fidelity.
            max_iterations: Maximum iterations for optimization.
        """
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply Compressive Sensing with L1 minimization using DCT transform.

        Args:
            degraded: Array with missing data (e.g., NaN or masked pixels).
            mask: Binary mask where 1 indicates missing pixels to fill.
            meta: Optional metadata (crs, transform, bands, etc.).

        Returns:
            Reconstructed array with same shape as degraded.
        """
        mask_2d = self._normalize_mask(mask)
        if degraded.ndim == 3 and degraded.shape[2] > 1:
            raise ValueError(
                "CS DCT inpainting implemented for single-channel only"
            )
        degraded = np.asarray(degraded, dtype=np.float32)
        if degraded.ndim == 3:
            degraded = degraded[..., 0]

        result = self._solve_dct_cs(degraded, mask_2d)
        return self._finalize(result)

    def _solve_dct_cs(
        self, channel_data: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Solve CS problem using DCT transform with iterative soft thresholding.

        Args:
            channel_data: Original channel data
            mask: Boolean mask

        Returns:
            Reconstructed image
        """
        # Initialize with simple interpolation
        reconstructed = channel_data.copy()
        valid_mask = ~mask
        if valid_mask.sum() > 0:
            mean_val = channel_data[valid_mask].mean()
            reconstructed[mask] = mean_val

        # Iterative refinement using DCT sparsity (ISTA algorithm)
        for _iteration in range(min(self.max_iterations, 20)):
            # DCT transform
            dct_coeffs = dct(dct(reconstructed.T, norm="ortho").T, norm="ortho")

            # Soft thresholding
            threshold = self.lambda_param * np.abs(dct_coeffs).max()
            dct_coeffs_thresh = np.sign(dct_coeffs) * np.maximum(
                np.abs(dct_coeffs) - threshold, 0
            )

            # Inverse DCT
            reconstructed_new = idct(
                idct(dct_coeffs_thresh.T, norm="ortho").T, norm="ortho"
            )

            # Enforce data fidelity on known pixels
            reconstructed_new[valid_mask] = channel_data[valid_mask]

            # Check convergence
            change = np.linalg.norm(
                reconstructed_new[mask] - reconstructed[mask]
            )
            reconstructed = np.asarray(reconstructed_new, dtype=np.float32)

            if change < 1e-4:
                break

        return np.asarray(reconstructed, dtype=np.float32)
