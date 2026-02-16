"""Transform-based and regularization inpainting: DCT, Wavelet, TV.

These methods exploit sparsity in transform domains (frequency, wavelet) or
promote piecewise smoothness (Total Variation) to reconstruct missing data.
"""

from __future__ import annotations

import logging

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.fftpack import dct, idct

from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

_DCT_CONVERGENCE_TOL = 1e-4
_WAVELET_CONVERGENCE_TOL = 1e-4
_TV_CONVERGENCE_TOL = 1e-4


class DCTInpainting(BaseMethod):
    r"""DCT (Discrete Cosine Transform) based inpainting.

    DCT inpainting exploits the energy compaction property of DCT in the frequency domain.
    Missing pixels are recovered by minimizing a cost function that combines data fidelity
    on known pixels with sparsity or smoothness in the DCT domain.

    Mathematical Formulation:
        The reconstruction is obtained by solving:

        $$\min_{\alpha} \|\alpha\|_1 \quad \text{subject to data fidelity constraints}$$

        where $\alpha$ are the DCT coefficients.

        Iterative thresholding algorithm:
        1. Forward DCT: $C = \text{DCT}(f)$
        2. Soft thresholding: $\hat{C} = \text{sign}(C) \cdot \max(|C| - \lambda, 0)$
        3. Inverse DCT: $f_{\text{new}} = \text{IDCT}(\hat{C})$
        4. Data fidelity: $f_{\text{new}}[\text{known}] = f_{\text{orig}}[\text{known}]$
        5. Repeat until convergence

    Citation: Wikipedia contributors. "Discrete cosine transform." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Discrete_cosine_transform
    """

    name = "dct"

    def __init__(
        self, max_iterations: int = 50, lambda_param: float = 0.05
    ) -> None:
        """Initialize DCT inpainting.

        Args:
            max_iterations: Maximum number of optimization iterations.
            lambda_param: Regularization parameter controlling DCT sparsity.
                          Lower values (0.01-0.05) preserve more detail for
                          satellite imagery. Higher values (0.1-0.2) produce
                          smoother results. Default reduced from 0.1 to 0.05
                          for better preservation of spectral characteristics.
        """
        self.max_iterations = max_iterations
        self.lambda_param = lambda_param

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply DCT-based inpainting to recover missing pixels.

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

        result = self._apply_channelwise(
            degraded, mask_2d, self._dct_inpaint_channel
        )
        return self._finalize(result)

    def _dct_inpaint_channel(
        self, channel_data: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """Apply DCT inpainting to a single channel.

        Args:
            channel_data: Single channel image data.
            mask: Boolean mask where True indicates missing pixels.

        Returns:
            Reconstructed channel.
        """
        # Initialize missing pixels with local mean
        reconstructed = channel_data.copy()
        valid_mask = ~mask

        if valid_mask.sum() == 0:
            return reconstructed

        # Initialize gaps with mean of valid values
        mean_val = channel_data[valid_mask].mean()
        reconstructed[mask] = mean_val

        # Iterative DCT-based reconstruction
        iteration = 0
        change = float("inf")
        for iteration in range(self.max_iterations):
            # Forward DCT
            dct_coeffs = dct(dct(reconstructed.T, norm="ortho").T, norm="ortho")

            # Soft thresholding for sparsity
            threshold = self.lambda_param * np.abs(dct_coeffs).max()
            dct_coeffs = np.sign(dct_coeffs) * np.maximum(
                np.abs(dct_coeffs) - threshold, 0
            )

            # Inverse DCT
            reconstructed_new = idct(
                idct(dct_coeffs.T, norm="ortho").T, norm="ortho"
            )

            # Enforce data fidelity on known pixels
            reconstructed_new[valid_mask] = channel_data[valid_mask]

            # Check convergence
            change = np.linalg.norm(
                reconstructed_new[mask] - reconstructed[mask]
            )
            reconstructed = reconstructed_new.astype(np.float32, copy=False)

            if change < _DCT_CONVERGENCE_TOL:
                logger.debug(
                    "DCT converged at iteration %d (change=%.2e).",
                    iteration + 1,
                    change,
                )
                break

        logger.debug(
            "DCT finished after %d iterations (final change=%.2e).",
            iteration + 1,
            change,
        )
        return reconstructed.astype(np.float32)


class WaveletInpainting(BaseMethod):
    r"""Wavelet-based inpainting.

    Wavelet inpainting exploits the multi-scale decomposition property of wavelets
    to reconstruct missing data. Natural images are typically sparse in wavelet domain,
    allowing effective reconstruction through iterative thresholding.

    Mathematical Formulation:
        The problem is formulated as sparse recovery in the wavelet domain:

        $$\min_f \|W(f)\|_1 \quad \text{subject to} \quad f|_{\Omega} = y$$

        where $W$ is the wavelet transform, $\Omega$ is the set of known pixels, and $y$ are
        the observed values.

        The iterative solution:
        1. Wavelet decomposition: $C = W(f)$
        2. Soft thresholding of detail coefficients: $\hat{C} = \text{thresh}(C, \lambda)$
        3. Wavelet reconstruction: $f_{\text{new}} = W^{-1}(\hat{C})$
        4. Data fidelity: $f_{\text{new}}[\text{known}] = y$
        5. Repeat until convergence

    Citation: Wikipedia contributors. "Wavelet transform." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Wavelet_transform
    """

    name = "wavelet"

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 3,
        max_iterations: int = 50,
        lambda_param: float = 0.05,
    ) -> None:
        """Initialize wavelet inpainting.

        Args:
            wavelet: Wavelet family (e.g., 'db4', 'haar', 'sym4', 'coif1').
                     'db4' (Daubechies-4) provides good balance between
                     smoothness and localization for satellite imagery.
            level: Decomposition level. For 64x64 patches, level=3 is
                   appropriate (8x8 approximation coefficients).
            max_iterations: Maximum number of iterations.
            lambda_param: Regularization parameter for wavelet sparsity.
                          Lower values preserve more texture detail.
        """
        self.wavelet = wavelet
        self.level = level
        self.max_iterations = max_iterations
        self.lambda_param = lambda_param

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply wavelet-based inpainting to recover missing pixels.

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

        result = self._apply_channelwise(
            degraded, mask_2d, self._wavelet_inpaint_channel
        )
        return self._finalize(result)

    def _wavelet_inpaint_channel(
        self, channel_data: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """Apply wavelet inpainting to a single channel.

        Args:
            channel_data: Single channel image data.
            mask: Boolean mask where True indicates missing pixels.

        Returns:
            Reconstructed channel.
        """
        reconstructed = np.asarray(channel_data, dtype=np.float32).copy()
        valid_mask = ~mask

        if valid_mask.sum() == 0:
            return reconstructed

        # Initialize gaps with mean
        mean_val = channel_data[valid_mask].mean()
        reconstructed[mask] = mean_val

        # Iterative wavelet-based reconstruction
        iteration = 0
        change = float("inf")
        for iteration in range(self.max_iterations):
            # Wavelet decomposition
            coeffs = pywt.wavedec2(
                reconstructed, self.wavelet, level=self.level
            )

            # Soft thresholding on detail coefficients
            coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
            for detail_level in coeffs[1:]:
                thresh_detail = []
                for detail_coeff in detail_level:
                    if detail_coeff is not None:
                        threshold = (
                            self.lambda_param * np.abs(detail_coeff).max()
                        )
                        thresholded = np.sign(detail_coeff) * np.maximum(
                            np.abs(detail_coeff) - threshold, 0
                        )
                        thresh_detail.append(thresholded)
                    else:
                        thresh_detail.append(None)
                coeffs_thresh.append(tuple(thresh_detail))

            # Wavelet reconstruction
            reconstructed_new = pywt.waverec2(coeffs_thresh, self.wavelet)

            # Handle size mismatch due to wavelet padding
            if reconstructed_new.shape != channel_data.shape:
                reconstructed_new = reconstructed_new[
                    : channel_data.shape[0], : channel_data.shape[1]
                ]

            reconstructed_new = np.real(reconstructed_new).astype(
                np.float32, copy=False
            )

            # Enforce data fidelity on known pixels
            reconstructed_new[valid_mask] = channel_data[valid_mask]

            # Check convergence
            change = np.linalg.norm(
                reconstructed_new[mask] - reconstructed[mask]
            )
            reconstructed = reconstructed_new

            if change < _WAVELET_CONVERGENCE_TOL:
                logger.debug(
                    "Wavelet converged at iteration %d (change=%.2e).",
                    iteration + 1,
                    change,
                )
                break

        logger.debug(
            "Wavelet finished after %d iterations (final change=%.2e).",
            iteration + 1,
            change,
        )
        return np.real(reconstructed).astype(np.float32, copy=False)


class TVInpainting(BaseMethod):
    r"""Total Variation (TV) inpainting.

    TV inpainting is a variational method that reconstructs missing data by minimizing
    the total variation norm, which measures the amount of variation in the image.
    This promotes piecewise smooth solutions while preserving edges.

    Mathematical Formulation:
        The energy functional to minimize is:

        $$E(u) = \int_{\Omega} |\nabla u| \, dx + \frac{\lambda}{2} \int_{\Omega \setminus D} (u - f)^2 \, dx$$

        where:
        - $\int |\nabla u|$ is the Total Variation (TV) term promoting piecewise smoothness.
        - The second term enforces fidelity to known data $f$ outside missing region $D$.
        - $\lambda$ balances smoothness and data fidelity.

        Solution via primal-dual algorithm:
        1. Initialize primal variable $u$ and dual variables $p$
        2. Update dual: $p \leftarrow \text{proj}(p + \sigma \nabla \bar{u})$
        3. Update primal: $u \leftarrow (u + \tau \text{div}(p) + \tau \lambda M f) / (1 + \tau \lambda M)$
        4. Extrapolation: $\bar{u} \leftarrow u + \theta (u - u_{\text{old}})$

    Citation: Wikipedia contributors. "Total variation denoising." Wikipedia, The Free Encyclopedia.
    https://en.wikipedia.org/wiki/Total_variation_denoising
    """

    name = "tv"

    def __init__(
        self,
        lambda_param: float = 0.1,
        max_iterations: int = 100,
    ) -> None:
        """Initialize TV inpainting.

        Args:
            lambda_param: Data fidelity weight. Higher values enforce
                stronger agreement with observed pixels.
            max_iterations: Maximum number of primal-dual iterations.
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
        """Apply Total Variation inpainting to recover missing pixels.

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

        result = self._apply_channelwise(
            degraded, mask_2d, self._tv_inpaint_channel
        )
        return self._finalize(result)

    def _tv_inpaint_channel(
        self, channel_data: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """Apply TV inpainting to a single channel.

        Args:
            channel_data: Single channel image data.
            mask: Boolean mask where True indicates missing pixels.

        Returns:
            Reconstructed channel.
        """
        if not np.any(mask):
            return channel_data.astype(np.float32, copy=False)

        valid_mask = ~mask
        if not np.any(valid_mask):
            return channel_data.astype(np.float32, copy=False)

        image = np.asarray(channel_data, dtype=np.float32)
        filled = image.copy()
        filled[mask] = float(np.mean(image[valid_mask]))

        weight = valid_mask.astype(np.float32)
        data_term = weight * image

        tau = 0.125
        sigma = 0.125
        theta = 1.0
        lambda_param = float(self.lambda_param)

        dual_x = np.zeros_like(filled)
        dual_y = np.zeros_like(filled)
        primal = filled
        primal_bar = primal

        for _iteration in range(self.max_iterations):
            grad_x = np.roll(primal_bar, -1, axis=1) - primal_bar
            grad_y = np.roll(primal_bar, -1, axis=0) - primal_bar
            grad_x[:, -1] = 0.0
            grad_y[-1, :] = 0.0

            dual_x_new = dual_x + sigma * grad_x
            dual_y_new = dual_y + sigma * grad_y
            dual_norm = np.maximum(
                1.0, np.sqrt(dual_x_new * dual_x_new + dual_y_new * dual_y_new)
            )
            dual_x = dual_x_new / dual_norm
            dual_y = dual_y_new / dual_norm

            div = dual_x - np.roll(dual_x, 1, axis=1)
            div[:, 0] = dual_x[:, 0]
            div += dual_y - np.roll(dual_y, 1, axis=0)
            div[0, :] += dual_y[0, :]

            primal_prev = primal
            numerator = primal + tau * div + tau * lambda_param * data_term
            denominator = 1.0 + tau * lambda_param * weight
            primal = numerator / denominator
            primal_bar = primal + theta * (primal - primal_prev)

        logger.debug(
            "TV primal-dual finished after %d iterations.", self.max_iterations
        )
        return primal.astype(np.float32, copy=False)
