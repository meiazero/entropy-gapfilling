r"""Compressive Sensing gap-filling with $\ell_1$ minimisation in transform domains.

Compressive Sensing (CS) exploits the sparsity of natural images in
appropriate transform domains (wavelet, DCT) to reconstruct signals from
incomplete observations.  Missing pixels are recovered by iterative soft
thresholding (ISTA), which alternates between promoting sparsity in the
transform domain and enforcing data fidelity on the observed pixels.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.fftpack import dct, idct

from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

_CS_WAVELET_CONVERGENCE_TOL = 1e-5
_CS_DCT_CONVERGENCE_TOL = 1e-5


class L1WaveletInpainting(BaseMethod):
    r"""Compressive Sensing gap-filling via $\ell_1$-wavelet minimisation.

    Mathematical Formulation
    ------------------------
    Let $\Psi$ denote the discrete wavelet transform and $\alpha = \Psi(f)$
    the wavelet coefficients of image $f$.  Natural images are approximately
    sparse in the wavelet domain ($\|\alpha\|_0 \ll N$).

    The reconstruction solves the basis-pursuit denoising problem:

    $$\min_{\alpha} \;\|\alpha\|_1
      \quad\text{subject to}\quad
      \|y - \Phi\,\Psi^{-1}\alpha\|_2 \le \varepsilon$$

    where $\Phi$ is the sampling operator (diagonal mask matrix) and $y$
    are the observed pixel values.

    **ISTA (Iterative Soft Thresholding Algorithm):**

    $$\alpha^{(k)} = \Psi\!\left(u^{(k)}\right)$$
    $$\hat\alpha^{(k)} = \mathcal{S}_\lambda\!\left(\alpha^{(k)}\right)
      = \operatorname{sign}(\alpha^{(k)})\,
        \max\!\left(|\alpha^{(k)}| - \lambda, 0\right)$$
    $$v^{(k)} = \Psi^{-1}\!\left(\hat\alpha^{(k)}\right)$$
    $$u^{(k+1)} = M \odot f + (1-M) \odot v^{(k)}$$

    where $M$ is the known-pixel mask and $\lambda$ is the thresholding
    parameter controlling the sparsity-fidelity trade-off.

    Properties
    ----------
    * **Complexity per iteration:** $O(HW)$ for the wavelet transform.
    * **Convergence:** Monotone; guaranteed for convex $\ell_1$ + quadratic.
    * **Sparsity:** Wavelet basis captures both smooth regions and edges via
      multi-resolution analysis.

    Citation
    --------
    Candes, E. J. and Wakin, M. B. (2008). "An introduction to compressive
    sampling." *IEEE Signal Processing Magazine*, 25(2), 21--30.

    Daubechies, I., Defrise, M. and De Mol, C. (2004). "An iterative
    thresholding algorithm for linear inverse problems with a sparsity
    constraint." *Communications on Pure and Applied Mathematics*,
    57(11), 1413--1457.
    """

    name = "cs_wavelet"

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 3,
        lambda_param: float = 0.05,
        max_iterations: int = 100,
    ) -> None:
        """Initialize CS wavelet interpolator.

        Args:
            wavelet: Wavelet family (e.g. 'db4', 'haar', 'sym4').
            level: Decomposition level.
            lambda_param: Soft-thresholding parameter (sparsity weight).
            max_iterations: Maximum ISTA iterations.
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
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply CS wavelet inpainting to recover missing pixels.

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
            degraded,
            mask_2d,
            self._cs_wavelet_channel,
        )
        return self._finalize(result)

    def _cs_wavelet_channel(
        self, channel: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """ISTA reconstruction for a single 2-D channel.

        Args:
            channel: Single channel image data.
            mask: Boolean mask where True indicates missing pixels.

        Returns:
            Reconstructed channel.
        """
        valid_mask = ~mask
        if not np.any(valid_mask):
            return channel.copy()

        reconstructed = channel.copy().astype(np.float64)
        reconstructed[mask] = float(np.mean(channel[valid_mask]))

        iteration = 0
        change = float("inf")
        for iteration in range(self.max_iterations):
            coeffs = pywt.wavedec2(
                reconstructed, self.wavelet, level=self.level
            )
            coeffs_flat, slices = pywt.coeffs_to_array(coeffs)

            threshold = self.lambda_param * float(np.abs(coeffs_flat).max())
            coeffs_flat = np.sign(coeffs_flat) * np.maximum(
                np.abs(coeffs_flat) - threshold, 0.0
            )

            coeffs_thresh = pywt.array_to_coeffs(
                coeffs_flat, slices, output_format="wavedec2"
            )
            new = pywt.waverec2(coeffs_thresh, self.wavelet)
            if new.shape != channel.shape:
                new = new[: channel.shape[0], : channel.shape[1]]

            new[valid_mask] = channel[valid_mask]

            change = float(np.linalg.norm(new[mask] - reconstructed[mask]))
            reconstructed = new
            if change < _CS_WAVELET_CONVERGENCE_TOL:
                logger.debug(
                    "CS wavelet converged at iteration %d (change=%.2e).",
                    iteration + 1,
                    change,
                )
                break

        logger.debug(
            "CS wavelet finished after %d iterations (final change=%.2e).",
            iteration + 1,
            change,
        )
        return reconstructed.astype(np.float32)


class L1DCTInpainting(BaseMethod):
    r"""Compressive Sensing gap-filling via $\ell_1$-DCT minimisation.

    Mathematical Formulation
    ------------------------
    Identical to ``L1WaveletInpainting`` but with $\Psi$ being the 2-D
    Discrete Cosine Transform (type-II, orthonormal):

    $$\Psi = \text{DCT-II}_{2D}, \qquad
      \alpha = \Psi(f), \qquad
      f = \Psi^{-1}(\alpha).$$

    The DCT concentrates energy in low-frequency coefficients for smooth
    natural images, providing an alternative sparsifying basis to wavelets.

    The ISTA iteration is:

    $$\alpha^{(k)} = \text{DCT}(u^{(k)})$$
    $$\hat\alpha^{(k)} = \mathcal{S}_\lambda(\alpha^{(k)})$$
    $$v^{(k)} = \text{IDCT}(\hat\alpha^{(k)})$$
    $$u^{(k+1)} = M \odot f + (1-M) \odot v^{(k)}$$

    Properties
    ----------
    * **Complexity per iteration:** $O(HW \log(HW))$ via FFT-based DCT.
    * **Basis:** DCT is real-valued and orthonormal; no complex arithmetic.
    * **Boundary:** Implicit symmetric (mirror) boundary conditions avoid
      Gibbs-like ringing at image edges.

    Citation
    --------
    Candes, E. J. and Wakin, M. B. (2008). "An introduction to compressive
    sampling." *IEEE Signal Processing Magazine*, 25(2), 21--30.
    """

    name = "cs_dct"

    def __init__(
        self,
        lambda_param: float = 0.05,
        max_iterations: int = 100,
    ) -> None:
        """Initialize CS DCT interpolator.

        Args:
            lambda_param: Soft-thresholding parameter (sparsity weight).
            max_iterations: Maximum ISTA iterations.
        """
        self.lambda_param = lambda_param
        self.max_iterations = max_iterations

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply CS DCT inpainting to recover missing pixels.

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
            degraded,
            mask_2d,
            self._cs_dct_channel,
        )
        return self._finalize(result)

    def _cs_dct_channel(
        self, channel: NDArray[np.float32], mask: NDArray[np.bool_]
    ) -> NDArray[np.float32]:
        """ISTA reconstruction for a single 2-D channel using DCT.

        Args:
            channel: Single channel image data.
            mask: Boolean mask where True indicates missing pixels.

        Returns:
            Reconstructed channel.
        """
        valid_mask = ~mask
        if not np.any(valid_mask):
            return channel.copy()

        reconstructed = channel.copy().astype(np.float64)
        reconstructed[mask] = float(np.mean(channel[valid_mask]))

        iteration = 0
        change = float("inf")
        for iteration in range(self.max_iterations):
            coeffs = dct(dct(reconstructed.T, norm="ortho").T, norm="ortho")

            threshold = self.lambda_param * float(np.abs(coeffs).max())
            coeffs = np.sign(coeffs) * np.maximum(
                np.abs(coeffs) - threshold, 0.0
            )

            new = idct(idct(coeffs.T, norm="ortho").T, norm="ortho")
            new[valid_mask] = channel[valid_mask]

            change = float(np.linalg.norm(new[mask] - reconstructed[mask]))
            reconstructed = new
            if change < _CS_DCT_CONVERGENCE_TOL:
                logger.debug(
                    "CS DCT converged at iteration %d (change=%.2e).",
                    iteration + 1,
                    change,
                )
                break

        logger.debug(
            "CS DCT finished after %d iterations (final change=%.2e).",
            iteration + 1,
            change,
        )
        return reconstructed.astype(np.float32)
