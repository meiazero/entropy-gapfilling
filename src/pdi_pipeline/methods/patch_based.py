"""Patch-based methods: non-local means style exemplar fill."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from skimage.restoration import denoise_nl_means, estimate_sigma

from pdi_pipeline.exceptions import (
    InsufficientDataError,
)
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


class NonLocalMeansInterpolator(BaseMethod):
    r"""Non-local means interpolation for image gap-filling.

    Mathematical Formulation
    ------------------------
    For each pixel $i$ in the gap region, the restored value is a weighted
    average over all observed pixels $j$:

    $$\hat{u}(i) = \frac{\sum_{j} w(i, j)\, u(j)}{\sum_{j} w(i, j)}$$

    where the weight between two pixels is determined by the similarity of
    their surrounding patches:

    $$w(i, j) = \exp\!\Bigl(-\frac{\|P_i - P_j\|_2^2}{h^2}\Bigr)$$

    Here $P_i$ and $P_j$ are the (patch_size x patch_size) patches centred
    at $i$ and $j$, and $h$ is a filtering parameter that controls the decay
    of the weights.

    Citation
    --------
    Buades, A., Coll, B. and Morel, J.-M. (2005). "A non-local algorithm
    for image denoising." *Proceedings of the IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR)*, vol. 2, 60--65.
    """

    name = "non_local"

    def __init__(
        self,
        patch_size: int = 5,
        patch_distance: int = 6,
        h_rel: float = 0.8,
    ) -> None:
        """Initialize non-local means interpolator.

        Args:
            patch_size: Size of patches used for denoising.
            patch_distance: Maximal distance in pixels where to search patches
                used for denoising.
            h_rel: Cut-off distance relative to the estimated noise standard
                deviation. Controls filter strength.
        """
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h_rel = h_rel

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply non-local means interpolation to recover missing pixels.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
                Shape ``(H, W)`` or broadcastable ``(H, W, C)``.
            meta: Optional metadata (CRS, transform, band names, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.

        Raises:
            InsufficientDataError: If no valid pixels are available to
                guide the fill.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        # Fill gaps with simple mean first to avoid NaNs in NL-means
        filled = degraded.copy()
        valid = ~mask_2d
        if not np.any(valid):
            raise InsufficientDataError(
                "Non-local means: no valid pixels to guide fill"
            )

        logger.debug(
            "Non-local means: filling %d gap pixels (patch_size=%d, "
            "patch_distance=%d, h_rel=%.2f).",
            int(np.sum(mask_2d)),
            self.patch_size,
            self.patch_distance,
            self.h_rel,
        )

        if degraded.ndim == 3:
            for ch in range(degraded.shape[2]):
                filled[..., ch] = self._fill_channel(
                    filled[..., ch], degraded[..., ch], mask_2d, valid
                )
        else:
            filled = self._fill_channel(filled, degraded, mask_2d, valid)

        return self._finalize(filled)

    def _fill_channel(
        self,
        channel: NDArray[np.float32],
        original: NDArray[np.float32],
        mask_2d: NDArray[np.bool_],
        valid: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Apply non-local means denoising to a single channel.

        Args:
            channel: Channel data (will be modified in-place for gap init).
            original: Original channel data for restoring known pixels.
            mask_2d: Boolean mask where True indicates missing pixels.
            valid: Boolean mask where True indicates known pixels.

        Returns:
            Denoised channel with known pixels restored.
        """
        channel[mask_2d] = float(channel[valid].mean())
        sigma = estimate_sigma(channel, channel_axis=None)
        h = self.h_rel * sigma
        denoised = denoise_nl_means(
            channel,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            h=h,
            channel_axis=None,
            fast_mode=True,
        )
        denoised[valid] = original[valid]
        return denoised


class ExemplarBasedInterpolator(BaseMethod):
    r"""Exemplar-based inpainting via biharmonic equation.

    Mathematical Formulation
    ------------------------
    The missing region $\Omega$ is filled by solving the biharmonic equation:

    $$\nabla^4 u = \Delta(\Delta u) = 0 \quad \text{in } \Omega$$

    subject to the Dirichlet boundary conditions $u = f$ on $\partial\Omega$,
    where $f$ denotes the known pixel values at the boundary of the gap.

    The biharmonic operator $\nabla^4$ is the composition of two Laplacians
    and yields a $C^1$-smooth surface that minimises the bending energy:

    $$E[u] = \iint_\Omega (\Delta u)^2 \, dx\,dy$$

    Citation
    --------
    Criminisi, A., Perez, P. and Toyama, K. (2004). "Region filling and
    object removal by exemplar-based image inpainting." *IEEE Transactions
    on Image Processing*, 13(9), 1200--1212.
    """

    name = "exemplar_based"

    def __init__(self) -> None:
        """Initialize exemplar-based interpolator."""

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply exemplar-based (biharmonic) inpainting to recover missing pixels.

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
        from skimage.restoration import inpaint

        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        logger.debug(
            "Exemplar-based: filling %d gap pixels via biharmonic inpainting.",
            int(np.sum(mask_2d)),
        )

        if degraded.ndim == 3:
            res = np.zeros_like(degraded)
            for i in range(degraded.shape[2]):
                res[..., i] = inpaint.inpaint_biharmonic(
                    degraded[..., i], mask_2d, channel_axis=None
                )
        else:
            res = inpaint.inpaint_biharmonic(
                degraded, mask_2d, channel_axis=None
            )
        return self._finalize(res)
