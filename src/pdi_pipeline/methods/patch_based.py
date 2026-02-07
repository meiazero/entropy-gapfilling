"""Patch-based methods: non-local means style exemplar fill."""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

from pdi_pipeline.methods.base import BaseMethod


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
        mask_bool = self._normalize_mask(mask)
        if degraded.shape[:2] != mask_bool.shape:
            raise ValueError("Image and mask spatial dimensions must match")
        # Fill gaps with simple mean first to avoid NaNs in NL-means
        filled = degraded.copy()
        valid = ~mask_bool
        if not np.any(valid):
            raise ValueError("Non-local means: no valid pixels to guide fill")
        if degraded.ndim == 3:
            for ch in range(degraded.shape[2]):
                channel = filled[..., ch]
                channel[mask_bool] = float(channel[valid].mean())
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
                denoised[valid] = degraded[..., ch][valid]
                filled[..., ch] = denoised
        else:
            channel = filled
            channel[mask_bool] = float(channel[valid].mean())
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
            denoised[valid] = degraded[valid]
            filled = denoised
        return self._finalize(filled)


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

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        from skimage.restoration import inpaint

        mask_bool = self._normalize_mask(mask)
        if degraded.ndim == 3:
            res = np.zeros_like(degraded)
            for i in range(degraded.shape[2]):
                res[..., i] = inpaint.inpaint_biharmonic(
                    degraded[..., i], mask_bool, channel_axis=None
                )
        else:
            res = inpaint.inpaint_biharmonic(
                degraded, mask_bool, channel_axis=None
            )
        return self._finalize(res)
