"""Patch-based methods: non-local means style exemplar fill."""

from __future__ import annotations

import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

from pdi_pipeline.methods.base import BaseMethod


class NonLocalMeansInterpolator(BaseMethod):
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
    """Exemplar-based inpainting proxy.

    Uses biharmonic inpainting to solve for missing pixels while maintaining
    global smoothness and boundary constraints.
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
