"""Base interface for interpolation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseMethod(ABC):
    """Interface for interpolation methods.

    Implementations should be stateless or carry only lightweight parameters.
    For methods requiring fit (e.g., kriging), implement fit() accordingly.
    """

    name: str

    @abstractmethod
    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply interpolation to recover missing pixels.

        Args:
            degraded: array with missing data (e.g., NaN or masked pixels).
            mask: binary mask where 1 indicates missing pixels to fill.
            meta: optional metadata (crs, transform, bands, etc.).
        Returns:
            reconstructed array with same shape as degraded.
        """

    def fit(self, *args: Any, **kwargs: Any) -> BaseMethod:  # optional
        return self

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim == 2:
            return mask_bool
        if mask_bool.ndim == 3:
            return np.any(mask_bool, axis=2)
        raise ValueError(f"Mask must be 2D or 3D, got shape {mask_bool.shape}")

    @staticmethod
    def _finalize(
        reconstructed: np.ndarray,
        *,
        clip_range: tuple[float, float] | None = (0.0, 1.0),
    ) -> np.ndarray:
        out = np.asarray(reconstructed, dtype=np.float32)
        if clip_range is not None:
            out = np.clip(out, clip_range[0], clip_range[1])
        out = np.nan_to_num(
            out,
            nan=0.0,
            posinf=clip_range[1] if clip_range else 0.0,
            neginf=clip_range[0] if clip_range else 0.0,
        )
        return out

    @staticmethod
    def _apply_channelwise(
        degraded: np.ndarray,
        mask_2d: np.ndarray,
        channel_fn,
    ) -> np.ndarray:
        """Apply a function to each channel of a multichannel image.

        Args:
            degraded: Input array (H, W) or (H, W, C).
            mask_2d: 2D boolean mask.
            channel_fn: Function signature fn(channel_2d, mask_2d) -> result_2d.

        Returns:
            Reconstructed array with same shape as degraded.
        """
        if degraded.ndim == 2:
            return channel_fn(degraded, mask_2d)

        # Multichannel case
        result = np.zeros_like(degraded)
        for c in range(degraded.shape[2]):
            result[:, :, c] = channel_fn(degraded[:, :, c], mask_2d)
        return result
