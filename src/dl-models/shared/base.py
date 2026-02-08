"""Base class for deep learning inpainting methods.

Extends BaseMethod with shared DL infrastructure: device selection,
checkpoint loading, numpy-to-tensor conversion, and blending.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pdi_pipeline.methods.base import BaseMethod


class BaseDLMethod(BaseMethod):
    """Base class for all deep learning gap-filling methods.

    Subclasses must implement ``_build_model()`` and ``_forward()``.
    The ``apply()`` method handles conversion between numpy arrays
    and tensors, model loading, inference, and blending.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)
        self._model: nn.Module | None = None

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Construct the model architecture. Called once on first use."""

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch tensor.

        Args:
            x: Input tensor of shape (1, C+1, H, W) on self._device.

        Returns:
            Reconstructed tensor of shape (1, C, H, W).
        """

    def _ensure_model(self) -> nn.Module:
        """Build and optionally load checkpoint weights."""
        if self._model is not None:
            return self._model
        self._model = self._build_model().to(self._device)
        if self._checkpoint_path is not None and self._checkpoint_path.exists():
            state = torch.load(
                self._checkpoint_path,
                map_location=self._device,
                weights_only=True,
            )
            if "model_state_dict" in state:
                self._model.load_state_dict(state["model_state_dict"])
            else:
                self._model.load_state_dict(state)
        self._model.eval()
        return self._model

    def _to_tensor(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
    ) -> torch.Tensor:
        """Convert numpy arrays to model input tensor.

        Args:
            degraded: (H, W, C) or (H, W) float32 array.
            mask: (H, W) float32 mask (1=gap).

        Returns:
            Tensor of shape (1, C+1, H, W) where:
            - First C channels = degraded * (1 - mask), gaps zeroed
            - Last channel = mask
        """
        if degraded.ndim == 2:
            degraded = degraded[:, :, np.newaxis]

        mask_2d = self._normalize_mask(mask).astype(np.float32)
        # Zero out gap pixels
        masked = degraded * (1.0 - mask_2d[:, :, np.newaxis])

        # (H, W, C) -> (C, H, W)
        img_t = torch.from_numpy(masked).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask_2d).unsqueeze(0).float()

        # Concatenate: (C+1, H, W)
        x = torch.cat([img_t, mask_t], dim=0)
        return x.unsqueeze(0).to(self._device)

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply DL inpainting to recover missing pixels.

        Args:
            degraded: (H, W, C) or (H, W) array with missing data.
            mask: (H, W) binary mask where 1=gap.
            meta: Optional metadata (unused).

        Returns:
            Reconstructed array with same shape as degraded.
        """
        self._ensure_model()
        x = self._to_tensor(degraded, mask)
        mask_bool = self._normalize_mask(mask)

        with torch.no_grad():
            out = self._forward(x)

        # (1, C, H, W) -> (H, W, C)
        result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # If input was 2D, squeeze back
        if degraded.ndim == 2:
            result = result.squeeze(-1)

        # Blend: keep original valid pixels, use model output for gaps
        if degraded.ndim == 2:
            blended = degraded.copy()
            blended[mask_bool] = result[mask_bool]
        else:
            blended = degraded.copy()
            mask_3d = np.broadcast_to(
                mask_bool[:, :, np.newaxis], degraded.shape
            )
            blended[mask_3d] = result[mask_3d]

        return self._finalize(blended)
