"""U-Net inpainting model in JAX/Flax — mirror of the PyTorch version.

Architecture (identical to dl_models/unet/model.py):
    Encoder: 4 blocks [64, 128, 256, 512] channels (stride-2 via max_pool)
    Bottleneck: 1024 channels with 2 residual convolutions
    Decoder: 4 blocks with skip connections [512, 256, 128, 64]
    Output: 1x1 conv -> sigmoid

Key differences from PyTorch version:
    - Channel layout: NHWC (JAX default) vs NCHW (PyTorch)
    - BatchNorm state: handled via Flax 'batch_stats' mutable collection
    - Params are separate from model definition (functional paradigm)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv -> BN -> GELU -> Conv -> BN -> GELU with residual projection.

    Mirrors _ConvBlock from the PyTorch version.
    """

    out_ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        residual = x

        h = nn.Conv(self.out_ch, (3, 3), padding="SAME", use_bias=False)(x)
        h = nn.BatchNorm(use_running_average=not training)(h)
        h = nn.gelu(h)
        h = nn.Conv(self.out_ch, (3, 3), padding="SAME", use_bias=False)(h)
        h = nn.BatchNorm(use_running_average=not training)(h)
        h = nn.gelu(h)

        # Residual projection when channel dims differ
        if residual.shape[-1] != self.out_ch:
            residual = nn.Conv(self.out_ch, (1, 1), use_bias=False)(residual)
            residual = nn.BatchNorm(use_running_average=not training)(residual)

        return h + residual


class UNetJAX(nn.Module):
    """U-Net with skip connections for gap inpainting.

    Input: (B, 64, 64, C+1) -> (B, 64, 64, C).
    Uses NHWC layout (JAX default).
    """

    out_channels: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # ── Encoder ──────────────────────────────────────────────
        e1 = ConvBlock(64)(x, training)
        d1 = nn.max_pool(e1, (2, 2), strides=(2, 2))  # 64 -> 32

        e2 = ConvBlock(128)(d1, training)
        d2 = nn.max_pool(e2, (2, 2), strides=(2, 2))  # 32 -> 16

        e3 = ConvBlock(256)(d2, training)
        d3 = nn.max_pool(e3, (2, 2), strides=(2, 2))  # 16 -> 8

        e4 = ConvBlock(512)(d3, training)
        d4 = nn.max_pool(e4, (2, 2), strides=(2, 2))  # 8 -> 4

        # ── Bottleneck ───────────────────────────────────────────
        b = ConvBlock(1024)(d4, training)

        # ── Decoder with skip connections ────────────────────────
        u4 = nn.ConvTranspose(512, (2, 2), strides=(2, 2))(b)
        u4 = jnp.concatenate([u4, e4], axis=-1)
        u4 = ConvBlock(512)(u4, training)

        u3 = nn.ConvTranspose(256, (2, 2), strides=(2, 2))(u4)
        u3 = jnp.concatenate([u3, e3], axis=-1)
        u3 = ConvBlock(256)(u3, training)

        u2 = nn.ConvTranspose(128, (2, 2), strides=(2, 2))(u3)
        u2 = jnp.concatenate([u2, e2], axis=-1)
        u2 = ConvBlock(128)(u2, training)

        u1 = nn.ConvTranspose(64, (2, 2), strides=(2, 2))(u2)
        u1 = jnp.concatenate([u1, e1], axis=-1)
        u1 = ConvBlock(64)(u1, training)

        # ── Output head ──────────────────────────────────────────
        out = nn.Conv(self.out_channels, (1, 1))(u1)
        return jax.nn.sigmoid(out)


# ---------------------------------------------------------------------------
# High-level inference wrapper (mirrors BaseDLMethod API)
# ---------------------------------------------------------------------------


class UNetInpaintingJAX:
    """U-Net gap-filling model (JAX/Flax).

    Provides the same ``apply(degraded, mask)`` interface as the
    PyTorch ``UNetInpainting`` so it can be used interchangeably.
    """

    name = "unet_jax_inpainting"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        in_channels: int = 5,
        out_channels: int = 4,
    ) -> None:
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._model = UNetJAX(out_channels=out_channels)
        self._variables: dict[str, Any] | None = None

        # Device selection (jax uses backends, not torch devices)
        if device and "cpu" in device:
            self._platform = "cpu"
        else:
            self._platform = jax.default_backend()

    # ── Initialization ────────────────────────────────────────────────

    def _ensure_variables(self) -> dict[str, Any]:
        """Initialize params from scratch or load from checkpoint."""
        if self._variables is not None:
            return self._variables

        # Init with dummy input
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, 64, 64, self._in_channels))
        self._variables = self._model.init(rng, dummy, training=False)

        # Load checkpoint if available
        if self._checkpoint_path is not None and self._checkpoint_path.exists():
            raw = self._checkpoint_path.read_bytes()
            self._variables = serialization.from_bytes(self._variables, raw)

        return self._variables

    # ── Inference ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        """Convert any mask format to 2D bool (True=gap)."""
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim == 2:
            return mask_bool
        if mask_bool.ndim == 3:
            return np.any(mask_bool, axis=2)
        msg = (
            f"Mask must be 2D or 3D, got ndim={mask_bool.ndim}, "
            f"shape={mask_bool.shape}"
        )
        raise ValueError(msg)

    @staticmethod
    def _finalize(reconstructed: np.ndarray) -> np.ndarray:
        """Clip to [0, 1], replace NaN/Inf, ensure float32."""
        out = np.asarray(reconstructed, dtype=np.float32)
        out = np.clip(out, 0.0, 1.0)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

    def _to_jax(self, degraded: np.ndarray, mask: np.ndarray) -> jnp.ndarray:
        """Convert numpy arrays to JAX model input (NHWC).

        Returns shape (1, H, W, C+1) where last channel is the mask.
        """
        if degraded.ndim == 2:
            degraded = degraded[:, :, np.newaxis]

        mask_2d = self._normalize_mask(mask).astype(np.float32)
        masked = degraded * (1.0 - mask_2d[:, :, np.newaxis])

        # Stack: (H, W, C) + (H, W, 1) -> (H, W, C+1)
        x = np.concatenate([masked, mask_2d[:, :, np.newaxis]], axis=-1)
        return jnp.array(x[np.newaxis, ...])  # (1, H, W, C+1)

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
        variables = self._ensure_variables()
        x = self._to_jax(degraded, mask)
        mask_bool = self._normalize_mask(mask)

        # Run forward pass (no BatchNorm update at inference)
        out = self._model.apply(variables, x, training=False)

        # (1, H, W, C) -> (H, W, C) numpy
        result = np.asarray(out[0])

        if degraded.ndim == 2:
            result = result.squeeze(-1)

        # Blend: keep valid pixels, replace gap pixels
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
