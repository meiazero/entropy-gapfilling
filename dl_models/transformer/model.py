"""Vision Transformer (ViT) for image inpainting.

ViT-style encoder with patch embeddings and a reconstruction head:
- PatchEmbed: Conv2d(5, 256, 8, stride=8) -> 64 tokens
- Encoder: 4 Transformer blocks (256-dim, 8 heads)
- Head: Linear(256, 8*8*4) -> reshape to (4, 64, 64)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dl_models.shared.base import BaseDLMethod


class _TransformerBlock(nn.Module):
    """Standard Transformer block with multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class _ViTInpaintingNet(nn.Module):
    """ViT-style network adapted for inpainting."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 4,
        patch_size: int = 8,
        img_size: int = 64,
        enc_dim: int = 256,
        enc_heads: int = 8,
        enc_depth: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        self.enc_dim = enc_dim

        self.patch_embed = nn.Conv2d(
            in_channels, enc_dim, patch_size, stride=patch_size
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, enc_dim) * 0.02
        )

        self.encoder = nn.Sequential(*[
            _TransformerBlock(enc_dim, enc_heads, mlp_ratio=4.0)
            for _ in range(enc_depth)
        ])
        self.encoder_norm = nn.LayerNorm(enc_dim)

        self.head = nn.Linear(enc_dim, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)

        pixels = self.head(tokens)
        ps = self.patch_size
        c = self.out_channels
        grid = self.img_size // ps

        pixels = pixels.view(b, grid, grid, ps, ps, c)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4)
        pixels = pixels.reshape(b, c, self.img_size, self.img_size)

        return torch.sigmoid(pixels)


class TransformerInpainting(BaseDLMethod):
    """ViT-style Transformer for gap-filling."""

    name = "transformer_inpainting"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        in_channels: int = 5,
        out_channels: int = 4,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._in_channels = in_channels
        self._out_channels = out_channels

    def _build_model(self) -> nn.Module:
        return _ViTInpaintingNet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            msg = "Model not initialized. Call _ensure_model() first."
            raise RuntimeError(msg)
        return self._model(x)
