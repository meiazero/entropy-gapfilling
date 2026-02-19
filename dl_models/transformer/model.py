"""MAE-style Transformer for image inpainting.

Adapted Masked Autoencoder architecture for gap-filling:
- PatchEmbed: Conv2d(5, 256, 8, stride=8) -> 64 tokens
- Encoder: 4 Transformer blocks (256-dim, 8 heads)
- Decoder: 2 Transformer blocks (128-dim, 4 heads) with mask tokens
- Unpatchify: Linear(128, 8*8*4) -> reshape to (4, 64, 64)
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


class _MAEInpaintingNet(nn.Module):
    """MAE-style network adapted for inpainting."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 4,
        patch_size: int = 8,
        img_size: int = 64,
        enc_dim: int = 256,
        enc_heads: int = 8,
        enc_depth: int = 4,
        dec_dim: int = 128,
        dec_heads: int = 4,
        dec_depth: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim

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

        self.enc_to_dec = nn.Linear(enc_dim, dec_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, dec_dim) * 0.02)

        self.dec_pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, dec_dim) * 0.02
        )

        self.decoder = nn.Sequential(*[
            _TransformerBlock(dec_dim, dec_heads, mlp_ratio=4.0)
            for _ in range(dec_depth)
        ])
        self.decoder_norm = nn.LayerNorm(dec_dim)

        self.head = nn.Linear(dec_dim, patch_size * patch_size * out_channels)

    def _patchify_mask(self, mask_channel: torch.Tensor) -> torch.Tensor:
        """Determine which patches are masked (>50% gap pixels).

        Args:
            mask_channel: (B, 1, H, W) mask channel.

        Returns:
            (B, n_patches) bool tensor, True = masked patch.
        """
        ps = self.patch_size
        mask_patches = nn.functional.avg_pool2d(
            mask_channel, kernel_size=ps, stride=ps
        )
        return mask_patches.flatten(1) > 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        mask_channel = x[:, -1:, :, :]
        patch_mask = self._patchify_mask(mask_channel)

        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        tokens = self.encoder(tokens)
        tokens = self.encoder_norm(tokens)

        tokens = self.enc_to_dec(tokens)

        mask_tokens = self.mask_token.expand(b, self.n_patches, -1)
        patch_mask_expanded = patch_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(patch_mask_expanded, mask_tokens, tokens)

        tokens = tokens + self.dec_pos_embed

        tokens = self.decoder(tokens)
        tokens = self.decoder_norm(tokens)

        pixels = self.head(tokens)
        ps = self.patch_size
        c = self.out_channels
        grid = self.img_size // ps

        pixels = pixels.view(b, grid, grid, ps, ps, c)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4)
        pixels = pixels.reshape(b, c, self.img_size, self.img_size)

        return torch.sigmoid(pixels)


class TransformerInpainting(BaseDLMethod):
    """MAE-style Transformer for gap-filling.

    Token masking: 8x8 patches with >50% gap pixels become mask tokens.
    """

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
        return _MAEInpaintingNet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            msg = "Model not initialized. Call _ensure_model() first."
            raise RuntimeError(msg)
        return self._model(x)
