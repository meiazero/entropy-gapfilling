"""U-Net inpainting model with skip connections.

Distinct from the GAN's internal UNet generator. Uses a deeper
4-level encoder-decoder with a 1024-channel bottleneck, residual
blocks within each stage, and GELU activations for improved gradient
flow.

Architecture:
    Encoder: 4 blocks [64, 128, 256, 512] channels (stride-2 conv)
    Bottleneck: 1024 channels with 2 residual convolutions
    Decoder: 4 blocks with skip connections [512, 256, 128, 64]
    Output: 1x1 conv -> sigmoid
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from dl_models.shared.base import BaseDLMethod


class _ConvBlock(nn.Module):
    """Conv -> BN -> GELU -> Conv -> BN -> GELU encoder/decoder block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        # Residual projection when channel dims differ
        self.proj: nn.Module
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.proj(x)


class _UNetNet(nn.Module):
    """U-Net with skip connections for gap inpainting.

    Input (C+1, 64, 64) -> (C, 64, 64).
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 4) -> None:
        super().__init__()

        # Encoder
        self.enc1 = _ConvBlock(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)  # 64 -> 32

        self.enc2 = _ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)  # 32 -> 16

        self.enc3 = _ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)  # 16 -> 8

        self.enc4 = _ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)  # 8 -> 4

        # Bottleneck
        self.bottleneck = _ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _ConvBlock(512 + 512, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _ConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _ConvBlock(64 + 64, 64)

        # Output
        self.head = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck
        b = self.bottleneck(self.down4(e4))

        # Decode with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


class UNetInpainting(BaseDLMethod):
    """U-Net gap-filling model with skip connections and residual blocks.

    Input: (H, W, C) image + (H, W) mask -> (H, W, C) reconstruction.
    Expects 64x64 patches with 4 spectral channels.
    """

    name = "unet_inpainting"

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
        return _UNetNet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            msg = "Model not initialized. Call _ensure_model() first."
            raise RuntimeError(msg)
        return self._model(x)
