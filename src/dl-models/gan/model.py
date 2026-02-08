"""GAN-based inpainting with UNet generator and PatchGAN discriminator.

The generator uses skip connections and dilated convolutions in the
bottleneck for a wider receptive field. At inference, only the
generator is used.
"""

from __future__ import annotations

from pathlib import Path

import torch
from shared.base import BaseDLMethod
from torch import nn


class _UNetGenerator(nn.Module):
    """UNet-like generator with skip connections and dilated bottleneck."""

    def __init__(self, in_channels: int = 5, out_channels: int = 4) -> None:
        super().__init__()
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Dilated bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks (with skip connections)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([b, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


class _PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (4 Conv2d blocks -> 1-channel output)."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GANInpainting(BaseDLMethod):
    """GAN-based gap-filling using UNet generator.

    At inference, only the generator is used. The discriminator is
    only needed during training.
    """

    name = "gan_inpainting"

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
        return _UNetGenerator(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            msg = "Model not initialized. Call _ensure_model() first."
            raise RuntimeError(msg)
        return self._model(x)
