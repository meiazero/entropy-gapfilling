"""Convolutional Autoencoder for image inpainting.

Architecture: 4-layer encoder-decoder with stride-2 convolutions,
batch normalization, and ReLU activations. The bottleneck compresses
to a 512-d vector before decoding back to (C, 64, 64).
"""

from __future__ import annotations

from pathlib import Path

import torch
from shared.base import BaseDLMethod
from torch import nn


class _AENet(nn.Module):
    """Encoder-decoder network for inpainting."""

    def __init__(self, in_channels: int = 5, out_channels: int = 4) -> None:
        super().__init__()
        # Encoder: (in_channels, 64, 64) -> (512, 4, 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Bottleneck: (512, 4, 4) -> (512, 1, 1)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4),
            nn.ReLU(inplace=True),
        )
        # Decoder: (512, 1, 1) -> (out_channels, 64, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.bottleneck(z)
        return self.decoder(z)


class AEInpainting(BaseDLMethod):
    """Convolutional autoencoder for gap-filling.

    Input: (H, W, C) image + (H, W) mask -> (H, W, C) reconstruction.
    Expects 64x64 patches with 4 spectral channels.
    """

    name = "ae_inpainting"

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
        return _AENet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            msg = "Model not initialized. Call _ensure_model() first."
            raise RuntimeError(msg)
        return self._model(x)
