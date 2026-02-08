"""Variational Autoencoder for image inpainting.

Same encoder architecture as the AE, but the bottleneck produces
mu/logvar for reparameterization. At inference time, uses mu directly
(no sampling) for deterministic output.
"""

from __future__ import annotations

from pathlib import Path

import torch
from shared.base import BaseDLMethod
from torch import nn


class _VAENet(nn.Module):
    """VAE network with reparameterization trick."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 4,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

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

        # Flatten (512, 4, 4) -> 8192 -> mu/logvar
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        # Decode from latent
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEInpainting(BaseDLMethod):
    """Variational autoencoder for gap-filling.

    At inference, uses mu directly (no sampling) for deterministic output.
    """

    name = "vae_inpainting"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        in_channels: int = 5,
        out_channels: int = 4,
        latent_dim: int = 256,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._latent_dim = latent_dim

    def _build_model(self) -> nn.Module:
        return _VAENet(
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            latent_dim=self._latent_dim,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            msg = "Model not initialized. Call _ensure_model() first."
            raise RuntimeError(msg)
        model: _VAENet = self._model  # type: ignore[assignment]
        mu, _logvar = model.encode(x)
        return model.decode(mu)
