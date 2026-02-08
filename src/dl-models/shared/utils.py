"""Shared training utilities for DL inpainting models.

Provides GapPixelLoss, EarlyStopping, and checkpoint save/load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class GapPixelLoss(nn.Module):
    """Loss computed only on gap pixels, weighted by the mask.

    Supports MSE and L1 reduction modes.

    Args:
        mode: "mse" for mean squared error, "l1" for mean absolute error.
    """

    def __init__(self, mode: str = "mse") -> None:
        super().__init__()
        if mode not in ("mse", "l1"):
            msg = f"Unsupported loss mode: {mode!r}"
            raise ValueError(msg)
        self.mode = mode

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss on gap pixels only.

        Args:
            pred: (B, C, H, W) predicted image.
            target: (B, C, H, W) clean reference.
            mask: (B, H, W) binary mask (1=gap).

        Returns:
            Scalar loss value.
        """
        # Expand mask to match channel dimension
        mask_expanded = mask.unsqueeze(1).expand_as(pred)
        diff = pred - target

        pixel_loss = diff**2 if self.mode == "mse" else torch.abs(diff)

        # Mean over gap pixels only
        masked_loss = pixel_loss * mask_expanded
        n_gap = mask_expanded.sum().clamp(min=1.0)
        return masked_loss.sum() / n_gap


class EarlyStopping:
    """Patience-based early stopping tracker.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update with current validation loss.

        Returns:
            True if training should stop.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    **extra: Any,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Output file path.
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch number.
        loss: Current loss value.
        **extra: Additional items to include.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **extra,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Full checkpoint dict (includes epoch, loss, etc.).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state
