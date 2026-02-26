"""Shared training utilities for DL inpainting models.

Provides GapPixelLoss, EarlyStopping, checkpoint save/load,
TrainingHistory for metric persistence, and setup_file_logging.
No imports from pdi_pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_file_logging(log_path: str | Path) -> None:
    """Add a file handler to the root logger.

    Writes all messages (DEBUG and above) to the given file with
    timestamps, so the full processing history is preserved
    independently of terminal output.

    Args:
        log_path: Path to the log file. Parent dirs are created
            automatically.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    logging.getLogger().addHandler(handler)


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
        mask_expanded = mask.unsqueeze(1).expand_as(pred)
        diff = pred - target

        pixel_loss = diff**2 if self.mode == "mse" else torch.abs(diff)

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
        path: Output file path (.pth).
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
    state: dict[str, Any] = torch.load(
        path, map_location=device, weights_only=False
    )
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


class TrainingHistory:
    """Accumulates per-epoch metrics and persists them as JSON.

    Saves after every epoch (overwrite) so partial history survives
    crashes. Stores run metadata alongside epoch records.

    Args:
        model_name: Identifier for the model (e.g. "ae", "vae").
        output_dir: Directory where the JSON file is written.
        metadata: Optional dict of hyperparams, dataset sizes, etc.
            Recommended keys to include for aggregate_results.py:
            - n_params (int): total trainable parameter count.
            - training_time_s (float): cumulative wall-clock seconds.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.metadata: dict[str, Any] = metadata or {}
        self.epochs: list[dict[str, Any]] = []
        self._training_start: float = time.time()

    @property
    def path(self) -> Path:
        return self.output_dir / f"{self.model_name}_history.json"

    def record(self, epoch_data: dict[str, Any]) -> None:
        """Append an epoch record and save to disk.

        Automatically updates ``metadata["training_time_s"]`` with the
        elapsed wall-clock time since this TrainingHistory was created.
        """
        self.metadata["training_time_s"] = time.time() - self._training_start
        self.epochs.append(epoch_data)
        self.save()

    def save(self) -> None:
        """Write history to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "metadata": self.metadata,
            "epochs": self.epochs,
        }
        self.path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> TrainingHistory:
        """Load a previously saved training history.

        Args:
            path: Path to the JSON history file.

        Returns:
            Reconstructed TrainingHistory instance.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        history = cls(
            model_name=data["model_name"],
            output_dir=path.parent,
            metadata=data.get("metadata", {}),
        )
        history.epochs = data.get("epochs", [])
        return history
