"""Shared training utilities for DL inpainting models.

Provides GapPixelLoss, EarlyStopping, checkpoint save/load,
TrainingHistory for metric persistence, and compute_validation_metrics
for bridging torch tensors to pdi_pipeline quality metrics.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# Add src/ to path for pdi_pipeline imports
_SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from pdi_pipeline.metrics import psnr, rmse, ssim

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


class TrainingHistory:
    """Accumulates per-epoch metrics and persists them as JSON.

    Saves after every epoch (overwrite) so partial history survives
    crashes. Stores run metadata alongside epoch records.

    Args:
        model_name: Identifier for the model (e.g. "ae", "vae").
        output_dir: Directory where the JSON file is written.
        metadata: Optional dict of hyperparams, dataset sizes, etc.
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

    @property
    def path(self) -> Path:
        return self.output_dir / f"{self.model_name}_history.json"

    def record(self, epoch_data: dict[str, Any]) -> None:
        """Append an epoch record and save to disk."""
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


def compute_validation_metrics(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    max_ssim_samples: int = 64,
) -> dict[str, float]:
    """Compute quality metrics bridging torch tensors to pdi_pipeline.

    Converts (B, C, H, W) tensors to numpy (H, W, C) and calls
    psnr/ssim/rmse per sample. Also computes pixel accuracy and F1
    at multiple thresholds.

    Args:
        preds: List of prediction tensors, each (B, C, H, W).
        targets: List of clean reference tensors, each (B, C, H, W).
        masks: List of mask tensors, each (B, H, W) with 1=gap.
        max_ssim_samples: Cap for SSIM computation (performance).

    Returns:
        Dict with val_psnr, val_ssim, val_rmse, and pixel accuracy /
        F1 at thresholds 0.02, 0.05, 0.10.
    """
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    rmse_vals: list[float] = []
    # Per-threshold accumulators
    thresholds = [0.02, 0.05, 0.10]
    acc_counts = dict.fromkeys(thresholds, 0.0)
    acc_totals = dict.fromkeys(thresholds, 0.0)

    ssim_count = 0

    for pred_batch, target_batch, mask_batch in zip(
        preds, targets, masks, strict=False
    ):
        b = pred_batch.shape[0]
        for i in range(b):
            # (C, H, W) -> (H, W, C)
            p = pred_batch[i].permute(1, 2, 0).numpy()
            t = target_batch[i].permute(1, 2, 0).numpy()
            m = mask_batch[i].numpy()

            # Squeeze single-channel to (H, W)
            if p.shape[2] == 1:
                p = p[:, :, 0]
                t = t[:, :, 0]

            psnr_vals.append(psnr(t, p, m))
            rmse_vals.append(rmse(t, p, m))

            if ssim_count < max_ssim_samples:
                ssim_vals.append(ssim(t, p, m))
                ssim_count += 1

            # Pixel accuracy at thresholds
            gap = m > 0.5
            if np.any(gap):
                if p.ndim == 3:
                    gap_3d = np.broadcast_to(gap[:, :, np.newaxis], p.shape)
                    diff = np.abs(p[gap_3d] - t[gap_3d])
                else:
                    diff = np.abs(p[gap] - t[gap])
                n_gap = float(diff.size)
                for tau in thresholds:
                    correct = float(np.sum(diff < tau))
                    acc_counts[tau] += correct
                    acc_totals[tau] += n_gap

    result: dict[str, float] = {}

    if psnr_vals:
        # Filter out inf values for mean
        finite_psnr = [v for v in psnr_vals if np.isfinite(v)]
        result["val_psnr"] = float(np.mean(finite_psnr)) if finite_psnr else 0.0
    else:
        result["val_psnr"] = 0.0

    result["val_ssim"] = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    result["val_rmse"] = float(np.mean(rmse_vals)) if rmse_vals else 0.0

    for tau in thresholds:
        tau_key = str(tau).replace(".", "")
        pa = acc_counts[tau] / acc_totals[tau] if acc_totals[tau] > 0 else 0.0
        f1 = 2 * pa / (1 + pa) if (1 + pa) > 0 else 0.0
        result[f"val_pixel_acc_{tau_key}"] = pa
        result[f"val_f1_{tau_key}"] = f1

    return result
