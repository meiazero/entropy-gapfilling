"""Training script for the Variational Autoencoder.

Usage:
    uv run python src/dl-models/vae/train.py \
        --manifest preprocessed/manifest.csv \
        --output src/dl-models/checkpoints/vae_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_DL_ROOT = Path(__file__).resolve().parent.parent
if str(_DL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DL_ROOT))

import torch
from shared.dataset import InpaintingDataset
from shared.utils import (
    EarlyStopping,
    GapPixelLoss,
    TrainingHistory,
    compute_validation_metrics,
    save_checkpoint,
    setup_file_logging,
)
from torch.utils.data import DataLoader
from vae.model import _VAENet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss = MSE_gap + beta * KL divergence.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    gap_loss = GapPixelLoss(mode="mse")
    recon = gap_loss(pred, target, mask)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kl
    return total, recon, kl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAE inpainting model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/dl-models/checkpoints/vae_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    setup_file_logging(args.output.parent / "vae_train.log")

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    train_ds = InpaintingDataset(args.manifest, split="train")
    val_ds = InpaintingDataset(args.manifest, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    model = _VAENet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(patience=args.patience)

    best_val_loss = float("inf")

    history = TrainingHistory(
        "vae",
        args.output.parent,
        metadata={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "beta": args.beta,
            "patience": args.patience,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "device": str(device),
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon_sum = 0.0
        train_kl_sum = 0.0
        for x, clean, mask in train_loader:
            x, clean, mask = x.to(device), clean.to(device), mask.to(device)
            optimizer.zero_grad()
            pred, mu, logvar = model(x)
            loss, recon, kl = vae_loss(
                pred, clean, mask, mu, logvar, beta=args.beta
            )
            loss.backward()
            optimizer.step()
            b = x.size(0)
            train_loss += loss.item() * b
            train_recon_sum += recon.item() * b
            train_kl_sum += kl.item() * b
        train_loss /= len(train_ds)
        train_recon_avg = train_recon_sum / len(train_ds)
        train_kl_avg = train_kl_sum / len(train_ds)

        model.eval()
        val_loss = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        val_masks: list[torch.Tensor] = []
        with torch.no_grad():
            for x, clean, mask in val_loader:
                x, clean, mask = (
                    x.to(device),
                    clean.to(device),
                    mask.to(device),
                )
                pred, mu, logvar = model(x)
                loss, recon, kl = vae_loss(
                    pred, clean, mask, mu, logvar, beta=args.beta
                )
                b = x.size(0)
                val_loss += loss.item() * b
                val_recon_sum += recon.item() * b
                val_kl_sum += kl.item() * b
                val_preds.append(pred.cpu())
                val_targets.append(clean.cpu())
                val_masks.append(mask.cpu())
        val_loss /= len(val_ds)
        val_recon_avg = val_recon_sum / len(val_ds)
        val_kl_avg = val_kl_sum / len(val_ds)

        metrics = compute_validation_metrics(val_preds, val_targets, val_masks)

        log.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  psnr=%.2f  ssim=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            metrics["val_psnr"],
            metrics["val_ssim"],
        )

        history.record({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recon_loss": train_recon_avg,
            "train_kl_loss": train_kl_avg,
            "val_loss": val_loss,
            "val_recon_loss": val_recon_avg,
            "val_kl_loss": val_kl_avg,
            **metrics,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output, model, optimizer, epoch, val_loss)
            log.info("Saved best checkpoint: %.6f", val_loss)

        if early_stop.step(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
