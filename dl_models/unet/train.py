"""Training script for the U-Net inpainting model.

Usage:
    uv run python -m dl_models.unet.train \
        --manifest preprocessed/manifest.csv \
        --output results/unet/checkpoints/unet_best.pth
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dl_models.shared.dataset import InpaintingDataset
from dl_models.shared.metrics import compute_validation_metrics
from dl_models.shared.trainer import (
    EarlyStopping,
    GapPixelLoss,
    TrainingHistory,
    save_checkpoint,
    setup_file_logging,
)
from dl_models.unet.model import _UNetNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train U-Net inpainting model."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/unet/checkpoints/unet_best.pth"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--satellite", type=str, default="sentinel2")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    if args.output.parent.name == "checkpoints":
        results_dir = args.output.parent.parent
    else:
        results_dir = args.output.parent
    checkpoints_dir = args.output.parent

    setup_file_logging(results_dir / "unet_train.log")

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    num_workers = args.num_workers or min(os.cpu_count() or 4, 8)
    pin = device.type == "cuda"

    train_ds = InpaintingDataset(
        args.manifest, split="train", satellite=args.satellite
    )
    val_ds = InpaintingDataset(
        args.manifest, split="val", satellite=args.satellite
    )

    if len(train_ds) == 0:
        log.error(
            "No training samples found in manifest '%s' "
            "(split='train', satellite='%s'). "
            "The manifest may only contain 'test' patches. "
            "Use the full preprocessing pipeline first.",
            args.manifest,
            args.satellite,
        )
        return

    _prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=_prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        prefetch_factor=_prefetch,
    )

    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    model = _UNetNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    criterion = GapPixelLoss(mode="mse")
    early_stop = EarlyStopping(patience=args.patience)

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")

    history = TrainingHistory(
        "unet",
        results_dir,
        metadata={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "satellite": args.satellite,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "device": str(device),
        },
    )

    global_step = 0
    n_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for step, (x, clean, mask) in enumerate(train_loader, 1):
            x = x.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x)
                loss = criterion(pred, clean, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            global_step += 1
            if step % 100 == 0 or step == n_batches:
                log.info(
                    "  [%d/%d] step %d/%d  loss=%.6f",
                    epoch,
                    args.epochs,
                    step,
                    n_batches,
                    loss.item(),
                )
            if global_step % 500 == 0:
                periodic_path = checkpoints_dir / f"unet_step_{global_step}.pth"
                save_checkpoint(
                    periodic_path,
                    model,
                    optimizer,
                    epoch,
                    train_loss / (step * args.batch_size),
                )
                log.info("  Periodic checkpoint saved: %s", periodic_path)
        train_loss /= len(train_ds)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_preds: list[torch.Tensor] = []
        val_targets: list[torch.Tensor] = []
        val_masks: list[torch.Tensor] = []
        with torch.no_grad():
            for x, clean, mask in val_loader:
                x = x.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    pred = model(x)
                    loss = criterion(pred, clean, mask)
                val_loss += loss.item() * x.size(0)
                val_preds.append(pred.cpu())
                val_targets.append(clean.cpu())
                val_masks.append(mask.cpu())
        val_loss /= len(val_ds)

        lr = optimizer.param_groups[0]["lr"]
        metrics = compute_validation_metrics(val_preds, val_targets, val_masks)

        log.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f"
            "  lr=%.2e  psnr=%.2f  ssim=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            lr,
            metrics["val_psnr"],
            metrics["val_ssim"],
        )

        history.record({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            **metrics,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output, model, optimizer, epoch, val_loss)
            log.info("Saved best checkpoint: %s", args.output)

        if early_stop.step(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
