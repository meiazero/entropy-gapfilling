"""Training script for the GAN inpainting model.

Usage:
    uv run python -m dl_models.gan.train \
        --manifest preprocessed/manifest.csv \
        --output results/gan/checkpoints/gan_best.pth
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dl_models.gan.model import _PatchDiscriminator, _UNetGenerator
from dl_models.shared.dataset import InpaintingDataset
from dl_models.shared.metrics import compute_validation_metrics
from dl_models.shared.trainer import (
    EarlyStopping,
    GapPixelLoss,
    TrainingHistory,
    save_checkpoint,
    setup_file_logging,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAN inpainting model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gan/checkpoints/gan_best.pth"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-l1", type=float, default=10.0)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--satellite", type=str, default="sentinel2")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    if args.output.parent.name == "checkpoints":
        results_dir = args.output.parent.parent
    else:
        results_dir = args.output.parent
    checkpoints_dir = args.output.parent

    setup_file_logging(results_dir / "gan_train.log")

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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    gen = _UNetGenerator().to(device)
    disc = _PatchDiscriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = torch.nn.BCELoss()
    gap_l1 = GapPixelLoss(mode="l1")
    early_stop = EarlyStopping(patience=args.patience)

    use_amp = device.type == "cuda"
    scaler_g = torch.amp.GradScaler(device.type, enabled=use_amp)
    scaler_d = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_loss = float("inf")

    history = TrainingHistory(
        "gan",
        results_dir,
        metadata={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lambda_l1": args.lambda_l1,
            "lambda_adv": args.lambda_adv,
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
        gen.train()
        disc.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0
        g_adv_sum = 0.0
        g_recon_sum = 0.0

        for step, (x, clean, mask) in enumerate(train_loader, 1):
            x, clean, mask = x.to(device), clean.to(device), mask.to(device)
            b = x.size(0)

            # --- Discriminator ---
            with torch.autocast(device_type=device.type, enabled=use_amp):
                fake = gen(x).detach()
                real_pred = disc(clean)
                fake_pred = disc(fake)
                real_label = torch.ones_like(real_pred)
                fake_label = torch.zeros_like(fake_pred)
                d_loss = (
                    bce(real_pred, real_label) + bce(fake_pred, fake_label)
                ) / 2
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler_d.step(opt_d)
            scaler_d.update()
            d_loss_sum += d_loss.item() * b

            # --- Generator ---
            with torch.autocast(device_type=device.type, enabled=use_amp):
                fake = gen(x)
                fake_pred = disc(fake)
                g_adv = bce(fake_pred, torch.ones_like(fake_pred))
                g_recon = gap_l1(fake, clean, mask)
                g_loss = args.lambda_l1 * g_recon + args.lambda_adv * g_adv
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
            g_loss_sum += g_loss.item() * b
            g_adv_sum += g_adv.item() * b
            g_recon_sum += g_recon.item() * b
            global_step += 1
            if step % 100 == 0 or step == n_batches:
                log.info(
                    "  [%d/%d] step %d/%d  g_loss=%.6f  d_loss=%.6f",
                    epoch,
                    args.epochs,
                    step,
                    n_batches,
                    g_loss.item(),
                    d_loss.item(),
                )
            if global_step % 500 == 0:
                periodic_path = checkpoints_dir / f"gan_step_{global_step}.pth"
                save_checkpoint(
                    periodic_path,
                    gen,
                    opt_g,
                    epoch,
                    g_loss_sum / (step * args.batch_size),
                    disc_state_dict=disc.state_dict(),
                    opt_d_state_dict=opt_d.state_dict(),
                )
                log.info("  Periodic checkpoint saved: %s", periodic_path)

        n = len(train_ds)
        g_loss_avg = g_loss_sum / n
        d_loss_avg = d_loss_sum / n
        g_adv_avg = g_adv_sum / n
        g_recon_avg = g_recon_sum / n

        gen.eval()
        val_loss = 0.0
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
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    fake = gen(x)
                    loss = gap_l1(fake, clean, mask)
                val_loss += loss.item() * x.size(0)
                val_preds.append(fake.cpu())
                val_targets.append(clean.cpu())
                val_masks.append(mask.cpu())
        val_loss /= len(val_ds)

        metrics = compute_validation_metrics(val_preds, val_targets, val_masks)

        log.info(
            "Epoch %d/%d  g_loss=%.6f  d_loss=%.6f  val_loss=%.6f"
            "  psnr=%.2f  ssim=%.4f",
            epoch,
            args.epochs,
            g_loss_avg,
            d_loss_avg,
            val_loss,
            metrics["val_psnr"],
            metrics["val_ssim"],
        )

        history.record({
            "epoch": epoch,
            "train_g_loss": g_loss_avg,
            "train_d_loss": d_loss_avg,
            "train_g_adv": g_adv_avg,
            "train_g_recon": g_recon_avg,
            "val_loss": val_loss,
            **metrics,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.output,
                gen,
                opt_g,
                epoch,
                val_loss,
                disc_state_dict=disc.state_dict(),
                opt_d_state_dict=opt_d.state_dict(),
            )
            log.info("Saved best checkpoint: %s", args.output)

        if early_stop.step(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
