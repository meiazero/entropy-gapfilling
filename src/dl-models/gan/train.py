"""Training script for the GAN inpainting model.

Usage:
    uv run python src/dl-models/gan/train.py \
        --manifest preprocessed/manifest.csv \
        --output src/dl-models/checkpoints/gan_best.pt
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
from gan.model import _PatchDiscriminator, _UNetGenerator
from shared.dataset import InpaintingDataset
from shared.utils import EarlyStopping, GapPixelLoss, save_checkpoint
from torch.utils.data import DataLoader

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
        default=Path("src/dl-models/checkpoints/gan_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-l1", type=float, default=10.0)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    args = parser.parse_args()

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

    gen = _UNetGenerator().to(device)
    disc = _PatchDiscriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    bce = torch.nn.BCELoss()
    gap_l1 = GapPixelLoss(mode="l1")
    early_stop = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        gen.train()
        disc.train()
        g_loss_sum = 0.0
        d_loss_sum = 0.0

        for x, clean, mask in train_loader:
            x, clean, mask = x.to(device), clean.to(device), mask.to(device)
            b = x.size(0)

            # --- Discriminator ---
            fake = gen(x).detach()
            real_pred = disc(clean)
            fake_pred = disc(fake)

            real_label = torch.ones_like(real_pred)
            fake_label = torch.zeros_like(fake_pred)

            d_loss = (
                bce(real_pred, real_label) + bce(fake_pred, fake_label)
            ) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            d_loss_sum += d_loss.item() * b

            # --- Generator ---
            fake = gen(x)
            fake_pred = disc(fake)

            g_adv = bce(fake_pred, torch.ones_like(fake_pred))
            g_recon = gap_l1(fake, clean, mask)
            g_loss = args.lambda_l1 * g_recon + args.lambda_adv * g_adv

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            g_loss_sum += g_loss.item() * b

        g_loss_avg = g_loss_sum / len(train_ds)
        d_loss_avg = d_loss_sum / len(train_ds)

        # Validate (generator only, L1 gap loss)
        gen.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, clean, mask in val_loader:
                x, clean, mask = (
                    x.to(device),
                    clean.to(device),
                    mask.to(device),
                )
                fake = gen(x)
                loss = gap_l1(fake, clean, mask)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        log.info(
            "Epoch %d/%d  g_loss=%.6f  d_loss=%.6f  val_loss=%.6f",
            epoch,
            args.epochs,
            g_loss_avg,
            d_loss_avg,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(args.output, gen, opt_g, epoch, val_loss)
            log.info("Saved best checkpoint: %.6f", val_loss)

        if early_stop.step(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
