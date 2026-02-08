"""Training script for the Transformer inpainting model.

Usage:
    uv run python src/dl-models/transformer/train.py \
        --manifest preprocessed/manifest.csv \
        --output src/dl-models/checkpoints/transformer_best.pt
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
from shared.utils import EarlyStopping, GapPixelLoss, save_checkpoint
from torch.utils.data import DataLoader
from transformer.model import _MAEInpaintingNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Transformer inpainting model."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/dl-models/checkpoints/transformer_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)
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

    model = _MAEInpaintingNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion = GapPixelLoss(mode="mse")
    early_stop = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, clean, mask in train_loader:
            x, clean, mask = x.to(device), clean.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, clean, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, clean, mask in val_loader:
                x, clean, mask = (
                    x.to(device),
                    clean.to(device),
                    mask.to(device),
                )
                pred = model(x)
                loss = criterion(pred, clean, mask)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_ds)

        lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            lr,
        )

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
