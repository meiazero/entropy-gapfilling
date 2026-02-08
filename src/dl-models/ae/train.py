"""Training script for the Convolutional Autoencoder.

Usage:
    uv run python src/dl-models/ae/train.py \
        --manifest preprocessed/manifest.csv \
        --output src/dl-models/checkpoints/ae_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure dl-models root is importable
_DL_ROOT = Path(__file__).resolve().parent.parent
if str(_DL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DL_ROOT))

import torch
from ae.model import _AENet
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
    parser = argparse.ArgumentParser(description="Train AE inpainting model.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/dl-models/checkpoints/ae_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    # Datasets
    train_ds = InpaintingDataset(args.manifest, split="train")
    val_ds = InpaintingDataset(args.manifest, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    model = _AENet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = GapPixelLoss(mode="mse")
    early_stop = EarlyStopping(patience=args.patience)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
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

        # Validate
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

        log.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
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
