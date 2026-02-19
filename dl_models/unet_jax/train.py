"""Training script for the U-Net inpainting model (JAX/Flax).

Usage:
    uv run python -m dl_models.unet_jax.train \
        --manifest preprocessed/manifest.csv \
        --output results/unet_jax/checkpoints/unet_jax_best.msgpack

Mirrors dl_models/unet/train.py but uses JAX, Flax, and Optax.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state

from dl_models.shared.dataset import InpaintingDataset
from dl_models.shared.trainer import (
    EarlyStopping,
    TrainingHistory,
    setup_file_logging,
)
from dl_models.unet_jax.model import UNetJAX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Train State (Flax convention) — includes batch_stats for BatchNorm
# ---------------------------------------------------------------------------


class TrainStateWithBN(train_state.TrainState):
    """TrainState extended with BatchNorm running statistics."""

    batch_stats: dict  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# Loss function (pure JAX — mirrors GapPixelLoss)
# ---------------------------------------------------------------------------


def gap_pixel_loss(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """MSE loss computed only on gap pixels.

    Args:
        pred: (B, H, W, C) predicted image (NHWC).
        target: (B, H, W, C) clean reference (NHWC).
        mask: (B, H, W) binary mask, 1=gap.
    """
    mask_expanded = jnp.expand_dims(mask, axis=-1)  # (B, H, W, 1)
    diff = pred - target
    pixel_loss = diff**2
    masked_loss = pixel_loss * mask_expanded
    n_gap = jnp.maximum(masked_loss.size, 1)
    n_gap = jnp.maximum(jnp.sum(mask_expanded), 1.0)
    return jnp.sum(masked_loss) / n_gap


# ---------------------------------------------------------------------------
# JIT-compiled train and eval steps
# ---------------------------------------------------------------------------


@jax.jit
def train_step(
    state: TrainStateWithBN,
    batch_x: jnp.ndarray,
    batch_clean: jnp.ndarray,
    batch_mask: jnp.ndarray,
) -> tuple[TrainStateWithBN, jnp.ndarray]:
    """Single training step with gradient computation."""

    def loss_fn(params: dict) -> tuple[jnp.ndarray, dict]:
        variables = {"params": params, "batch_stats": state.batch_stats}
        pred, updates = state.apply_fn(
            variables,
            batch_x,
            training=True,
            mutable=["batch_stats"],
        )
        loss = gap_pixel_loss(pred, batch_clean, batch_mask)
        return loss, updates

    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )

    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss


@jax.jit
def eval_step(
    state: TrainStateWithBN,
    batch_x: jnp.ndarray,
    batch_clean: jnp.ndarray,
    batch_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Single evaluation step (no gradient, no BN update)."""
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    pred = state.apply_fn(variables, batch_x, training=False)
    return gap_pixel_loss(pred, batch_clean, batch_mask)


# ---------------------------------------------------------------------------
# Utility: PyTorch DataLoader -> JAX batches (NCHW -> NHWC)
# ---------------------------------------------------------------------------


def pytorch_to_jax(
    x: jnp.ndarray,
    clean: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert PyTorch tensors (NCHW) to JAX arrays (NHWC).

    Args:
        x: (B, C+1, H, W) input tensor.
        clean: (B, C, H, W) target tensor.
        mask: (B, H, W) mask tensor.

    Returns:
        Tuple of JAX arrays in NHWC format.
    """
    x_np = x.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
    clean_np = clean.numpy().transpose(0, 2, 3, 1)
    mask_np = mask.numpy()

    return jnp.array(x_np), jnp.array(clean_np), jnp.array(mask_np)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:

    import torch  # noqa: F401
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        description="Train U-Net inpainting model (JAX/Flax)."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/unet_jax/checkpoints/unet_jax_best.msgpack"),
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--satellite", type=str, default="sentinel2")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    if args.output.parent.name == "checkpoints":
        results_dir = args.output.parent.parent
    else:
        results_dir = args.output.parent
    checkpoints_dir = args.output.parent

    setup_file_logging(results_dir / "unet_jax_train.log")

    backend = jax.default_backend()
    log.info("JAX backend: %s", backend)
    log.info("JAX devices: %s", jax.devices())

    # ── Data loaders (reusing PyTorch Dataset) ────────────────────
    # JAX is multithreaded; os.fork() from DataLoader workers causes
    # deadlocks, so we force num_workers=0.
    num_workers = 0

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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    log.info("Train: %d, Val: %d", len(train_ds), len(val_ds))

    # ── Model init ───────────────────────────────────────────────
    model = UNetJAX(out_channels=4)
    rng = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, 64, 64, 5))  # NHWC
    variables = model.init(rng, dummy, training=True)

    # Optax optimizer: AdamW + cosine decay
    total_steps = args.epochs * len(train_loader)
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=total_steps,
        alpha=1e-6 / args.lr,
    )
    optimizer = optax.adamw(
        learning_rate=schedule, weight_decay=args.weight_decay
    )

    state = TrainStateWithBN.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        batch_stats=variables["batch_stats"],
    )

    early_stop = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")

    history = TrainingHistory(
        "unet_jax",
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
            "backend": backend,
            "framework": "jax/flax",
        },
    )

    # ── Training loop ────────────────────────────────────────────
    global_step = 0
    n_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = 0.0
        n_train = 0
        for step, (x, clean, mask) in enumerate(train_loader, 1):
            x_jax, clean_jax, mask_jax = pytorch_to_jax(x, clean, mask)
            state, loss = train_step(state, x_jax, clean_jax, mask_jax)
            bs = x_jax.shape[0]
            train_loss += float(loss) * bs
            n_train += bs
            global_step += 1
            if step % 100 == 0 or step == n_batches:
                log.info(
                    "  [%d/%d] step %d/%d  loss=%.6f",
                    epoch,
                    args.epochs,
                    step,
                    n_batches,
                    float(loss),
                )
            if global_step % 500 == 0:
                periodic_path = (
                    checkpoints_dir / f"unet_jax_step_{global_step}.msgpack"
                )
                periodic_path.parent.mkdir(parents=True, exist_ok=True)
                raw = serialization.to_bytes({
                    "params": state.params,
                    "batch_stats": state.batch_stats,
                })
                periodic_path.write_bytes(raw)
                log.info("  Periodic checkpoint saved: %s", periodic_path)
        train_loss /= max(n_train, 1)

        # Validate
        val_loss = 0.0
        n_val = 0
        for x, clean, mask in val_loader:
            x_jax, clean_jax, mask_jax = pytorch_to_jax(x, clean, mask)
            loss = eval_step(state, x_jax, clean_jax, mask_jax)
            bs = x_jax.shape[0]
            val_loss += float(loss) * bs
            n_val += bs
        val_loss /= max(n_val, 1)

        # Current learning rate
        lr = float(schedule(state.step))

        log.info(
            "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            lr,
        )

        history.record({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            raw = serialization.to_bytes({
                "params": state.params,
                "batch_stats": state.batch_stats,
            })
            args.output.write_bytes(raw)
            log.info("Saved best checkpoint: %s", args.output)

        if early_stop.step(val_loss):
            log.info("Early stopping at epoch %d", epoch)
            break

    log.info("Training complete. Best val loss: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
