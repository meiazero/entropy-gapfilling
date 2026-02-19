"""Training curve visualization functions.

All plot_* functions extracted from plot_training.py.
No sys.path hacks, no pdi_pipeline imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dl_models.shared.trainer import TrainingHistory

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # colorblind-safe
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save_fig(fig: Figure, output_dir: Path, name: str) -> None:
    """Save figure as both PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", backend="pdf")
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _get_epochs(hist: dict[str, Any]) -> list[int]:
    return [e["epoch"] for e in hist["epochs"]]


def _get_values(hist: dict[str, Any], key: str) -> list[float]:
    return [e.get(key, float("nan")) for e in hist["epochs"]]


def load_histories(paths: list[Path]) -> list[dict[str, Any]]:
    """Load TrainingHistory JSON files into plain dicts.

    Args:
        paths: List of paths to *_history.json files.

    Returns:
        List of dicts with 'model_name', 'metadata', 'epochs'.
    """
    histories: list[dict[str, Any]] = []
    for p in paths:
        h = TrainingHistory.load(p)
        histories.append({
            "model_name": h.model_name,
            "metadata": h.metadata,
            "epochs": h.epochs,
        })
    return histories


def plot_loss_curves(histories: list[dict[str, Any]], output_dir: Path) -> None:
    """Train vs val loss. Single model: 2 lines. Multiple: grid."""
    n = len(histories)
    if n == 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        h = histories[0]
        epochs = _get_epochs(h)
        name = h["model_name"]
        train_key = "train_g_loss" if name == "gan" else "train_loss"
        ax.plot(
            epochs, _get_values(h, train_key), label="Train", color=COLORS[0]
        )
        ax.plot(
            epochs, _get_values(h, "val_loss"), label="Val", color=COLORS[1]
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{name.upper()} - Loss Curves")
        ax.legend()
    else:
        rows = 2
        cols = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes_flat = np.asarray(axes).flatten()
        for i, h in enumerate(histories):
            ax = axes_flat[i]
            epochs = _get_epochs(h)
            name = h["model_name"]
            train_key = "train_g_loss" if name == "gan" else "train_loss"
            ax.plot(
                epochs,
                _get_values(h, train_key),
                label="Train",
                color=COLORS[0],
            )
            ax.plot(
                epochs, _get_values(h, "val_loss"), label="Val", color=COLORS[1]
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{name.upper()}")
            ax.legend(fontsize=8)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Loss Curves", fontsize=12, y=1.02)
        fig.tight_layout()

    _save_fig(fig, output_dir, "loss_curves")


def plot_psnr_curves(histories: list[dict[str, Any]], output_dir: Path) -> None:
    """Val PSNR over epochs per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(
            _get_epochs(h),
            _get_values(h, "val_psnr"),
            label=h["model_name"].upper(),
            color=COLORS[i % len(COLORS)],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.legend()
    _save_fig(fig, output_dir, "psnr_curves")


def plot_ssim_curves(histories: list[dict[str, Any]], output_dir: Path) -> None:
    """Val SSIM over epochs per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(
            _get_epochs(h),
            _get_values(h, "val_ssim"),
            label=h["model_name"].upper(),
            color=COLORS[i % len(COLORS)],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.set_title("Validation SSIM")
    ax.legend()
    _save_fig(fig, output_dir, "ssim_curves")


def plot_rmse_curves(histories: list[dict[str, Any]], output_dir: Path) -> None:
    """Val RMSE over epochs per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(
            _get_epochs(h),
            _get_values(h, "val_rmse"),
            label=h["model_name"].upper(),
            color=COLORS[i % len(COLORS)],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation RMSE")
    ax.legend()
    _save_fig(fig, output_dir, "rmse_curves")


def plot_pixel_accuracy_f1(
    histories: list[dict[str, Any]], output_dir: Path
) -> None:
    """Pixel accuracy + F1 at 3 thresholds, one subplot per model."""
    thresholds = ["002", "005", "010"]
    tau_labels = ["0.02", "0.05", "0.10"]
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, h in enumerate(histories):
        ax = axes[0, i]
        epochs = _get_epochs(h)
        for j, (tk, tl) in enumerate(zip(thresholds, tau_labels, strict=True)):
            pa = _get_values(h, f"val_pixel_acc_{tk}")
            f1 = _get_values(h, f"val_f1_{tk}")
            c = COLORS[j % len(COLORS)]
            ax.plot(epochs, pa, color=c, linestyle="-", label=f"PA t={tl}")
            ax.plot(epochs, f1, color=c, linestyle="--", label=f"F1 t={tl}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"{h['model_name'].upper()}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Pixel Accuracy & F1 at Multiple Thresholds", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "pixel_accuracy_f1")


def plot_gan_balance(histories: list[dict[str, Any]], output_dir: Path) -> None:
    """GAN-only: G loss vs D loss, g_adv vs g_recon decomposition."""
    gan_hists = [h for h in histories if h["model_name"] == "gan"]
    if not gan_hists:
        return

    h = gan_hists[0]
    epochs = _get_epochs(h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(
        epochs, _get_values(h, "train_g_loss"), label="G Loss", color=COLORS[0]
    )
    ax1.plot(
        epochs, _get_values(h, "train_d_loss"), label="D Loss", color=COLORS[1]
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Generator vs Discriminator")
    ax1.legend()

    ax2.plot(
        epochs,
        _get_values(h, "train_g_adv"),
        label="G Adversarial",
        color=COLORS[2],
    )
    ax2.plot(
        epochs,
        _get_values(h, "train_g_recon"),
        label="G Reconstruction",
        color=COLORS[3],
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Generator Loss Decomposition")
    ax2.legend()

    fig.suptitle("GAN Training Balance", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "gan_balance")


def plot_vae_decomposition(
    histories: list[dict[str, Any]], output_dir: Path
) -> None:
    """VAE-only: KL vs recon loss (train + val)."""
    vae_hists = [h for h in histories if h["model_name"] == "vae"]
    if not vae_hists:
        return

    h = vae_hists[0]
    epochs = _get_epochs(h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(
        epochs,
        _get_values(h, "train_recon_loss"),
        label="Train Recon",
        color=COLORS[0],
    )
    ax1.plot(
        epochs,
        _get_values(h, "val_recon_loss"),
        label="Val Recon",
        color=COLORS[1],
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.set_title("Reconstruction Loss")
    ax1.legend()

    ax2.plot(
        epochs,
        _get_values(h, "train_kl_loss"),
        label="Train KL",
        color=COLORS[2],
    )
    ax2.plot(
        epochs, _get_values(h, "val_kl_loss"), label="Val KL", color=COLORS[3]
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("KL Divergence")
    ax2.set_title("KL Divergence")
    ax2.legend()

    fig.suptitle("VAE Loss Decomposition", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "vae_decomposition")


def plot_lr_schedule(histories: list[dict[str, Any]], output_dir: Path) -> None:
    """Transformer-only: LR over epochs with val_loss dual y-axis."""
    tf_hists = [h for h in histories if h["model_name"] == "transformer"]
    if not tf_hists:
        return

    h = tf_hists[0]
    epochs = _get_epochs(h)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(
        epochs, _get_values(h, "lr"), label="Learning Rate", color=COLORS[0]
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Learning Rate", color=COLORS[0])
    ax1.tick_params(axis="y", labelcolor=COLORS[0])

    ax2.plot(
        epochs,
        _get_values(h, "val_loss"),
        label="Val Loss",
        color=COLORS[1],
        alpha=0.7,
    )
    ax2.set_ylabel("Val Loss", color=COLORS[1])
    ax2.tick_params(axis="y", labelcolor=COLORS[1])

    fig.suptitle("Transformer LR Schedule", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "lr_schedule")


def plot_model_comparison(
    histories: list[dict[str, Any]], output_dir: Path
) -> None:
    """Bar chart: best PSNR/SSIM/RMSE across all models."""
    if len(histories) < 2:
        return

    names = []
    best_psnr = []
    best_ssim = []
    best_rmse = []

    for h in histories:
        names.append(h["model_name"].upper())
        psnr_vals = _get_values(h, "val_psnr")
        ssim_vals = _get_values(h, "val_ssim")
        rmse_vals = _get_values(h, "val_rmse")
        best_psnr.append(max(psnr_vals) if psnr_vals else 0)
        best_ssim.append(max(ssim_vals) if ssim_vals else 0)
        best_rmse.append(min(rmse_vals) if rmse_vals else 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(names))
    w = 0.6

    axes[0].bar(x, best_psnr, w, color=COLORS[0])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Best PSNR")

    axes[1].bar(x, best_ssim, w, color=COLORS[1])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Best SSIM")

    axes[2].bar(x, best_rmse, w, color=COLORS[2])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("Best RMSE (lower is better)")

    fig.suptitle("Model Comparison", y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_dir, "model_comparison")
