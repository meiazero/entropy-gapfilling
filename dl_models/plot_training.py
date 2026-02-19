"""Script shell for plotting DL training curves.

All plotting logic lives in dl_models.shared.visualization.
This script parses arguments and delegates to the plot functions.

Usage:
    uv run python -m dl_models.plot_training \
        --history dl_models/checkpoints/ae_history.json \
                  dl_models/checkpoints/vae_history.json \
        --output results/dl_plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_models.shared.trainer import TrainingHistory
from dl_models.shared.visualization import (
    plot_gan_balance,
    plot_loss_curves,
    plot_lr_schedule,
    plot_model_comparison,
    plot_pixel_accuracy_f1,
    plot_psnr_curves,
    plot_rmse_curves,
    plot_ssim_curves,
    plot_vae_decomposition,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training curves from history JSON files."
    )
    parser.add_argument(
        "--history",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to *_history.json file(s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dl_plots"),
        help="Directory for output plots.",
    )
    args = parser.parse_args()

    histories = []
    for p in args.history:
        h = TrainingHistory.load(p)
        histories.append({
            "model_name": h.model_name,
            "metadata": h.metadata,
            "epochs": h.epochs,
        })

    plot_loss_curves(histories, args.output)
    plot_psnr_curves(histories, args.output)
    plot_ssim_curves(histories, args.output)
    plot_rmse_curves(histories, args.output)
    plot_pixel_accuracy_f1(histories, args.output)
    plot_gan_balance(histories, args.output)
    plot_vae_decomposition(histories, args.output)
    plot_lr_schedule(histories, args.output)
    plot_model_comparison(histories, args.output)

    print(f"Plots saved to {args.output}")


if __name__ == "__main__":
    main()
