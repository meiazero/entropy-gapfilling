"""Figure 9: DL validation metrics per epoch."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..data_loader import load_all_dl_histories
from .cli import run_no_df
from .common import save_figure, style_axes


def fig9_dl_val_metrics(output_dir: Path) -> None:
    """Val PSNR/SSIM/RMSE per epoch.

    One figure per scenario x metric x model.
    """
    histories = load_all_dl_histories()
    metric_specs = [
        ("val_psnr", "PSNR (dB)"),
        ("val_ssim", "SSIM"),
        ("val_rmse", "RMSE"),
    ]

    for scenario, models in histories.items():
        for model, hist in models.items():
            epochs_data = hist.get("epochs", [])
            if not epochs_data:
                continue
            epochs = [e["epoch"] for e in epochs_data]

            for key, ylabel in metric_specs:
                values = [e.get(key) for e in epochs_data]
                if not any(v is not None for v in values):
                    continue

                fig, ax = plt.subplots(figsize=(4, 2.8))
                every = max(1, len(epochs) // 8)
                ax.plot(
                    epochs,
                    values,
                    linewidth=1.2,
                    color="#2ca02c",
                    marker="o",
                    markevery=every,
                    markersize=3,
                )
                style_axes(
                    ax,
                    title=(
                        f"{model.upper()} — {ylabel} "
                        f"({scenario.replace('_', ' ').title()})"
                    ),
                    xlabel="Época",
                    ylabel=ylabel,
                )
                plt.tight_layout()
                metric_name = key.replace("val_", "")
                save_figure(
                    fig,
                    output_dir,
                    f"fig9_dl_val_metrics_{scenario}_{metric_name}_{model}",
                )
                plt.close(fig)


def main() -> None:
    run_no_df(fig9_dl_val_metrics, "DL validation metrics")


if __name__ == "__main__":
    main()
