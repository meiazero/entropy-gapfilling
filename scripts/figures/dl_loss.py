"""Figure 8: DL loss curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..data_loader import load_all_dl_histories
from .cli import run_no_df
from .common import FONT_SIZE, save_figure, style_axes


def fig8_dl_loss(output_dir: Path) -> None:
    """Train vs val loss per model, one figure per scenario x model."""
    histories = load_all_dl_histories()

    for scenario, models in histories.items():
        for model, hist in models.items():
            epochs_data = hist.get("epochs", [])
            if not epochs_data:
                continue

            epochs = [e["epoch"] for e in epochs_data]
            train_loss = [e.get("train_loss") for e in epochs_data]
            val_loss = [e.get("val_loss") for e in epochs_data]

            fig, ax = plt.subplots(figsize=(4, 2.8))

            every = max(1, len(epochs) // 8)
            ax.plot(
                epochs,
                train_loss,
                label="Treino",
                linewidth=1.2,
                color="#1f77b4",
                marker="o",
                markevery=every,
                markersize=3,
            )
            ax.plot(
                epochs,
                val_loss,
                label="Validação",
                linewidth=1.2,
                color="#ff7f0e",
                linestyle="--",
                marker="s",
                markevery=every,
                markersize=3,
            )
            style_axes(
                ax,
                title=f"{model.upper()} — {scenario.replace('_', ' ').title()}",
                xlabel="Época",
                ylabel="Perda",
            )
            ax.legend(fontsize=FONT_SIZE - 1, loc="best")
            plt.tight_layout()
            save_figure(fig, output_dir, f"fig8_dl_loss_{scenario}_{model}")
            plt.close(fig)


def main() -> None:
    run_no_df(fig8_dl_loss, "DL loss curves")


if __name__ == "__main__":
    main()
