"""Figure 5: F1 score by threshold."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data_loader import NOISE_ORDER, noise_label
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


def fig5_f1_threshold(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart of F1 scores per model at each threshold."""
    thresholds = [
        ("f1_002", r"$\tau=0{,}02$"),
        ("f1_005", r"$\tau=0{,}05$"),
        ("f1_01", r"$\tau=0{,}10$"),
    ]
    available_thresholds = [
        (key, label) for key, label in thresholds if key in df.columns
    ]
    if not available_thresholds:
        log.warning("No F1 columns for fig5")
        return

    methods = sorted(df["method"].unique())
    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]
    palette = sns.color_palette("Set2", len(methods))

    for noise in noises:
        subset = df[df["noise_level"] == noise]
        if subset.empty:
            continue
        suffix = noise.replace("inf", "gap_only")

        means = {}
        for method in methods:
            mdf = subset[subset["method"] == method]
            means[method] = [
                float(mdf[k].mean()) if k in mdf.columns else 0.0
                for k, _ in available_thresholds
            ]

        n_t = len(available_thresholds)
        n_m = len(methods)
        x = np.arange(n_t)
        width = 0.75 / n_m
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

        fig, ax = plt.subplots(figsize=(6, 3.5))

        for i, method in enumerate(methods):
            bars = ax.bar(
                x + offsets[i],
                means[method],
                width=width,
                label=method,
                color=palette[i],
                edgecolor="#333333",
                linewidth=0.5,
            )
            for bar, val in zip(bars, means[method], strict=False):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_SIZE - 3,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in available_thresholds])
        style_axes(
            ax,
            title=f"F1 por Limiar — {noise_label(noise)}",
            xlabel="Limiar de erro",
            ylabel="F1 Score",
            grid_axis="y",
        )
        ax.set_ylim(0, 1.15)
        ax.legend(
            fontsize=FONT_SIZE - 2,
            loc="upper left",
            frameon=True,
            ncol=min(4, n_m),
        )
        plt.tight_layout()
        save_figure(fig, output_dir, f"fig5_f1_threshold_{suffix}")
        plt.close(fig)


def main() -> None:
    run_with_df(fig5_f1_threshold, "F1 score by threshold")


if __name__ == "__main__":
    main()
