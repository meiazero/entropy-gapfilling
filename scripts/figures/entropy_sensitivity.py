"""Figure 3: Entropy sensitivity plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_loader import ENTROPY_WINDOWS, NOISE_ORDER, noise_label, select_top_n
from figures.cli import run_with_df
from figures.common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


def fig3_sensitivity(df: pd.DataFrame, output_dir: Path) -> None:
    """Regression plots of SAM/ERGAS vs entropy for top methods."""
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    selected = top_classic + top_dl

    if not selected:
        log.warning("No methods for fig3")
        return

    metrics_to_plot = [m for m in ["sam", "ergas"] if m in df.columns]
    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]
    palette = sns.color_palette("Set2", len(selected))

    for metric in metrics_to_plot:
        for ws in ENTROPY_WINDOWS:
            ecol = f"entropy_{ws}"
            if ecol not in df.columns:
                continue
            for noise in noises:
                subset = df[df["noise_level"] == noise]
                if subset.empty:
                    continue
                suffix = noise.replace("inf", "gap_only")

                fig, ax = plt.subplots(figsize=(5, 3.5))

                for idx, method in enumerate(selected):
                    mdf = subset[subset["method"] == method][
                        [ecol, metric]
                    ].dropna()
                    if len(mdf) < 10:
                        continue
                    sns.regplot(
                        data=mdf,
                        x=ecol,
                        y=metric,
                        ax=ax,
                        color=palette[idx],
                        scatter_kws={"s": 8, "alpha": 0.4, "rasterized": True},
                        line_kws={"linewidth": 1.5},
                        label=method,
                        ci=95,
                    )

                style_axes(
                    ax,
                    title=(
                        f"Sensibilidade à Entropia — {metric.upper()} "
                        f"({noise_label(noise)})"
                    ),
                    xlabel=f"Entropia ({ws}x{ws})",
                    ylabel=metric.upper(),
                    grid_alpha=0.2,
                )
                ax.legend(loc="best", fontsize=FONT_SIZE - 2, frameon=True)
                plt.tight_layout()
                save_figure(
                    fig,
                    output_dir,
                    f"fig3_sensitivity_{metric}_{suffix}_e{ws}",
                )
                plt.close(fig)


def main() -> None:
    run_with_df(fig3_sensitivity, "Entropy sensitivity")


if __name__ == "__main__":
    main()
