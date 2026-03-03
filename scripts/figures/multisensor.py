"""Figure 4: Multi-sensor violin plot (classic only)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from data_loader import NOISE_ORDER, noise_label, select_top_n
from figures.common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


def fig4_multisensor(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin plot of SSIM per satellite for top classic methods."""
    classic = df[df["type"] == "Clássico"]
    if classic.empty:
        log.warning("No classic data for fig4")
        return

    top = select_top_n(classic, n=3)
    if not top:
        return

    noises = [n for n in NOISE_ORDER if n in classic["noise_level"].unique()]

    for noise in noises:
        subset = classic[
            (classic["noise_level"] == noise) & (classic["method"].isin(top))
        ]
        if subset.empty:
            continue
        suffix = noise.replace("inf", "gap_only")

        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.violinplot(
            data=subset,
            x="satellite",
            y="ssim",
            hue="method",
            palette="Set2",
            ax=ax,
            inner="quartile",
            linewidth=0.8,
        )
        style_axes(
            ax,
            title=(
                f"Distribuição SSIM por Sensor — {noise_label(noise)} "
                f"(Top-3 Clássicos)"
            ),
            xlabel="Satélite",
            ylabel="SSIM",
            grid_axis="y",
        )
        ax.legend(title="Método", loc="best", fontsize=FONT_SIZE - 2)
        plt.tight_layout()
        save_figure(fig, output_dir, f"fig4_multisensor_{suffix}")
        plt.close(fig)
