"""DL robustness to additive noise."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..data_loader import load_combined
from .common import FONT_SIZE, save_figure, style_axes


def fig_dl_noise_robustness(output_dir: Path) -> None:
    """Median PSNR by noise level for the main DL models."""
    df = load_combined()
    if df.empty or "type" not in df.columns:
        return

    dl = df[df["type"] == "DL"].copy()
    if dl.empty:
        return
    if "entropy_scenario" in dl.columns and "entropy_all" in set(
        dl["entropy_scenario"]
    ):
        dl = dl[dl["entropy_scenario"] == "entropy_all"]

    ranked = (
        dl
        .groupby("method", observed=True)["psnr"]
        .median()
        .sort_values(ascending=False)
    )
    methods = ranked.head(4).index.tolist()
    plot_df = (
        dl[dl["method"].isin(methods)]
        .groupby(["method", "noise_level"], observed=True)["psnr"]
        .median()
        .reset_index()
    )
    if plot_df.empty:
        return

    noise_order = [
        noise
        for noise in ["inf", "40", "30", "20"]
        if noise in set(plot_df["noise_level"])
    ]
    plot_df["noise_level"] = pd.Categorical(
        plot_df["noise_level"],
        categories=noise_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["method", "noise_level"])
    noise_positions = dict(
        zip(noise_order, range(len(noise_order)), strict=True)
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    palette = sns.color_palette("Set2", len(methods))
    for index, method in enumerate(methods):
        method_df = plot_df[plot_df["method"] == method]
        ax.plot(
            method_df["noise_level"].map(noise_positions),
            method_df["psnr"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            color=palette[index],
            label=method.upper(),
        )

    ax.set_xticks(list(noise_positions.values()))
    ax.set_xticklabels(noise_order)
    style_axes(
        ax,
        xlabel="Ruído (dB)",
        ylabel="PSNR mediano (dB)",
        grid_axis="y",
    )
    ax.legend(loc="best", fontsize=FONT_SIZE - 2, frameon=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_dl_noise_robustness")
    plt.close(fig)
