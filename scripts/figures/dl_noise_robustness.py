"""DL robustness to additive noise."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from ..data_loader import display_method_name, noise_label
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

_METHOD_MARKERS: tuple[str, ...] = (
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
    "v",
    "<",
    ">",
    "h",
)


def fig_dl_noise_robustness(df: pd.DataFrame, output_dir: Path) -> None:
    """Median PSNR by noise level for the main DL models."""
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
    methods = ranked.index.tolist()
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

    fig, ax = plt.subplots(figsize=(4.1, 4.0))
    palette = sns.color_palette("Set2", len(methods))
    legend_handles: list[Line2D] = []
    for index, method in enumerate(methods):
        method_df = plot_df[plot_df["method"] == method]
        marker = _METHOD_MARKERS[index % len(_METHOD_MARKERS)]
        color = palette[index]
        ax.plot(
            method_df["noise_level"].map(noise_positions),
            method_df["psnr"],
            marker=marker,
            linewidth=1.6,
            markersize=4.8,
            color=color,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="",
                markersize=5.5,
                label=display_method_name(method),
            )
        )

    ax.set_xticks(list(noise_positions.values()))
    ax.set_xticklabels([noise_label(noise) for noise in noise_order])
    ax.margins(x=0.05, y=0.08)
    style_axes(
        ax,
        xlabel="Ruído (dB)",
        ylabel="PSNR mediano (dB)",
        grid_axis="y",
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fontsize=FONT_SIZE - 3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, output_dir, "fig_dl_noise_robustness")
    plt.close(fig)


def main() -> None:
    run_with_df(
        fig_dl_noise_robustness,
        "DL noise robustness",
    )


if __name__ == "__main__":
    main()
