"""Classical robustness to additive noise."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..data_loader import (
    display_method_name,
    entropy_terciles,
    noise_label,
    select_top_n,
)
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)

_BIN_SUFFIX = {
    "baixa": "baixa",
    "média": "media",
    "alta": "alta",
}

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


def _plot_entropy_bin(
    plot_df: pd.DataFrame,
    methods: list[str],
    noise_order: list[str],
    entropy_bin: str,
    output_dir: Path,
    *,
    show_legend: bool,
    show_ylabel: bool,
) -> None:
    subset = plot_df[plot_df["entropy_bin"] == entropy_bin].copy()
    noise_positions = dict(
        zip(noise_order, range(len(noise_order)), strict=True)
    )
    method_markers = {
        method: _METHOD_MARKERS[index % len(_METHOD_MARKERS)]
        for index, method in enumerate(methods)
    }
    subset["noise_level"] = pd.Categorical(
        subset["noise_level"], categories=noise_order, ordered=True
    )
    subset = subset.sort_values("noise_level")
    if subset.empty:
        return

    fig, axis = plt.subplots(figsize=(4.0, 3.1))
    palette = sns.color_palette("Set2", len(methods))
    for index, method in enumerate(methods):
        method_df = subset[subset["method"] == method]
        if method_df.empty:
            continue
        axis.plot(
            method_df["noise_level"].map(noise_positions),
            method_df["psnr"],
            marker=method_markers[method],
            linewidth=1.4,
            markersize=4,
            color=palette[index],
            label=display_method_name(method),
        )

    axis.set_xticks(list(noise_positions.values()))
    axis.set_xticklabels([noise_label(noise) for noise in noise_order])
    style_axes(
        axis,
        xlabel="Ruído (dB)",
        ylabel="PSNR mediano (dB)" if show_ylabel else None,
        grid=True,
        grid_axis="both",
    )
    axis.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.5)
    if show_legend:
        axis.legend(loc="best", fontsize=FONT_SIZE - 2, frameon=True)

    fig.tight_layout()
    suffix = _BIN_SUFFIX.get(entropy_bin, entropy_bin)
    save_figure(fig, output_dir, f"fig_classical_noise_robustness_{suffix}")
    plt.close(fig)


def fig_classical_noise_robustness(df: pd.DataFrame, output_dir: Path) -> None:
    """Median PSNR by noise level, exported per entropy tercile."""
    classic = df[df["type"] == "Clássico"] if "type" in df.columns else df
    if classic.empty or "entropy_15" not in classic.columns:
        return

    methods = select_top_n(classic, n=3, noise_filter=None)
    if not methods:
        return

    binned = entropy_terciles(classic, entropy_col="entropy_15")
    binned = binned[binned["method"].isin(methods)]
    plot_df = (
        binned
        .groupby(["method", "entropy_bin", "noise_level"], observed=True)[
            "psnr"
        ]
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
    entropy_order = [
        bin_name
        for bin_name in ["baixa", "média", "alta"]
        if bin_name in set(plot_df["entropy_bin"])
    ]
    for index, entropy_bin in enumerate(entropy_order):
        _plot_entropy_bin(
            plot_df,
            methods,
            noise_order,
            entropy_bin,
            output_dir,
            show_legend=index == 0,
            show_ylabel=index == 0,
        )


def main() -> None:
    run_with_df(
        fig_classical_noise_robustness,
        "Classical noise robustness",
    )


if __name__ == "__main__":
    main()
