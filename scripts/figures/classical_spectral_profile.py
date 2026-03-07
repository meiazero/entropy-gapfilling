"""Classical spectral profile figure."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.figures.cli import run_with_df

from ..data_loader import display_method_name, select_top_n
from ..tables.common import bootstrap_ci_half
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)

_BAND_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")


def _prepare_classical_subset(df: pd.DataFrame) -> pd.DataFrame:
    classic = df[df["type"] == "Clássico"] if "type" in df.columns else df
    if classic.empty:
        return classic
    available_noises = set(classic["noise_level"].unique())
    if "noise_level" in classic.columns and "inf" in available_noises:
        return classic[classic["noise_level"] == "inf"]
    return classic


def _build_plot_rows(
    classic: pd.DataFrame,
    top_methods: list[str],
    bands: list[tuple[str, str]],
) -> pd.DataFrame:
    plot_rows: list[dict[str, float | str]] = []
    for method in top_methods:
        method_df = classic[classic["method"] == method]
        cluster_ids = (
            method_df["patch_id"] if "patch_id" in method_df.columns else None
        )
        for band_col, band_label in bands:
            vals = method_df[band_col].dropna()
            if vals.empty:
                continue
            ci = bootstrap_ci_half(
                vals,
                cluster_ids,
                stat_fn=pd.Series.median,
            )
            plot_rows.append({
                "method": method,
                "band": band_label,
                "median": float(vals.median()),
                "ci": float(ci),
            })
    return pd.DataFrame(plot_rows)


def fig_classical_spectral_profile(df: pd.DataFrame, output_dir: Path) -> None:
    """Dot plot of per-band RMSE for the top classical methods."""
    bands = [
        ("rmse_b0", "B2 (Blue)"),
        ("rmse_b1", "B3 (Green)"),
        ("rmse_b2", "B4 (Red)"),
        ("rmse_b3", "B8 (NIR)"),
    ]
    if not all(col in df.columns for col, _ in bands):
        log.warning("Missing band columns for classical spectral profile")
        return

    classic = _prepare_classical_subset(df)
    if classic.empty:
        return

    top_methods = select_top_n(classic, n=3, noise_filter=None)
    if not top_methods:
        return

    plot_df = _build_plot_rows(classic, top_methods, bands)
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    palette = sns.color_palette("Set2", plot_df["band"].nunique())
    band_order = [label for _, label in bands]
    offsets = dict(
        zip(
            band_order,
            [-0.18, -0.06, 0.06, 0.18],
            strict=True,
        )
    )
    method_order = top_methods

    for band_index, band in enumerate(band_order):
        band_df = plot_df[plot_df["band"] == band]
        for method_index, method in enumerate(method_order):
            row = band_df[band_df["method"] == method]
            if row.empty:
                continue
            y = method_index + offsets[band]
            ax.errorbar(
                float(row["median"].iloc[0]),
                y,
                xerr=float(row["ci"].iloc[0]),
                fmt=_BAND_MARKERS[band_index % len(_BAND_MARKERS)],
                color=palette[band_index],
                ecolor=palette[band_index],
                capsize=2,
                markersize=5,
                linewidth=0.8,
            )

    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels([display_method_name(method) for method in method_order])
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=_BAND_MARKERS[index % len(_BAND_MARKERS)],
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="",
        )
        for index, color in enumerate(palette)
    ]
    style_axes(ax, xlabel=r"RMSE mediano (IC95%)", ylabel="Método")
    ax.legend(
        handles,
        band_order,
        title="Banda",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        fontsize=FONT_SIZE - 2,
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.28)
    save_figure(fig, output_dir, "fig_classical_spectral_profile")
    plt.close(fig)


def main() -> None:
    run_with_df(
        fig_classical_spectral_profile,
        "Classical spectral profile",
    )


if __name__ == "__main__":
    main()
