"""Classical spectral profile figure."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..data_loader import select_top_n
from ..tables.common import bootstrap_ci_half
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


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
        ("rmse_b0", "B0"),
        ("rmse_b1", "B1"),
        ("rmse_b2", "B2"),
        ("rmse_b3", "B3"),
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

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
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
                fmt="o",
                color=palette[band_index],
                ecolor=palette[band_index],
                capsize=2,
                markersize=5,
                linewidth=0.8,
            )

    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order)
    handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="")
        for color in palette
    ]
    ax.legend(handles, band_order, title="Banda", fontsize=FONT_SIZE - 2)
    style_axes(
        ax,
        xlabel=r"RMSE mediano (IC95\%)",
        ylabel="Método",
        grid_axis="x",
    )
    plt.tight_layout()
    save_figure(fig, output_dir, "fig_classical_spectral_profile")
    plt.close(fig)
