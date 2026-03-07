"""Classical Pareto front (time x quality)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..data_loader import _normalize_method_key, display_method_name
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)

_METHOD_MARKERS: dict[str, str] = {
    "nearest_neighbor": "o",
    "bilinear": "s",
    "bicubic": "^",
    "lanczos": "D",
    "idw": "P",
    "rbf": "X",
    "thin_plate_spline": "v",
    "ordinary_kriging": "<",
    "dct_ista": ">",
    "wavelet_ista": "h",
    "total_variation": "H",
    "l1_dct": "p",
    "l1_wavelet": "*",
    "non_local_means": "d",
    "exemplar_based": "8",
}

_METHOD_COLORS: dict[str, str] = {
    "nearest_neighbor": "#4e79a7",
    "bilinear": "#f28e2b",
    "bicubic": "#e15759",
    "lanczos": "#76b7b2",
    "idw": "#59a14f",
    "rbf": "#edc948",
    "thin_plate_spline": "#b07aa1",
    "ordinary_kriging": "#ff9da7",
    "dct_ista": "#9c755f",
    "wavelet_ista": "#bab0ab",
    "total_variation": "#1f77b4",
    "l1_dct": "#d62728",
    "l1_wavelet": "#9467bd",
    "non_local_means": "#8c564b",
    "exemplar_based": "#2ca02c",
}

_LEGEND_LABELS: dict[str, str] = {
    "nearest_neighbor": "Nearest",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "lanczos": "Lanczos",
    "idw": "IDW",
    "rbf": "RBF",
    "thin_plate_spline": "Thin Plate",
    "ordinary_kriging": "Ord. Kriging",
    "dct_ista": "DCT-ISTA",
    "wavelet_ista": "Wavelet-ISTA",
    "total_variation": "Total Var.",
    "l1_dct": "L1-DCT",
    "l1_wavelet": "L1-Wavelet",
    "non_local_means": "Non-Local",
    "exemplar_based": "Exemplar",
}


def _iqr_bounds(values: pd.Series) -> tuple[float, float, float]:
    vals = values.dropna().to_numpy()
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    q1 = float(np.percentile(vals, 25))
    q2 = float(np.percentile(vals, 50))
    q3 = float(np.percentile(vals, 75))
    return q1, q2, q3


def _canonical_method_key(method_name: str) -> str:
    normalized = _normalize_method_key(display_method_name(method_name))
    normalized = re.sub(r"\(.*?\)", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")

    aliases = {
        "nearest": "nearest_neighbor",
        "thin_plate": "thin_plate_spline",
        "ord_kriging": "ordinary_kriging",
        "ordinary_kriging": "ordinary_kriging",
        "non_local": "non_local_means",
        "exemplar": "exemplar_based",
        "l1_wavelet": "l1_wavelet",
        "l1_dct": "l1_dct",
        "dct_ista": "dct_ista",
        "wavelet_ista": "wavelet_ista",
        "lanczos": "lanczos",
    }
    return aliases.get(normalized, normalized)


def _pareto_stats(subset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, mtype), grp in subset.groupby(
        ["method", "type"], observed=True
    ):
        if "elapsed_s" not in grp.columns or grp["elapsed_s"].isna().all():
            continue
        q1, med, q3 = _iqr_bounds(grp["elapsed_s"])
        psnr_med = float(grp["psnr"].median())
        rows.append({
            "method": method,
            "type": mtype,
            "time_med": med,
            "time_q1": q1,
            "time_q3": q3,
            "psnr_med": psnr_med,
        })
    return pd.DataFrame(rows)


def _marker_for_method(method_name: str) -> str:
    method_key = _canonical_method_key(method_name)
    return _METHOD_MARKERS.get(method_key, "o")


def _color_for_method(method_name: str) -> str:
    method_key = _canonical_method_key(method_name)
    return _METHOD_COLORS.get(method_key, "#1f77b4")


def _legend_label(method_name: str) -> str:
    method_key = _canonical_method_key(method_name)
    return _LEGEND_LABELS.get(method_key, display_method_name(method_name))


def _legend_handles(stats_df: pd.DataFrame) -> list[Line2D]:
    ordered_methods = (
        stats_df
        .sort_values(["time_med", "psnr_med"], ascending=[True, False])[
            "method"
        ]
        .drop_duplicates()
        .tolist()
    )
    return [
        Line2D(
            [0],
            [0],
            marker=_marker_for_method(method_name),
            color=_color_for_method(str(method_name)),
            markerfacecolor=_color_for_method(str(method_name)),
            markeredgecolor=_color_for_method(str(method_name)),
            linestyle="",
            markersize=5.5,
            label=_legend_label(str(method_name)),
        )
        for method_name in ordered_methods
    ]


def _expand_time_axis(ax: plt.Axes, stats_df: pd.DataFrame) -> None:
    valid_q1 = stats_df.loc[stats_df["time_q1"] > 0, "time_q1"]
    valid_q3 = stats_df.loc[stats_df["time_q3"] > 0, "time_q3"]
    if valid_q1.empty or valid_q3.empty:
        return
    ax.set_xlim(float(valid_q1.min()) / 1.8, float(valid_q3.max()) * 2.4)


def _plot_pareto(
    stats_df: pd.DataFrame,
    output_dir: Path,
    name: str,
    type_filter: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4.4))
    legend_handles: list[Line2D] = []

    types = [type_filter] if type_filter else stats_df["type"].unique()
    for mtype in types:
        sub = stats_df[stats_df["type"] == mtype]
        if sub.empty:
            continue
        legend_handles = _legend_handles(sub)
        np.vstack([
            sub["time_med"] - sub["time_q1"],
            sub["time_q3"] - sub["time_med"],
        ])
        for idx, (_, row) in enumerate(sub.iterrows()):
            method_color = _color_for_method(str(row["method"]))
            ax.errorbar(
                row["time_med"],
                row["psnr_med"],
                xerr=np.array([
                    [row["time_med"] - row["time_q1"]],
                    [row["time_q3"] - row["time_med"]],
                ]),
                fmt=_marker_for_method(str(row["method"])),
                color=method_color,
                ecolor=method_color,
                elinewidth=0.8,
                capsize=2,
                markersize=5.5,
                zorder=3 + idx / 100,
            )

    ax.set_xscale("log")
    _expand_time_axis(ax, stats_df)
    ax.margins(y=0.08)
    style_axes(
        ax,
        xlabel="Tempo de Inferência (s/patch, mediana - IQR)",
        ylabel="PSNR (dB, mediana)",
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        fontsize=FONT_SIZE - 3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.32)
    save_figure(fig, output_dir, name)
    plt.close(fig)


def fig1_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of PSNR vs elapsed_s for classical methods only."""
    if "elapsed_s" not in df.columns:
        log.warning("No elapsed_s for fig1")
        return

    classic = df[df["type"] == "Clássico"] if "type" in df.columns else df
    if classic.empty:
        log.warning("No classical data for pareto figure")
        return

    preferred_noise = "inf" if "noise_level" in classic.columns else None
    subset = (
        classic[classic["noise_level"] == preferred_noise]
        if preferred_noise is not None
        and preferred_noise in set(classic["noise_level"].unique())
        else classic
    )
    if subset.empty:
        return

    stats_df = _pareto_stats(subset)
    if stats_df.empty:
        return

    _plot_pareto(
        stats_df,
        output_dir,
        "fig_classical_pareto",
        type_filter="Clássico",
    )


def main() -> None:
    run_with_df(fig1_pareto, "Pareto front")


if __name__ == "__main__":
    main()
