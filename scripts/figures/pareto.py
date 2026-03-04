"""Figure 1: Pareto front (time x quality)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data_loader import NOISE_ORDER, noise_label
from .cli import run_with_df
from .common import FONT_SIZE, save_figure, style_axes

log = logging.getLogger(__name__)


def _iqr_bounds(values: pd.Series) -> tuple[float, float, float]:
    vals = values.dropna().to_numpy()
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    q1 = float(np.percentile(vals, 25))
    q2 = float(np.percentile(vals, 50))
    q3 = float(np.percentile(vals, 75))
    return q1, q2, q3


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


def _plot_pareto(
    stats_df: pd.DataFrame,
    output_dir: Path,
    name: str,
    title: str,
    type_filter: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    palette = {"Clássico": "#1f77b4", "DL": "#ff7f0e"}
    markers = {"Clássico": "o", "DL": "s"}

    types = [type_filter] if type_filter else stats_df["type"].unique()
    for mtype in types:
        sub = stats_df[stats_df["type"] == mtype]
        if sub.empty:
            continue
        xerr = np.vstack([
            sub["time_med"] - sub["time_q1"],
            sub["time_q3"] - sub["time_med"],
        ])
        ax.errorbar(
            sub["time_med"],
            sub["psnr_med"],
            xerr=xerr,
            fmt=markers.get(mtype, "o"),
            color=palette.get(mtype, "#1f77b4"),
            ecolor=palette.get(mtype, "#1f77b4"),
            elinewidth=0.8,
            capsize=2,
            markersize=5,
            label=mtype,
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["method"],
                (row["time_med"], row["psnr_med"]),
                fontsize=FONT_SIZE - 2,
                ha="left",
                va="bottom",
                xytext=(3, 2),
                textcoords="offset points",
            )

    ax.set_xscale("log")
    style_axes(
        ax,
        title=title,
        xlabel="Tempo de Inferência (s/patch, mediana - IQR)",
        ylabel="PSNR (dB, mediana)",
    )
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    plt.tight_layout()
    save_figure(fig, output_dir, name)
    plt.close(fig)


def _plot_pareto_variants(
    stats_df: pd.DataFrame,
    subset: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    noise: str,
) -> None:
    title = (
        "Trade-off Qualidade x Velocidade - "
        f"{noise_label(noise) if noise != 'all' else 'Global'}"
    )
    _plot_pareto(stats_df, output_dir, f"fig1_pareto_{suffix}", title)

    _plot_pareto(
        stats_df,
        output_dir,
        f"fig1_pareto_classic_{suffix}",
        f"Clássicos: Qualidade x Velocidade - {noise_label(noise)}",
        type_filter="Clássico",
    )

    if "satellite" in subset.columns:
        classic = subset[subset["type"] == "Clássico"]
        for sat in sorted(classic["satellite"].unique()):
            sat_df = _pareto_stats(classic[classic["satellite"] == sat])
            if sat_df.empty:
                continue
            _plot_pareto(
                sat_df,
                output_dir,
                f"fig1_pareto_classic_{sat}_{suffix}",
                f"Clássicos ({sat}) - {noise_label(noise)}",
                type_filter="Clássico",
            )

    if "satellite" in subset.columns:
        sent2 = subset[subset["satellite"] == "sentinel2"]
        if not sent2.empty:
            sent_stats = _pareto_stats(sent2)
            if not sent_stats.empty:
                _plot_pareto(
                    sent_stats,
                    output_dir,
                    f"fig1_pareto_sentinel2_{suffix}",
                    f"Sentinel-2: Clássico vs DL - {noise_label(noise)}",
                )


def fig1_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of PSNR vs elapsed_s with median/IQR time."""
    if "elapsed_s" not in df.columns:
        log.warning("No elapsed_s for fig1")
        return

    noises = ["all"] + [
        n for n in NOISE_ORDER if n in df["noise_level"].unique()
    ]

    for noise in noises:
        subset = df if noise == "all" else df[df["noise_level"] == noise]
        if subset.empty:
            continue
        suffix = noise.replace("inf", "gap_only")

        stats_df = _pareto_stats(subset)
        if stats_df.empty:
            continue

        _plot_pareto_variants(stats_df, subset, output_dir, suffix, noise)


def main() -> None:
    run_with_df(fig1_pareto, "Pareto front")


if __name__ == "__main__":
    main()
