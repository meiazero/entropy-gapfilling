"""Classical Pareto front (time x quality)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        xlabel="Tempo de Inferência (s/patch, mediana - IQR)",
        ylabel="PSNR (dB, mediana)",
    )
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    plt.tight_layout()
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
