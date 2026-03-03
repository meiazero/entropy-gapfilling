"""Figure 2: Spectral error visualizations."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_loader import CATEGORY_LABELS, NOISE_ORDER, noise_label
from figures.common import FONT_SIZE, SETTINGS, save_figure, style_axes
from scipy import stats

log = logging.getLogger(__name__)


def _cluster_bootstrap_ci(
    values: pd.Series,
    cluster_ids: pd.Series | None,
    *,
    n_boot: int | None = None,
    seed: int = 42,
) -> float:
    samples = n_boot if n_boot is not None else int(SETTINGS.bootstrap_samples)
    vals = values.dropna()
    if len(vals) < 2:
        return 0.0
    if cluster_ids is None or cluster_ids.isna().all():
        return float(
            stats.t.ppf(0.975, len(vals) - 1) * vals.std() / np.sqrt(len(vals))
        )

    data = pd.DataFrame({"value": values, "cluster": cluster_ids}).dropna()
    clusters = data["cluster"].unique().tolist()
    if len(clusters) < 2:
        return float(
            stats.t.ppf(0.975, len(vals) - 1) * vals.std() / np.sqrt(len(vals))
        )

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(samples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = data[data["cluster"].isin(sampled)]
        boot_means.append(float(sample["value"].mean()))
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float((hi - lo) / 2.0)


def _build_spectral_dotplot_rows(
    subset: pd.DataFrame,
    cat_col: str,
    bands: list[str],
    band_labels: list[str],
) -> pd.DataFrame:
    rows = []
    for cat, cat_df in subset.groupby(cat_col, observed=True):
        for band, label in zip(bands, band_labels, strict=False):
            mean_val = float(cat_df[band].mean())
            ci = _cluster_bootstrap_ci(
                cat_df[band],
                cat_df["patch_id"] if "patch_id" in cat_df.columns else None,
                n_boot=int(SETTINGS.bootstrap_samples),
            )
            rows.append({
                "category": CATEGORY_LABELS.get(cat, cat),
                "band": label,
                "mean": mean_val,
                "ci": ci,
            })
    return pd.DataFrame(rows)


def _plot_spectral_dotplot(
    plot_df: pd.DataFrame,
    band_labels: list[str],
    noise: str,
    output_dir: Path,
    suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    categories = plot_df["category"].unique().tolist()
    n_bands = len(band_labels)
    offsets = np.linspace(-0.2, 0.2, n_bands)
    palette = sns.color_palette("Set2", n_bands)

    for b_idx, band in enumerate(band_labels):
        sub = plot_df[plot_df["band"] == band]
        for c_idx, cat in enumerate(categories):
            row = sub[sub["category"] == cat]
            if row.empty:
                continue
            mean_val = float(row["mean"].iloc[0])
            ci = float(row["ci"].iloc[0])
            y = c_idx + offsets[b_idx]
            ax.errorbar(
                mean_val,
                y,
                xerr=ci,
                fmt="o",
                color=palette[b_idx],
                ecolor="#555555",
                elinewidth=0.8,
                capsize=2,
            )

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    handles = [
        plt.Line2D([0], [0], marker="o", color=c, linestyle="") for c in palette
    ]
    ax.legend(
        handles,
        band_labels,
        title="Banda",
        fontsize=FONT_SIZE - 2,
    )
    style_axes(
        ax,
        title=f"RMSE por banda — {noise_label(noise)}",
        xlabel=r"RMSE (IC95\%)",
        ylabel="Categoria",
        grid_axis="x",
    )
    plt.tight_layout()
    save_figure(fig, output_dir, f"fig2_spectral_dotplot_{suffix}")
    plt.close(fig)


def fig2_spectral_bar(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of RMSE per band with CI95 by method category."""
    bands = ["rmse_b0", "rmse_b1", "rmse_b2", "rmse_b3"]
    band_labels = ["B0\n(Azul)", "B1\n(Verde)", "B2\n(Verm.)", "B3\n(NIR)"]

    if not all(b in df.columns for b in bands):
        log.warning("Missing band columns for fig2")
        return

    if "method_category" in df.columns:
        cat_col = "method_category"
    elif "type" in df.columns:
        cat_col = "type"
    else:
        log.warning("No category column for fig2")
        return

    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]

    for noise in noises:
        subset = df[df["noise_level"] == noise]
        if subset.empty:
            continue
        suffix = noise.replace("inf", "gap_only")

        categories = sorted(subset[cat_col].unique())
        palette = sns.color_palette("Set2", len(categories))
        x = np.arange(len(bands))
        width = 0.8 / max(1, len(categories))
        fig, ax = plt.subplots(figsize=(6.5, 3.5))

        for idx, cat in enumerate(categories):
            cat_df = subset[subset[cat_col] == cat]
            means = []
            cis = []
            for b in bands:
                means.append(float(cat_df[b].mean()))
                cis.append(
                    _cluster_bootstrap_ci(
                        cat_df[b],
                        cat_df["patch_id"]
                        if "patch_id" in cat_df.columns
                        else None,
                        n_boot=int(SETTINGS.bootstrap_samples),
                    )
                )
            offsets = x + (idx - (len(categories) - 1) / 2) * width
            ax.bar(
                offsets,
                means,
                width=width,
                yerr=cis,
                label=CATEGORY_LABELS.get(cat, cat),
                color=palette[idx],
                edgecolor="#333333",
                linewidth=0.5,
                capsize=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(band_labels)
        style_axes(
            ax,
            title=rf"RMSE por banda (IC95\%) — {noise_label(noise)}",
            ylabel="RMSE",
            grid_axis="y",
            title_size=FONT_SIZE + 1,
        )
        ax.legend(loc="best", fontsize=FONT_SIZE - 2, frameon=True)
        plt.tight_layout()
        save_figure(fig, output_dir, f"fig2_spectral_bar_{suffix}")
        plt.close(fig)


def fig2_spectral_dotplot(df: pd.DataFrame, output_dir: Path) -> None:
    """Dot plot of RMSE per band with CI95 by method category."""
    bands = ["rmse_b0", "rmse_b1", "rmse_b2", "rmse_b3"]
    band_labels = ["B0", "B1", "B2", "B3"]

    if not all(b in df.columns for b in bands):
        log.warning("Missing band columns for fig2 dotplot")
        return

    if "method_category" in df.columns:
        cat_col = "method_category"
    elif "type" in df.columns:
        cat_col = "type"
    else:
        log.warning("No category column for fig2 dotplot")
        return

    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]
    for noise in noises:
        subset = df[df["noise_level"] == noise]
        if subset.empty:
            continue
        suffix = noise.replace("inf", "gap_only")

        plot_df = _build_spectral_dotplot_rows(
            subset, cat_col, bands, band_labels
        )
        if plot_df.empty:
            continue
        _plot_spectral_dotplot(plot_df, band_labels, noise, output_dir, suffix)
