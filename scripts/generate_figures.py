"""Generate publication-quality figures from experiment results.

Data-driven rewrite. Produces 11 figure types with variations per noise
level, entropy window, model, and entropy scenario.

Usage:
    uv run python scripts/generate_figures.py
    uv run python scripts/generate_figures.py --output docs/figures
    uv run python scripts/generate_figures.py --png-only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data_loader import (
    CATEGORY_LABELS,
    ENTROPY_WINDOWS,
    NOISE_ORDER,
    entropy_terciles,
    load_all_dl_histories,
    load_combined,
    noise_label,
    select_top_n,
)
from scipy import stats
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Styling ───────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STYLE_PATH = _PROJECT_ROOT / "images" / "style.mplstyle"

DPI = 300
FONT_SIZE = 8

_PUB_FONT = {
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

_SETTINGS: dict[str, bool] = {"png_only": False}


def _setup_style() -> None:
    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))
    plt.rcParams.update(_PUB_FONT)
    sns.set_palette("Set2")


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    if not _SETTINGS["png_only"]:
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    log.info("Saved %s", name)


def _iqr_bounds(values: pd.Series) -> tuple[float, float, float]:
    vals = values.dropna().to_numpy()
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    q1 = float(np.percentile(vals, 25))
    q2 = float(np.percentile(vals, 50))
    q3 = float(np.percentile(vals, 75))
    return q1, q2, q3


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    # d = (2U / (n1*n2)) - 1
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = a.size, b.size
    return float((2.0 * u_stat / (n1 * n2)) - 1.0)


# ── Fig 1: Pareto Front (Time x Quality) ─────────────────────────────


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
    ax.set_xlabel("Tempo de Inferência (s/patch, mediana - IQR)")
    ax.set_ylabel("PSNR (dB, mediana)")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, framealpha=0.85)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save(fig, output_dir, name)
    plt.close(fig)


def _plot_pareto_variants(
    stats_df: pd.DataFrame,
    subset: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    noise: str,
) -> None:
    title = (
        f"Trade-off Qualidade x Velocidade - "
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


# ── Fig 2: Spectral Error Radar Chart ─────────────────────────────────


def _cluster_bootstrap_ci(
    values: pd.Series,
    cluster_ids: pd.Series | None,
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> float:
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
    for _ in range(n_boot):
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
    ax.set_xlabel(r"RMSE (IC95\%)")
    ax.set_ylabel("Categoria")
    ax.set_title(f"RMSE por banda — {noise_label(noise)}")
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save(fig, output_dir, f"fig2_spectral_dotplot_{suffix}")
    plt.close(fig)


def fig2_spectral_bar(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of RMSE per band with CI95 by method category."""
    bands = ["rmse_b0", "rmse_b1", "rmse_b2", "rmse_b3"]
    band_labels = ["B0\n(Azul)", "B1\n(Verde)", "B2\n(Verm.)", "B3\n(NIR)"]

    if not all(b in df.columns for b in bands):
        log.warning("Missing band columns for fig2")
        return

    # Build category display names
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
        ax.set_ylabel("RMSE")
        ax.set_title(
            rf"RMSE por banda (IC95\%) — {noise_label(noise)}",
            fontsize=FONT_SIZE + 1,
        )
        ax.legend(loc="best", fontsize=FONT_SIZE - 2, frameon=True)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        _save(fig, output_dir, f"fig2_spectral_bar_{suffix}")
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


# ── Fig 3: Entropy Sensitivity (SAM/ERGAS vs Entropy) ────────────────


def fig3_sensitivity(df: pd.DataFrame, output_dir: Path) -> None:
    """Regression plots of SAM/ERGAS vs entropy for top methods."""
    top_classic = select_top_n(df[df["type"] == "Clássico"], n=3)
    top_dl = select_top_n(df[df["type"] == "DL"], n=3)
    selected = top_classic + top_dl

    if not selected:
        log.warning("No methods for fig3")
        return

    metrics_to_plot = [m for m in ["sam", "ergas"] if m in df.columns]
    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]
    palette = sns.color_palette("Set2", len(selected))

    for metric in metrics_to_plot:
        for ws in ENTROPY_WINDOWS:
            ecol = f"entropy_{ws}"
            if ecol not in df.columns:
                continue
            for noise in noises:
                subset = df[df["noise_level"] == noise]
                if subset.empty:
                    continue
                suffix = noise.replace("inf", "gap_only")

                fig, ax = plt.subplots(figsize=(5, 3.5))

                for idx, method in enumerate(selected):
                    mdf = subset[subset["method"] == method][
                        [ecol, metric]
                    ].dropna()
                    if len(mdf) < 10:
                        continue
                    sns.regplot(
                        data=mdf,
                        x=ecol,
                        y=metric,
                        ax=ax,
                        color=palette[idx],
                        scatter_kws={"s": 8, "alpha": 0.4, "rasterized": True},
                        line_kws={"linewidth": 1.5},
                        label=method,
                        ci=95,
                    )

                ax.set_xlabel(f"Entropia ({ws}x{ws})")
                ax.set_ylabel(metric.upper())
                ax.set_title(
                    f"Sensibilidade à Entropia — {metric.upper()} "
                    f"({noise_label(noise)})"
                )
                ax.legend(loc="best", fontsize=FONT_SIZE - 2, frameon=True)
                ax.grid(True, alpha=0.2, linewidth=0.5)
                plt.tight_layout()
                _save(
                    fig, output_dir, f"fig3_sensitivity_{metric}_{suffix}_e{ws}"
                )
                plt.close(fig)


# ── Fig 4: Multi-Sensor Violin (Classic Only) ────────────────────────


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
        ax.set_xlabel("Satélite")
        ax.set_ylabel("SSIM")
        ax.set_title(
            f"Distribuição SSIM por Sensor — {noise_label(noise)} "
            f"(Top-3 Clássicos)"
        )
        ax.legend(title="Método", loc="best", fontsize=FONT_SIZE - 2)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        _save(fig, output_dir, f"fig4_multisensor_{suffix}")
        plt.close(fig)


# ── Fig 5: F1 Score by Threshold ──────────────────────────────────────


def fig5_f1_threshold(df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart of F1 scores per model at each threshold."""
    thresholds = [
        ("f1_002", r"$\tau=0{,}02$"),
        ("f1_005", r"$\tau=0{,}05$"),
        ("f1_01", r"$\tau=0{,}10$"),
    ]
    available_thresholds = [
        (key, label) for key, label in thresholds if key in df.columns
    ]
    if not available_thresholds:
        log.warning("No F1 columns for fig5")
        return

    methods = sorted(df["method"].unique())
    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]
    palette = sns.color_palette("Set2", len(methods))

    for noise in noises:
        subset = df[df["noise_level"] == noise]
        if subset.empty:
            continue
        suffix = noise.replace("inf", "gap_only")

        means = {}
        for method in methods:
            mdf = subset[subset["method"] == method]
            means[method] = [
                float(mdf[k].mean()) if k in mdf.columns else 0.0
                for k, _ in available_thresholds
            ]

        n_t = len(available_thresholds)
        n_m = len(methods)
        x = np.arange(n_t)
        width = 0.75 / n_m
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

        fig, ax = plt.subplots(figsize=(6, 3.5))

        for i, method in enumerate(methods):
            bars = ax.bar(
                x + offsets[i],
                means[method],
                width=width,
                label=method,
                color=palette[i],
                edgecolor="#333333",
                linewidth=0.5,
            )
            for bar, val in zip(bars, means[method], strict=False):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=FONT_SIZE - 3,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in available_thresholds])
        ax.set_xlabel("Limiar de erro")
        ax.set_ylabel("F1 Score")
        ax.set_title(f"F1 por Limiar — {noise_label(noise)}")
        ax.set_ylim(0, 1.15)
        ax.legend(
            fontsize=FONT_SIZE - 2,
            loc="upper left",
            frameon=True,
            ncol=min(4, n_m),
        )
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        _save(fig, output_dir, f"fig5_f1_threshold_{suffix}")
        plt.close(fig)


# ── Fig 6: Boxplot PSNR by Entropy Bin ────────────────────────────────


def fig6_psnr_entropy(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of PSNR per entropy tercile, top-k per category."""
    noises = [n for n in NOISE_ORDER if n in df["noise_level"].unique()]

    for ws in ENTROPY_WINDOWS:
        ecol = f"entropy_{ws}"
        if ecol not in df.columns:
            continue

        for noise in noises:
            subset = df[df["noise_level"] == noise]
            if subset.empty:
                continue
            suffix = noise.replace("inf", "gap_only")
            binned = entropy_terciles(subset, entropy_col=ecol)

            top_methods: list[str] = []
            for mtype in ["Clássico", "DL"]:
                mdf = binned[binned["type"] == mtype]
                top_methods.extend(select_top_n(mdf, n=5, noise_filter=None))
            binned = binned[binned["method"].isin(top_methods)]
            if binned.empty:
                continue

            # Add Cliff's delta (high vs low entropy) to method labels
            effect_labels: dict[str, str] = {}
            for method, mdf in binned.groupby("method", observed=True):
                low = mdf[mdf["entropy_bin"] == "baixa"]["psnr"].dropna()
                high = mdf[mdf["entropy_bin"] == "alta"]["psnr"].dropna()
                delta = _cliffs_delta(high.to_numpy(), low.to_numpy())
                if np.isnan(delta):
                    label = method
                else:
                    label = f"{method}\nΔ={delta:+.2f}"
                effect_labels[method] = label
            binned = binned.assign(
                method_label=binned["method"].map(effect_labels)
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(
                data=binned,
                x="method_label",
                y="psnr",
                hue="entropy_bin",
                hue_order=["baixa", "média", "alta"],
                palette="Set2",
                ax=ax,
                fliersize=3,
                linewidth=0.8,
                showmeans=True,
                meanprops={
                    "marker": "x",
                    "markeredgecolor": "#333",
                    "markersize": 4,
                },
            )
            ax.set_xlabel("Método (Δ = Cliff's d alta vs baixa)")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title(
                f"PSNR por Faixa de Entropia ({ws}x{ws}) - {noise_label(noise)}"
            )
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="Entropia", loc="best", fontsize=FONT_SIZE - 2)
            plt.tight_layout()
            _save(fig, output_dir, f"fig6_psnr_entropy_{suffix}_e{ws}")
            plt.close(fig)


# ── Fig 7: Correlation Heatmap ────────────────────────────────────────


def fig7_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of Spearman rho with FDR and effect filtering."""
    metrics = [m for m in ["psnr", "ssim", "sam", "ergas"] if m in df.columns]
    if not metrics:
        log.warning("No metrics for fig7")
        return

    methods = sorted(df["method"].unique())

    # Build matrix: rows=methods, cols=entropy_ws x metric
    col_labels = []
    for ws in ENTROPY_WINDOWS:
        for m in metrics:
            col_labels.append(f"e{ws}x{m.upper()}")

    rho_matrix = np.full((len(methods), len(col_labels)), np.nan)
    p_matrix = np.full((len(methods), len(col_labels)), np.nan)

    for i, method in enumerate(methods):
        mdf = df[df["method"] == method]
        col_idx = 0
        for ws in ENTROPY_WINDOWS:
            ecol = f"entropy_{ws}"
            for m in metrics:
                if ecol in mdf.columns and m in mdf.columns:
                    valid = mdf[[ecol, m]].dropna()
                    if len(valid) >= 3:
                        rho, p = stats.spearmanr(valid[ecol], valid[m])
                        rho_matrix[i, col_idx] = rho
                        p_matrix[i, col_idx] = p
                col_idx += 1

    # FDR correction across all tests
    p_vals = p_matrix.ravel()
    valid = ~np.isnan(p_vals)
    corr_p = np.full_like(p_vals, np.nan, dtype=float)
    sig = np.full_like(p_vals, False, dtype=bool)
    if np.any(valid):
        reject, p_corr, _, _ = multipletests(p_vals[valid], method="fdr_bh")
        corr_p[valid] = p_corr
        sig[valid] = reject
    sig_matrix = sig.reshape(p_matrix.shape)

    # Mask low magnitude or non-significant correlations
    effect_mask = np.abs(rho_matrix) < 0.1
    final = rho_matrix.copy()
    final[~sig_matrix] = np.nan
    final[effect_mask] = np.nan

    fig, ax = plt.subplots(
        figsize=(len(col_labels) * 0.8 + 1, len(methods) * 0.4 + 1)
    )
    sns.heatmap(
        pd.DataFrame(final, index=methods, columns=col_labels),
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        ax=ax,
        annot_kws={"size": FONT_SIZE - 2},
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title(
        "Correlação Spearman (FDR, |rho|>=0,1)",
        fontsize=FONT_SIZE + 1,
    )
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    _save(fig, output_dir, "fig7_correlation_heatmap")
    plt.close(fig)


# ── Fig 8: DL Loss Curves (per model) ────────────────────────────────


def fig8_dl_loss(output_dir: Path) -> None:
    """Train vs val loss per model, one figure per scenario x model."""
    histories = load_all_dl_histories()

    for scenario, models in histories.items():
        for model, hist in models.items():
            epochs_data = hist.get("epochs", [])
            if not epochs_data:
                continue

            epochs = [e["epoch"] for e in epochs_data]
            train_loss = [e.get("train_loss") for e in epochs_data]
            val_loss = [e.get("val_loss") for e in epochs_data]

            fig, ax = plt.subplots(figsize=(4, 2.8))

            every = max(1, len(epochs) // 8)
            ax.plot(
                epochs,
                train_loss,
                label="Treino",
                linewidth=1.2,
                color="#1f77b4",
                marker="o",
                markevery=every,
                markersize=3,
            )
            ax.plot(
                epochs,
                val_loss,
                label="Validação",
                linewidth=1.2,
                color="#ff7f0e",
                linestyle="--",
                marker="s",
                markevery=every,
                markersize=3,
            )
            ax.set_xlabel("Época")
            ax.set_ylabel("Perda")
            ax.set_title(
                f"{model.upper()} — {scenario.replace('_', ' ').title()}"
            )
            ax.legend(fontsize=FONT_SIZE - 1, loc="best")
            ax.grid(True, alpha=0.3, linewidth=0.5)
            plt.tight_layout()
            _save(fig, output_dir, f"fig8_dl_loss_{scenario}_{model}")
            plt.close(fig)


# ── Fig 9: DL Validation Metrics (per model x metric) ────────────────


def fig9_dl_val_metrics(output_dir: Path) -> None:
    """Val PSNR/SSIM/RMSE per epoch.

    One figure per scenario x metric x model.
    """
    histories = load_all_dl_histories()
    metric_specs = [
        ("val_psnr", "PSNR (dB)"),
        ("val_ssim", "SSIM"),
        ("val_rmse", "RMSE"),
    ]

    for scenario, models in histories.items():
        for model, hist in models.items():
            epochs_data = hist.get("epochs", [])
            if not epochs_data:
                continue
            epochs = [e["epoch"] for e in epochs_data]

            for key, ylabel in metric_specs:
                values = [e.get(key) for e in epochs_data]
                if not any(v is not None for v in values):
                    continue

                fig, ax = plt.subplots(figsize=(4, 2.8))
                every = max(1, len(epochs) // 8)
                ax.plot(
                    epochs,
                    values,
                    linewidth=1.2,
                    color="#2ca02c",
                    marker="o",
                    markevery=every,
                    markersize=3,
                )
                ax.set_xlabel("Época")
                ax.set_ylabel(ylabel)
                ax.set_title(
                    f"{model.upper()} — {ylabel} "
                    f"({scenario.replace('_', ' ').title()})"
                )
                ax.grid(True, alpha=0.3, linewidth=0.5)
                plt.tight_layout()
                metric_name = key.replace("val_", "")
                _save(
                    fig,
                    output_dir,
                    f"fig9_dl_val_metrics_{scenario}_{metric_name}_{model}",
                )
                plt.close(fig)


# ── Fig 10: VAE/GAN Component Decomposition ──────────────────────────


def fig10_components(output_dir: Path) -> None:
    """VAE reconstruction+KL and GAN generator+discriminator decomposition."""
    histories = load_all_dl_histories()

    for scenario, models in histories.items():
        # VAE decomposition
        vae_hist = models.get("vae")
        if vae_hist and vae_hist.get("epochs"):
            epochs_data = vae_hist["epochs"]
            if any("train_recon_loss" in e for e in epochs_data):
                epochs = [e["epoch"] for e in epochs_data]
                fig, axes = plt.subplots(
                    1, 2, figsize=(7, 2.8), constrained_layout=True
                )

                axes[0].plot(
                    epochs,
                    [e.get("train_recon_loss") for e in epochs_data],
                    label="Treino",
                    linewidth=1.2,
                    color="#1f77b4",
                )
                axes[0].plot(
                    epochs,
                    [e.get("val_recon_loss") for e in epochs_data],
                    label="Validação",
                    linewidth=1.2,
                    color="#ff7f0e",
                    linestyle="--",
                )
                axes[0].set_title("Perda de Reconstrução")
                axes[0].set_xlabel("Época")
                axes[0].set_ylabel("Perda")
                axes[0].legend(fontsize=FONT_SIZE - 1)
                axes[0].grid(True, alpha=0.3, linewidth=0.5)

                axes[1].plot(
                    epochs,
                    [e.get("train_kl_loss") for e in epochs_data],
                    label="Treino",
                    linewidth=1.2,
                    color="#1f77b4",
                )
                axes[1].plot(
                    epochs,
                    [e.get("val_kl_loss") for e in epochs_data],
                    label="Validação",
                    linewidth=1.2,
                    color="#ff7f0e",
                    linestyle="--",
                )
                axes[1].set_title("Divergência KL")
                axes[1].set_xlabel("Época")
                axes[1].set_ylabel("Perda")
                axes[1].legend(fontsize=FONT_SIZE - 1)
                axes[1].grid(True, alpha=0.3, linewidth=0.5)

                fig.suptitle(
                    f"VAE — {scenario.replace('_', ' ').title()}",
                    fontsize=FONT_SIZE + 1,
                )
                _save(fig, output_dir, f"fig10_vae_components_{scenario}")
                plt.close(fig)

        # GAN decomposition
        gan_hist = models.get("gan")
        if gan_hist and gan_hist.get("epochs"):
            epochs_data = gan_hist["epochs"]
            if any("train_g_loss" in e for e in epochs_data):
                epochs = [e["epoch"] for e in epochs_data]
                fig, axes = plt.subplots(
                    1, 3, figsize=(9, 2.8), constrained_layout=True
                )

                axes[0].plot(
                    epochs,
                    [e.get("train_g_loss") for e in epochs_data],
                    linewidth=1.2,
                    color="#2ca02c",
                )
                axes[0].set_title("Perda do Gerador")
                axes[0].set_xlabel("Época")
                axes[0].set_ylabel("Perda")
                axes[0].grid(True, alpha=0.3, linewidth=0.5)

                axes[1].plot(
                    epochs,
                    [e.get("train_d_loss") for e in epochs_data],
                    linewidth=1.2,
                    color="#d62728",
                )
                axes[1].set_title("Perda do Discriminador")
                axes[1].set_xlabel("Época")
                axes[1].set_ylabel("Perda")
                axes[1].grid(True, alpha=0.3, linewidth=0.5)

                axes[2].plot(
                    epochs,
                    [e.get("val_loss") for e in epochs_data],
                    linewidth=1.2,
                    color="#9467bd",
                )
                axes[2].set_title("Perda de Validação (G)")
                axes[2].set_xlabel("Época")
                axes[2].set_ylabel("Perda")
                axes[2].grid(True, alpha=0.3, linewidth=0.5)

                fig.suptitle(
                    f"GAN — {scenario.replace('_', ' ').title()}",
                    fontsize=FONT_SIZE + 1,
                )
                _save(fig, output_dir, f"fig10_gan_components_{scenario}")
                plt.close(fig)


# ── Fig 11: DL Model Comparison Heatmap ───────────────────────────────


def _select_best_epoch(hist: dict) -> dict:
    epochs = hist.get("epochs", [])
    if not epochs:
        return {}
    # Prefer lowest val_loss, fallback to highest val_psnr, else last epoch
    valid_loss = [e for e in epochs if e.get("val_loss") is not None]
    if valid_loss:
        return min(valid_loss, key=lambda e: e.get("val_loss"))
    valid_psnr = [e for e in epochs if e.get("val_psnr") is not None]
    if valid_psnr:
        return max(valid_psnr, key=lambda e: e.get("val_psnr"))
    return epochs[-1]


def _build_dl_comparison_frame(
    models: dict[str, dict[str, list[dict[str, float | None]]]],
    metric_keys: list[tuple[str, str, bool]],
) -> tuple[pd.DataFrame, list[str]]:
    model_labels: list[str] = []
    rows: list[dict[str, float | None]] = []
    for model, hist in models.items():
        best = _select_best_epoch(hist)
        if not best:
            continue
        row = {label: best.get(key) for key, label, _ in metric_keys}
        rows.append(row)
        model_labels.append(model.upper())
    return pd.DataFrame(rows, index=model_labels), model_labels


def _normalize_dl_metrics(
    raw_df: pd.DataFrame,
    higher_better: dict[str, bool],
) -> pd.DataFrame:
    normed = pd.DataFrame(
        index=raw_df.index, columns=raw_df.columns, dtype=float
    )
    for col in raw_df.columns:
        col_min = raw_df[col].min()
        col_max = raw_df[col].max()
        rng = col_max - col_min
        if rng < 1e-12:
            normed[col] = 0.5
        elif higher_better.get(col, True):
            normed[col] = (raw_df[col] - col_min) / rng
        else:
            normed[col] = (col_max - raw_df[col]) / rng
    return normed


def _format_dl_cell_value(col: str, raw_val: float | None) -> str:
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return "N/D"
    if col == "PSNR":
        return f"{raw_val:.2f}"
    return f"{raw_val:.4f}"


def _render_dl_comparison_heatmap(
    raw_df: pd.DataFrame,
    normed: pd.DataFrame,
    col_labels: list[str],
    model_labels: list[str],
    scenario: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            len(col_labels) * 1.0 + 0.8,
            len(model_labels) * 0.6 + 0.8,
        ),
        constrained_layout=True,
    )

    im = ax.imshow(
        normed.values.astype(float),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=FONT_SIZE)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=FONT_SIZE)

    for r_idx in range(len(model_labels)):
        for c_idx, col in enumerate(col_labels):
            raw_val = raw_df.iloc[r_idx][col]
            cell_text = _format_dl_cell_value(col, raw_val)
            bg_val = normed.iloc[r_idx][col]
            bg = 0.5 if pd.isna(bg_val) else float(bg_val)
            text_color = "black" if 0.25 < bg < 0.85 else "white"
            ax.text(
                c_idx,
                r_idx,
                cell_text,
                ha="center",
                va="center",
                fontsize=FONT_SIZE - 1,
                color=text_color,
            )

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="Pontuação norm.")
    scenario_title = scenario.replace("_", " ").title()
    ax.set_title(
        f"Comparação DL (melhor época) - {scenario_title}",
        fontsize=FONT_SIZE + 1,
        pad=8,
    )
    _save(fig, output_dir, f"fig11_dl_comparison_{scenario}")
    plt.close(fig)


def fig11_dl_comparison(output_dir: Path) -> None:
    """Heatmap comparing all DL models across final-epoch metrics."""
    histories = load_all_dl_histories()

    metric_keys = [
        ("val_psnr", "PSNR", True),
        ("val_ssim", "SSIM", True),
        ("val_rmse", "RMSE", False),
        ("val_sam", "SAM", False),
        ("val_ergas", "ERGAS", False),
        ("val_f1_002", "F1@0.02", True),
        ("val_f1_005", "F1@0.05", True),
        ("val_f1_01", "F1@0.10", True),
    ]

    for scenario, models in histories.items():
        if not models:
            continue

        raw_df, model_labels = _build_dl_comparison_frame(models, metric_keys)
        col_labels = [label for _, label, _ in metric_keys]
        higher_better = {label: higher for _, label, higher in metric_keys}
        normed = _normalize_dl_metrics(raw_df, higher_better)
        _render_dl_comparison_heatmap(
            raw_df,
            normed,
            col_labels,
            model_labels,
            scenario,
            output_dir,
        )


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication figures."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to paper_assets/figures/",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Save PNG only (skip PDF).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _SETTINGS["png_only"] = args.png_only
    _setup_style()

    output_dir = args.output or Path("paper_assets/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined()

    # ── Evaluation data figures ──
    if not df.empty:
        eval_figs = [
            ("Fig 1: Pareto", fig1_pareto),
            ("Fig 2a: Spectral Bar", fig2_spectral_bar),
            ("Fig 2b: Spectral Dotplot", fig2_spectral_dotplot),
            ("Fig 3: Entropy Sensitivity", fig3_sensitivity),
            ("Fig 4: Multi-Sensor Violin", fig4_multisensor),
            ("Fig 5: F1 Threshold", fig5_f1_threshold),
            ("Fig 6: PSNR by Entropy", fig6_psnr_entropy),
            ("Fig 7: Correlation Heatmap", fig7_correlation_heatmap),
        ]
        for name, func in eval_figs:
            try:
                log.info("Generating %s...", name)
                func(df, output_dir)
            except Exception:
                log.exception("Error generating %s", name)
    else:
        log.warning("No evaluation data loaded. Skipping eval figures.")

    # ── DL training history figures ──
    dl_figs = [
        ("Fig 8: DL Loss Curves", fig8_dl_loss),
        ("Fig 9: DL Val Metrics", fig9_dl_val_metrics),
        ("Fig 10: VAE/GAN Components", fig10_components),
        ("Fig 11: DL Comparison", fig11_dl_comparison),
    ]
    for name, func in dl_figs:
        try:
            log.info("Generating %s...", name)
            func(output_dir)
        except Exception:
            log.exception("Error generating %s", name)

    log.info("All figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
