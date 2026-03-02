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


# ── Fig 1: Pareto Front (Time x Quality) ─────────────────────────────


def fig1_pareto(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of PSNR vs elapsed_s (log scale) per method."""
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

        stats_df = (
            subset
            .groupby(["method", "type"], observed=True)
            .agg(psnr=("psnr", "mean"), time=("elapsed_s", "mean"))
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(5, 3.5))

        for mtype, color, marker in [
            ("Clássico", "#1f77b4", "o"),
            ("DL", "#ff7f0e", "s"),
        ]:
            sub = stats_df[stats_df["type"] == mtype]
            if sub.empty:
                continue
            ax.scatter(
                sub["time"],
                sub["psnr"],
                c=color,
                marker=marker,
                s=40,
                edgecolors="white",
                linewidth=0.5,
                label=mtype,
                zorder=3,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["method"],
                    (row["time"], row["psnr"]),
                    fontsize=FONT_SIZE - 2,
                    ha="left",
                    va="bottom",
                    xytext=(3, 2),
                    textcoords="offset points",
                )

        ax.set_xscale("log")
        ax.set_xlabel("Tempo de Inferência (s/patch, log)")
        ax.set_ylabel("PSNR (dB)")
        noise_title = noise_label(noise) if noise != "all" else "Global"
        ax.set_title(f"Trade-off Qualidade x Velocidade - {noise_title}")
        ax.legend(loc="best", frameon=True, framealpha=0.85)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        _save(fig, output_dir, f"fig1_pareto_{suffix}")
        plt.close(fig)


# ── Fig 2: Spectral Error Radar Chart ─────────────────────────────────


def fig2_spectral_radar(df: pd.DataFrame, output_dir: Path) -> None:
    """Radar/spider chart of RMSE per band per method category."""
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
        category_means = {}
        for cat in categories:
            cat_df = subset[subset[cat_col] == cat]
            means = [float(cat_df[b].mean()) for b in bands]
            category_means[cat] = means

        N = len(bands)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={"polar": True})
        palette = sns.color_palette("Set2", len(categories))

        for idx, cat in enumerate(categories):
            values = category_means[cat] + [category_means[cat][0]]
            label = CATEGORY_LABELS.get(cat, cat)
            ax.plot(
                angles,
                values,
                linewidth=1.5,
                color=palette[idx],
                label=label,
                marker="o",
                markersize=3,
            )
            ax.fill(angles, values, alpha=0.1, color=palette[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(band_labels, fontsize=FONT_SIZE)
        ax.set_title(
            f"Perfil Espectral de Erro — {noise_label(noise)}",
            fontsize=FONT_SIZE + 1,
            pad=20,
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.1),
            fontsize=FONT_SIZE - 2,
            frameon=True,
        )
        plt.tight_layout()
        _save(fig, output_dir, f"fig2_spectral_radar_{suffix}")
        plt.close(fig)


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
    """Boxplot of PSNR per entropy tercile, faceted by method category."""
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

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(
                data=binned,
                x="method",
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
            ax.set_xlabel("Método")
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
    """Heatmap of Spearman rho: methods x (entropy_window x metric)."""
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

    matrix = np.full((len(methods), len(col_labels)), np.nan)

    for i, method in enumerate(methods):
        mdf = df[df["method"] == method]
        col_idx = 0
        for ws in ENTROPY_WINDOWS:
            ecol = f"entropy_{ws}"
            for m in metrics:
                if ecol in mdf.columns and m in mdf.columns:
                    valid = mdf[[ecol, m]].dropna()
                    if len(valid) >= 3:
                        rho, _ = stats.spearmanr(valid[ecol], valid[m])
                        matrix[i, col_idx] = rho
                col_idx += 1

    fig, ax = plt.subplots(
        figsize=(len(col_labels) * 0.8 + 1, len(methods) * 0.4 + 1)
    )
    sns.heatmap(
        pd.DataFrame(matrix, index=methods, columns=col_labels),
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
        "Correlação de Spearman: Entropia x Métricas", fontsize=FONT_SIZE + 1
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


def _build_dl_comparison_frame(
    models: dict[str, dict[str, list[dict[str, float | None]]]],
    metric_keys: list[tuple[str, str, bool]],
) -> tuple[pd.DataFrame, list[str]]:
    model_labels: list[str] = []
    rows: list[dict[str, float | None]] = []
    for model, hist in models.items():
        last = hist["epochs"][-1]
        row = {label: last.get(key) for key, label, _ in metric_keys}
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
        f"Comparação DL (última época) - {scenario_title}",
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
            ("Fig 2: Spectral Radar", fig2_spectral_radar),
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
