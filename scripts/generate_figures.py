"""Generate publication-quality figures from experiment results.

Produces 9 figure types for the journal paper. All figures use
matplotlib + seaborn with consistent styling.

Usage:
    uv run python scripts/generate_figures.py --results results/paper_results
    uv run python scripts/generate_figures.py \
        --results results/paper_results --figure 2
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

from pdi_pipeline.aggregation import (
    load_results,
)
from pdi_pipeline.logging_utils import setup_file_logging, setup_logging
from pdi_pipeline.visualization import to_display_rgb

matplotlib.use("Agg")

setup_logging()
log = logging.getLogger(__name__)

# Publication styling
FIGSIZE_SINGLE = (3.5, 3.0)
FIGSIZE_DOUBLE = (7.0, 4.0)
FIGSIZE_GRID = (7.0, 8.0)
DPI = 300
FONT_SIZE = 8

STYLE_PARAMS = {
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


_PNG_ONLY: bool = False


def _collect_numeric_stems(base_dir: Path, pattern: str) -> set[int]:
    ids: set[int] = set()
    if not base_dir.exists():
        return ids

    for method_dir in base_dir.iterdir():
        if not method_dir.is_dir() or method_dir.name.startswith("_"):
            continue
        for path in method_dir.glob(pattern):
            try:
                ids.add(int(path.stem))
            except ValueError:
                continue
    return ids


def _available_recon_patch_ids(
    recon_dir: Path,
    noise_level: str = "inf",
) -> set[int]:
    """Return patch IDs that have reconstruction arrays or PNGs."""
    npy_base = recon_dir.parent / "reconstruction_arrays" / noise_level
    if npy_base.exists():
        return _collect_numeric_stems(npy_base, "*.npy")
    return _collect_numeric_stems(recon_dir, "*.png")


def _pick_entropy_representatives(
    df: pd.DataFrame,
    percentiles: dict[str, float],
    extra_col: str,
) -> list[tuple[str, int, str]]:
    reps: list[tuple[str, int, str]] = []
    for label, q in percentiles.items():
        target = float(df["entropy_7"].quantile(q))
        closest = df.iloc[(df["entropy_7"] - target).abs().argsort()[:1]].iloc[
            0
        ]
        reps.append((label, int(closest["patch_id"]), str(closest[extra_col])))
    return reps


def _load_clean_and_mask(
    preprocessed: Path,
    satellite: str,
    patch_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    clean_path = preprocessed / "test" / satellite / f"{patch_id:07d}_clean.npy"
    mask_path = preprocessed / "test" / satellite / f"{patch_id:07d}_mask.npy"
    if not clean_path.exists() or not mask_path.exists():
        return None

    clean = np.load(clean_path)
    mask = np.load(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return clean, mask


def _compute_error_map(clean: np.ndarray, recon: np.ndarray) -> np.ndarray:
    recon_matched = recon
    clean_matched = clean
    if clean.ndim == 3 and recon.ndim == 3:
        common_channels = min(clean.shape[2], recon.shape[2])
        clean_matched = clean[:, :, :common_channels]
        recon_matched = recon[:, :, :common_channels]

    return np.mean(
        (clean_matched.astype(np.float32) - recon_matched.astype(np.float32))
        ** 2,
        axis=2 if clean_matched.ndim == 3 else None,
    )


def _choose_fig5_methods(valid: pd.DataFrame) -> list[str] | None:
    method_means = (
        valid.groupby("method")["psnr"].mean().sort_values(ascending=False)
    )
    if len(method_means) < 2:
        return None

    best_method = str(method_means.index[0])
    mid_method = str(method_means.index[len(method_means) // 2])
    return [best_method, mid_method]


def _render_fig5_row(
    axes: np.ndarray,
    row_idx: int,
    ncols: int,
    results_dir: Path,
    preprocessed: Path,
    patch_id: int,
    satellite: str,
    label: str,
    selected_methods: list[str],
    cmap: matplotlib.colors.Colormap,
) -> None:
    from pdi_pipeline.statistics import spatial_autocorrelation

    loaded = _load_clean_and_mask(preprocessed, satellite, patch_id)
    if loaded is None:
        log.warning("Patch files not found for patch_id=%d", patch_id)
        for c in range(ncols):
            axes[row_idx, c].set_visible(False)
        return

    clean, mask = loaded
    for col_idx, method in enumerate(selected_methods):
        ax = axes[row_idx, col_idx]
        recon = _load_recon_array(results_dir, method, patch_id)
        if recon is None:
            ax.set_visible(False)
            continue

        error_map = _compute_error_map(clean, recon)
        if col_idx == 0:
            ax.imshow(error_map, cmap="hot")
            ax.set_title(f"{method}\n(MSE)", fontsize=FONT_SIZE - 1)
            ax.set_ylabel(f"{label}\npatch {patch_id}", fontsize=FONT_SIZE)
            ax.axis("off")
            continue

        ax.set_title(f"{method}\n(LISA)", fontsize=FONT_SIZE - 1)
        try:
            result = spatial_autocorrelation(error_map, mask)
            ax.imshow(result.lisa_labels, cmap=cmap, vmin=0, vmax=4)
        except Exception:
            log.warning("LISA failed for patch=%d method=%s", patch_id, method)
            ax.text(
                0.5,
                0.5,
                "LISA failed",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=FONT_SIZE,
            )
        ax.axis("off")


def _render_fig6_row(
    axes: np.ndarray,
    row_idx: int,
    ncols: int,
    preprocessed: Path,
    sat: str,
    patch_id: int,
    noise_level: str,
    results_dir: Path,
    top_methods: list[str],
    patch_metrics: pd.DataFrame,
    label: str,
) -> None:
    clean_path = preprocessed / "test" / sat / f"{patch_id:07d}_clean.npy"
    degraded_path = (
        preprocessed
        / "test"
        / sat
        / f"{patch_id:07d}_degraded_{noise_level}.npy"
    )
    mask_path = preprocessed / "test" / sat / f"{patch_id:07d}_mask.npy"

    if not clean_path.exists() or not degraded_path.exists():
        log.warning("Patch files not found for patch_id=%d", patch_id)
        for c in range(ncols):
            axes[row_idx, c].set_visible(False)
        return

    clean = np.load(clean_path)
    degraded = np.load(degraded_path)
    mask = np.load(mask_path) if mask_path.exists() else None
    if mask is not None and mask.ndim == 3:
        mask = mask[:, :, 0]

    axes[row_idx, 0].imshow(to_display_rgb(clean))
    axes[row_idx, 0].set_title("Clean", fontsize=FONT_SIZE)
    axes[row_idx, 0].axis("off")
    axes[row_idx, 0].set_ylabel(label, fontsize=FONT_SIZE)

    axes[row_idx, 1].imshow(to_display_rgb(degraded))
    axes[row_idx, 1].set_title("Degraded", fontsize=FONT_SIZE)
    axes[row_idx, 1].axis("off")

    if mask is not None:
        axes[row_idx, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    else:
        axes[row_idx, 2].text(
            0.5,
            0.5,
            "N/A",
            ha="center",
            va="center",
            transform=axes[row_idx, 2].transAxes,
        )
    axes[row_idx, 2].set_title("Mask", fontsize=FONT_SIZE)
    axes[row_idx, 2].axis("off")

    for offset, method in enumerate(top_methods):
        col = 3 + offset
        recon = _load_recon_array(results_dir, method, patch_id, noise_level)
        if recon is not None:
            axes[row_idx, col].imshow(to_display_rgb(recon))
            method_row = patch_metrics[patch_metrics["method"] == method]
            if not method_row.empty:
                psnr_val = float(method_row.iloc[0]["psnr"])
                title = f"{method}\n{psnr_val:.1f} dB"
            else:
                title = method
            axes[row_idx, col].set_title(title, fontsize=FONT_SIZE - 1)
        else:
            axes[row_idx, col].set_title(method, fontsize=FONT_SIZE - 1)
        axes[row_idx, col].axis("off")


def _load_recon_array(
    results_dir: Path,
    method: str,
    patch_id: int,
    noise_level: str = "inf",
) -> np.ndarray | None:
    """Load a reconstruction array, preferring raw .npy over PNG.

    Returns None if no reconstruction is found.
    """
    npy_path = (
        results_dir
        / "reconstruction_arrays"
        / noise_level
        / method
        / f"{patch_id:07d}.npy"
    )
    if npy_path.exists():
        return np.load(npy_path)
    png_path = (
        results_dir / "reconstruction_images" / method / f"{patch_id:07d}.png"
    )
    if png_path.exists():
        return plt.imread(str(png_path))[:, :, :3]
    return None


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure in configured formats (PNG always, PDF unless --png-only)."""
    fig.savefig(output_dir / f"{name}.png")
    if not _PNG_ONLY:
        fig.savefig(output_dir / f"{name}.pdf")


def _setup_style() -> None:
    plt.rcParams.update(STYLE_PARAMS)
    sns.set_palette("colorblind")


def fig1_entropy_examples(results_dir: Path, output_dir: Path) -> None:
    """Fig 1: Entropy map examples at 3 scales, 2 satellites."""
    from pdi_pipeline.entropy import shannon_entropy

    log.info("Figure 1: entropy map examples (computed on-the-fly)")

    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"

    satellites = ["sentinel2", "landsat8"]
    window_sizes = [7, 15, 31]

    # Find which satellites actually have data
    available_sats = [
        sat
        for sat in satellites
        if (preprocessed / "test" / sat).exists()
        and list((preprocessed / "test" / sat).glob("*_clean.npy"))
    ]

    if not available_sats:
        log.warning("No preprocessed test data found. Skipping fig1.")
        return

    nrows = len(available_sats)
    ncols = len(window_sizes)

    fig_width = 2.3 * ncols
    fig_height = 2.3 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Local Shannon Entropy at Multiple Scales")

    for row_idx, sat in enumerate(available_sats):
        clean_files = sorted((preprocessed / "test" / sat).glob("*_clean.npy"))
        if not clean_files:
            continue
        # Use first available patch
        clean = np.load(clean_files[0])

        for col_idx, ws in enumerate(window_sizes):
            ax = axes[row_idx, col_idx]
            ent = shannon_entropy(clean, ws)
            im = ax.imshow(ent, cmap="magma")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{sat} - {ws}x{ws}")
            ax.axis("off")

    _save_figure(fig, output_dir, "fig1_entropy_examples")
    plt.close(fig)
    log.info("Saved fig1_entropy_examples")


def fig2_entropy_vs_psnr(results_dir: Path, output_dir: Path) -> None:
    """Fig 2: Scatterplot of entropy vs PSNR per method."""
    df = load_results(results_dir)

    methods = sorted(df["method"].unique())
    n_methods = len(methods)
    ncols = min(5, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.5, nrows * 2.5),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes_flat[idx]
        mdf = df.loc[df["method"] == method, ["entropy_7", "psnr"]].dropna()
        if mdf.empty:
            ax.set_title(method)
            continue

        sns.scatterplot(
            data=mdf,
            x="entropy_7",
            y="psnr",
            s=1,
            alpha=0.3,
            rasterized=True,
            ax=ax,
        )
        ax.set_xlabel("Entropy (7x7)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(method)

    # Hide unused axes
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    _save_figure(fig, output_dir, "fig2_entropy_vs_psnr")
    plt.close(fig)
    log.info("Saved fig2_entropy_vs_psnr")


def fig3_psnr_by_entropy_bin(results_dir: Path, output_dir: Path) -> None:
    """Fig 3: Boxplot of PSNR per method grouped by entropy bin."""
    df = load_results(results_dir)

    # Add entropy bin column
    valid = df.loc[:, ["entropy_7", "psnr", "method"]].dropna()
    if valid.empty:
        log.warning("No valid data for fig3")
        return

    t1 = float(valid["entropy_7"].quantile(1 / 3))
    t2 = float(valid["entropy_7"].quantile(2 / 3))

    valid = valid.copy()
    valid["entropy_bin"] = pd.cut(
        valid["entropy_7"],
        bins=[-np.inf, t1, t2, np.inf],
        labels=["low", "medium", "high"],
        right=True,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    sns.boxplot(
        data=valid,
        x="method",
        y="psnr",
        hue="entropy_bin",
        hue_order=["low", "medium", "high"],
        ax=ax,
        fliersize=0.5,
        linewidth=0.5,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("PSNR (dB)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Entropy Bin", loc="upper right")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig3_psnr_by_entropy_bin")
    plt.close(fig)
    log.info("Saved fig3_psnr_by_entropy_bin")


def fig4_psnr_by_noise(results_dir: Path, output_dir: Path) -> None:
    """Fig 4: Boxplot of PSNR per method grouped by noise level."""
    df = load_results(results_dir)

    valid = df.dropna(subset=["psnr", "noise_level", "method"])
    if valid.empty:
        log.info("No valid data for fig4")
        return

    hue_order = [
        level
        for level in ["inf", "40", "30", "20"]
        if level in set(valid["noise_level"].astype(str))
    ]
    if not hue_order:
        log.info("No noise levels available for fig4")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    if len(hue_order) == 1:
        sns.boxplot(
            data=valid,
            x="method",
            y="psnr",
            ax=ax,
            fliersize=0.5,
            linewidth=0.5,
        )
    else:
        sns.boxplot(
            data=valid,
            x="method",
            y="psnr",
            hue="noise_level",
            hue_order=hue_order,
            ax=ax,
            fliersize=0.5,
            linewidth=0.5,
        )
    ax.set_xlabel("Method")
    ax.set_ylabel("PSNR (dB)")
    ax.tick_params(axis="x", rotation=45)
    if len(hue_order) > 1:
        ax.legend(title="Noise (dB)", loc="upper right")

    fig.tight_layout()
    _save_figure(fig, output_dir, "fig4_psnr_by_noise")
    plt.close(fig)
    log.info("Saved fig4_psnr_by_noise")


def fig5_lisa_clusters(results_dir: Path, output_dir: Path) -> None:
    """Fig 5: LISA cluster maps computed on-the-fly from reconstruction errors.

    Selects 3 representative patches at entropy percentiles P10, P50, P90
    and 2 methods (best and mid-range by mean PSNR). For each combination,
    computes the squared error map and runs spatial autocorrelation (LISA).
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    log.info("Figure 5: LISA clusters (on-the-fly computation)")

    df = load_results(results_dir)
    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"
    recon_dir = results_dir / "reconstruction_images"
    arrays_dir = results_dir / "reconstruction_arrays"

    if not recon_dir.exists() and not arrays_dir.exists():
        log.info("No reconstruction data found. Skipping fig5.")
        return

    valid = df.dropna(subset=["entropy_7", "psnr"])
    if valid.empty:
        log.info("No valid data for fig5")
        return

    # Constrain to patches that have reconstruction data
    avail_ids = _available_recon_patch_ids(recon_dir)
    valid = valid[valid["patch_id"].isin(avail_ids)]
    if valid.empty:
        log.info("No patches with reconstruction data for fig5")
        return

    percentiles = {"P10": 0.10, "P50": 0.50, "P90": 0.90}
    representatives = _pick_entropy_representatives(
        valid,
        percentiles,
        extra_col="satellite",
    )

    selected_methods = _choose_fig5_methods(valid)
    if selected_methods is None:
        log.warning("Need at least 2 methods for fig5")
        return

    # LISA label names: 0=NS, 1=HH, 2=LH, 3=LL, 4=HL
    lisa_names = ["NS", "HH", "LH", "LL", "HL"]
    lisa_colors = ["#d9d9d9", "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
    cmap = ListedColormap(lisa_colors)

    nrows = len(representatives)
    ncols = 1 + len(selected_methods)  # error_map(best) + LISA per method
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5), squeeze=False
    )

    for row_idx, (label, patch_id, satellite) in enumerate(representatives):
        _render_fig5_row(
            axes,
            row_idx,
            ncols,
            results_dir,
            preprocessed,
            patch_id,
            satellite,
            label,
            selected_methods,
            cmap,
        )

    # Legend
    legend_patches = [
        Patch(facecolor=c, label=n)
        for n, c in zip(lisa_names, lisa_colors, strict=True)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=FONT_SIZE - 1,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save_figure(fig, output_dir, "fig5_lisa_clusters")
    plt.close(fig)
    log.info("Saved fig5_lisa_clusters")


def fig6_visual_examples(results_dir: Path, output_dir: Path) -> None:
    """Fig 6: Visual reconstruction examples.

    Shows clean / degraded / mask / top-4 methods for 3 patches at
    entropy percentiles P10, P50, P90. Produces one composite figure
    per satellite.
    """
    df = load_results(results_dir)

    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"
    recon_dir = results_dir / "reconstruction_images"
    arrays_dir = results_dir / "reconstruction_arrays"

    if not recon_dir.exists() and not arrays_dir.exists():
        log.warning(
            "No reconstruction data found. "
            "Run experiment with --save-reconstructions first.",
        )
        return

    valid = df.dropna(subset=["entropy_7", "psnr"])
    if valid.empty:
        log.warning("No valid data for fig6")
        return

    # Constrain to patches that have reconstruction data
    avail_ids = _available_recon_patch_ids(recon_dir)
    valid_with_recon = valid[valid["patch_id"].isin(avail_ids)]

    # Top 4 methods by global mean PSNR (consistent across all rows)
    method_ranking = (
        valid.groupby("method")["psnr"].mean().sort_values(ascending=False)
    )
    top_methods = method_ranking.head(4).index.tolist()
    n_show = len(top_methods)

    if n_show == 0:
        log.warning("No methods with PSNR data for fig6")
        return

    satellites = sorted(valid["satellite"].unique())

    for sat in satellites:
        sat_df = valid_with_recon[valid_with_recon["satellite"] == sat]
        if sat_df.empty:
            log.warning(
                "No patches with reconstruction images for fig6 (%s)", sat
            )
            continue

        percentiles = {
            "P10 (low)": 0.10,
            "P50 (median)": 0.50,
            "P90 (high)": 0.90,
        }
        representatives = _pick_entropy_representatives(
            sat_df,
            percentiles,
            extra_col="noise_level",
        )

        # Columns: Clean | Degraded | Mask | Method1..MethodN
        ncols = 3 + n_show
        nrows = len(representatives)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 1.8, nrows * 2.0),
            squeeze=False,
            constrained_layout=True,
        )

        for row_idx, (label, patch_id, noise_level) in enumerate(
            representatives
        ):
            patch_metrics = valid[
                (valid["patch_id"] == patch_id)
                & (valid["noise_level"] == noise_level)
            ]

            _render_fig6_row(
                axes,
                row_idx,
                ncols,
                preprocessed,
                sat,
                patch_id,
                noise_level,
                results_dir,
                top_methods,
                patch_metrics,
                label,
            )

        _save_figure(fig, output_dir, f"fig6_visual_examples_{sat}")
        plt.close(fig)
        log.info("Saved fig6_visual_examples_%s", sat)

    log.info("Saved fig6_visual_examples (all satellites)")


def fig7_correlation_heatmap(results_dir: Path, output_dir: Path) -> None:
    """Fig 7: Correlation heatmap (method x entropy_window x metric)."""
    from pdi_pipeline.statistics import correlation_matrix

    df = load_results(results_dir)

    entropy_cols = [c for c in df.columns if c.startswith("entropy_")]
    metric_cols = ["psnr", "ssim", "rmse", "sam"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    if not entropy_cols or not metric_cols:
        log.warning("Missing entropy or metric columns for fig7")
        return

    corr_df = correlation_matrix(df, entropy_cols, metric_cols)
    if corr_df.empty:
        log.warning("Empty correlation matrix for fig7")
        return

    methods = sorted(corr_df["method"].unique())

    for mcol in metric_cols:
        subset = corr_df[corr_df["metric_col"] == mcol]
        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="method",
            columns="entropy_col",
            values="spearman_rho",
        )
        pivot = (
            pivot.reindex(index=methods)
            .dropna(how="all")
            .dropna(axis=1, how="all")
        )

        if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 1:
            log.warning(
                "Insufficient data for fig7 heatmap (%s): %s",
                mcol,
                pivot.shape,
            )
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            annot_kws={"size": 6},
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Spearman rho: Entropy vs {mcol.upper()}")
        ax.set_ylabel("")

        fig.tight_layout()
        _save_figure(fig, output_dir, f"fig7_corr_heatmap_{mcol}")
        plt.close(fig)

    log.info("Saved fig7_correlation_heatmap(s)")


def fig9_local_metric_maps(results_dir: Path, output_dir: Path) -> None:
    """Fig 9: Aggregated local PSNR and SSIM maps for a representative patch.

    Shows spatial distribution of reconstruction quality to support
    the LISA hotspot analysis (fig5).
    """
    from pdi_pipeline.metrics import local_psnr, local_ssim

    df = load_results(results_dir)
    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"
    recon_dir = results_dir / "reconstruction_images"
    arrays_dir = results_dir / "reconstruction_arrays"

    if not recon_dir.exists() and not arrays_dir.exists():
        log.info("No reconstruction data found. Skipping fig9.")
        return

    valid = df.dropna(subset=["entropy_7", "psnr"])
    if valid.empty:
        log.info("No valid data for fig9")
        return

    # Constrain to patches that have reconstruction data
    avail_ids = _available_recon_patch_ids(recon_dir)
    valid = valid[valid["patch_id"].isin(avail_ids)]
    if valid.empty:
        log.info("No patches with reconstruction data for fig9")
        return

    # Pick median-entropy patch
    median_ent = float(valid["entropy_7"].median())
    ref_row = valid.iloc[
        (valid["entropy_7"] - median_ent).abs().argsort()[:1]
    ].iloc[0]
    patch_id = int(ref_row["patch_id"])
    satellite = ref_row["satellite"]

    clean_path = preprocessed / "test" / satellite / f"{patch_id:07d}_clean.npy"
    mask_path = preprocessed / "test" / satellite / f"{patch_id:07d}_mask.npy"

    if not clean_path.exists() or not mask_path.exists():
        log.info("Patch files not found for patch_id=%d", patch_id)
        return

    clean = np.load(clean_path)
    mask = np.load(mask_path)

    # Pick top 3 methods by PSNR for this patch
    patch_df = valid[valid["patch_id"] == patch_id]
    top_methods = (
        patch_df.groupby("method")["psnr"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    if not top_methods:
        log.info("No method data for patch_id=%d", patch_id)
        return

    nrows = len(top_methods)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(5.0, nrows * 2.5),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, method in enumerate(top_methods):
        recon = _load_recon_array(results_dir, method, patch_id)
        if recon is None:
            axes[row_idx, 0].set_visible(False)
            axes[row_idx, 1].set_visible(False)
            continue

        clean_eval = clean
        recon_eval = recon
        if (
            clean_eval.ndim == 3
            and recon_eval.ndim == 3
            and clean_eval.shape[:2] == recon_eval.shape[:2]
            and clean_eval.shape[2] != recon_eval.shape[2]
        ):
            common_channels = min(clean_eval.shape[2], recon_eval.shape[2])
            clean_eval = clean_eval[:, :, :common_channels]
            recon_eval = recon_eval[:, :, :common_channels]

        if clean_eval.shape != recon_eval.shape:
            log.warning(
                (
                    "Skipping fig9 method=%s patch=%d due shape mismatch: "
                    "clean=%s recon=%s"
                ),
                method,
                patch_id,
                clean_eval.shape,
                recon_eval.shape,
            )
            axes[row_idx, 0].set_visible(False)
            axes[row_idx, 1].set_visible(False)
            continue

        # Use raw arrays for local metrics (not normalized PNGs)
        lpsnr = local_psnr(clean_eval, recon_eval, mask, window=15)
        lssim = local_ssim(clean_eval, recon_eval, mask, window=15)

        im0 = axes[row_idx, 0].imshow(lpsnr, cmap="RdYlGn")
        axes[row_idx, 0].set_title(f"{method} - Local PSNR")
        axes[row_idx, 0].axis("off")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        im1 = axes[row_idx, 1].imshow(lssim, cmap="RdYlGn", vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"{method} - Local SSIM")
        axes[row_idx, 1].axis("off")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Local Quality Maps (patch {patch_id}, {satellite})",
        fontsize=FONT_SIZE + 1,
    )
    _save_figure(fig, output_dir, "fig9_local_metric_maps")
    plt.close(fig)
    log.info("Saved fig9_local_metric_maps")


ALL_FIGURES = {
    1: fig1_entropy_examples,
    2: fig2_entropy_vs_psnr,
    3: fig3_psnr_by_entropy_bin,
    4: fig4_psnr_by_noise,
    5: fig5_lisa_clusters,
    6: fig6_visual_examples,
    7: fig7_correlation_heatmap,
    9: fig9_local_metric_maps,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to experiment results directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for figures. Default: results_dir/figures/",
    )
    parser.add_argument(
        "--figure",
        type=int,
        default=None,
        help="Generate only this figure number (1-8).",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Save figures in PNG only (skip PDF). Use for quick validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global _PNG_ONLY
    args = parse_args(argv)
    _PNG_ONLY = args.png_only
    _setup_style()

    results_dir = args.results
    output_dir = args.output or results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(results_dir, name="figures")

    if args.figure is not None:
        if args.figure not in ALL_FIGURES:
            log.error("Invalid figure number: %d", args.figure)
            return
        ALL_FIGURES[args.figure](results_dir, output_dir)
    else:
        results_dir.parent / "dl_eval"
        for num, func in ALL_FIGURES.items():
            try:
                func(results_dir, output_dir)
            except Exception:
                log.exception("Error generating figure %d", num)

    log.info("Figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
