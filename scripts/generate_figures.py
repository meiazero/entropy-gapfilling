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
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

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

# Shared boxplot keyword arguments (reference: fig3)
_BOXPLOT_KW: dict[str, object] = {
    "fliersize": 6,
    "linewidth": 1.2,
    "showmeans": True,
    "boxprops": {
        "edgecolor": "#333333",
        "linewidth": 1.2,
    },
    "whiskerprops": {
        "color": "#333333",
        "linewidth": 1.0,
    },
    "capprops": {
        "color": "#333333",
        "linewidth": 1.0,
    },
    "medianprops": {
        "color": "#ff7f0e",
        "linewidth": 1.0,
    },
    "meanprops": {
        "marker": "x",
        "markeredgecolor": "#333333",
        "markersize": 5,
    },
    "flierprops": {
        "markeredgecolor": "#d62728",
        "markerfacecolor": "#d62728",
        "markersize": 6,
    },
}

# Shared legend keyword arguments
_LEGEND_KW: dict[str, object] = {
    "loc": "best",
    "frameon": True,
    "framealpha": 0.85,
    "fontsize": FONT_SIZE - 1,
}

# Shared scatter keyword arguments
_SCATTER_KW: dict[str, object] = {
    "s": 12,
    "alpha": 0.7,
    "edgecolor": "white",
    "linewidth": 0.3,
    "rasterized": True,
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STYLE_PATH = _PROJECT_ROOT / "images" / "style.mplstyle"

# Publication font-size overrides applied on top of the .mplstyle file
_PUB_FONT_OVERRIDES = {
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

_PATCH_SPLITS: dict[int, str] = {}


def _load_patch_splits() -> dict[int, str]:
    """Load patch_id -> split mapping from the manifest (cached)."""
    if _PATCH_SPLITS:
        return _PATCH_SPLITS

    manifest_path = _PROJECT_ROOT / "preprocessed" / "manifest.csv"
    if not manifest_path.exists():
        return _PATCH_SPLITS

    df = pd.read_csv(manifest_path, usecols=["patch_id", "split"])
    for _, row in df.iterrows():
        _PATCH_SPLITS[int(row["patch_id"])] = str(row["split"])
    return _PATCH_SPLITS


def _patch_split(patch_id: int) -> str:
    """Return the split for a given patch_id, defaulting to 'train'."""
    splits = _load_patch_splits()
    return splits.get(patch_id, "train")


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


def _has_preprocessed_clean(
    preprocessed: Path,
    satellite: str,
    patch_id: int,
) -> bool:
    """Check whether the preprocessed clean/mask files exist for a patch."""
    split = _patch_split(patch_id)
    base = preprocessed / split / satellite
    clean = base / f"{patch_id:07d}_clean.npy"
    mask = base / f"{patch_id:07d}_mask.npy"
    return clean.exists() and mask.exists()


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
    split = _patch_split(patch_id)
    clean_path = preprocessed / split / satellite / f"{patch_id:07d}_clean.npy"
    mask_path = preprocessed / split / satellite / f"{patch_id:07d}_mask.npy"
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
            ax.set_title(f"{method}\n(EQM)", fontsize=FONT_SIZE - 1)
            ax.set_ylabel(f"{label}\nrecorte {patch_id}", fontsize=FONT_SIZE)
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
                "LISA falhou",
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
    split = _patch_split(patch_id)
    clean_path = preprocessed / split / sat / f"{patch_id:07d}_clean.npy"
    degraded_path = (
        preprocessed
        / split
        / sat
        / f"{patch_id:07d}_degraded_{noise_level}.npy"
    )
    mask_path = preprocessed / split / sat / f"{patch_id:07d}_mask.npy"

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
    axes[row_idx, 0].set_title("Limpa", fontsize=FONT_SIZE)
    axes[row_idx, 0].axis("off")
    axes[row_idx, 0].set_ylabel(label, fontsize=FONT_SIZE)

    axes[row_idx, 1].imshow(to_display_rgb(degraded))
    axes[row_idx, 1].set_title("Degradada", fontsize=FONT_SIZE)
    axes[row_idx, 1].axis("off")

    if mask is not None:
        axes[row_idx, 2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    else:
        axes[row_idx, 2].text(
            0.5,
            0.5,
            "N/D",
            ha="center",
            va="center",
            transform=axes[row_idx, 2].transAxes,
        )
    axes[row_idx, 2].set_title("Máscara", fontsize=FONT_SIZE)
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
    fig.savefig(output_dir / f"{name}.png", dpi=DPI, bbox_inches="tight")
    if not _SETTINGS["png_only"]:
        fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")


def _choose_entropy_column(df: pd.DataFrame) -> tuple[str, int] | None:
    if "entropy_7" in df.columns:
        return "entropy_7", 7
    if "entropy_15" in df.columns:
        return "entropy_15", 15

    entropy_cols = sorted(c for c in df.columns if c.startswith("entropy_"))
    if not entropy_cols:
        return None

    entropy_col = entropy_cols[0]
    try:
        window_size = int(entropy_col.split("_")[-1])
    except ValueError:
        window_size = 0
    return entropy_col, window_size


def _scatter_grid(
    n_items: int, min_cols: int = 2, max_cols: int = 4
) -> tuple[int, int]:
    if n_items <= 0:
        return 1, 1
    cols = min(max_cols, max(min_cols, math.ceil(math.sqrt(n_items))))
    rows = math.ceil(n_items / cols)
    if rows / cols > 1.35 and cols < max_cols:
        cols += 1
        rows = math.ceil(n_items / cols)
    return rows, cols


def _add_entropy_psnr_fit(
    ax: Axes, mdf: pd.DataFrame, entropy_col: str
) -> None:
    if len(mdf) < 3:
        return

    x_vals = mdf[entropy_col].to_numpy()
    y_vals = mdf["psnr"].to_numpy()

    beta, alpha = np.polyfit(x_vals, y_vals, 1)
    y_pred = beta * x_vals + alpha
    ss_res = float(np.sum((y_vals - y_pred) ** 2))
    ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    n = len(y_vals)
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - 2) if n > 2 else r2

    sns.regplot(
        data=mdf,
        x=entropy_col,
        y="psnr",
        scatter=False,
        ci=None,
        ax=ax,
        color="#555555",
        line_kws={"linewidth": 1.0, "linestyle": "--", "label": "Ajuste"},
    )

    stats_text = (
        f"y = {alpha:.2f} + {beta:.2f}x\n"
        f"$r^2$ = {r2:.3f}\n"
        f"$r^2_{{adj}}$ = {r2_adj:.3f}\n"
        f"$\\beta$ = {beta:.2f}"
    )
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=FONT_SIZE - 2,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.75,
        },
    )


def _setup_style() -> None:
    if _STYLE_PATH.exists():
        plt.style.use(str(_STYLE_PATH))
    plt.rcParams.update(_PUB_FONT_OVERRIDES)
    sns.set_palette("Set2")


def fig1_entropy_examples(results_dir: Path, output_dir: Path) -> None:
    """Fig 1: Entropy map examples at 3 scales, 2 satellites."""
    from pdi_pipeline.entropy import shannon_entropy

    log.info("Figure 1: entropy map examples (computed on-the-fly)")

    preprocessed = _PROJECT_ROOT / "preprocessed"

    satellites = ["sentinel2", "landsat8"]
    window_sizes = [7, 15, 31]

    # Find which satellites actually have data (check all splits)
    _splits = ("test", "val", "train")
    available_sats: list[str] = []
    _sat_split: dict[str, str] = {}
    for sat in satellites:
        for split in _splits:
            sat_dir = preprocessed / split / sat
            if sat_dir.exists() and list(sat_dir.glob("*_clean.npy")):
                available_sats.append(sat)
                _sat_split[sat] = split
                break

    if not available_sats:
        log.warning("No preprocessed data found. Skipping fig1.")
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
    )

    for row_idx, sat in enumerate(available_sats):
        clean_files = sorted(
            (preprocessed / _sat_split[sat] / sat).glob("*_clean.npy")
        )
        if not clean_files:
            continue
        # Use first available patch
        clean = np.load(clean_files[0])

        for col_idx, ws in enumerate(window_sizes):
            ax = axes[row_idx, col_idx]
            ent = shannon_entropy(clean, ws)
            im = ax.imshow(ent, cmap="RdYlBu_r")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{sat} - {ws}x{ws}")
            if col_idx == 0:
                ax.set_ylabel("Entropia")
            ax.axis("off")

    plt.tight_layout()
    _save_figure(fig, output_dir, "fig1_entropy_examples")
    plt.close(fig)
    log.info("Saved fig1_entropy_examples")


def fig2_entropy_vs_psnr(results_dir: Path, output_dir: Path) -> None:
    """Fig 2: Scatterplot of entropy vs PSNR per method."""
    df = load_results(results_dir)

    chosen = _choose_entropy_column(df)
    if chosen is None:
        log.warning("No entropy columns for fig2")
        return
    entropy_col, ws = chosen

    methods = sorted(df["method"].unique())
    n_methods = len(methods)

    nrows, ncols = _scatter_grid(n_methods)

    palette = sns.color_palette("Set2", n_methods)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.2, nrows * 1.6),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes_flat[idx]
        mdf = df.loc[df["method"] == method, [entropy_col, "psnr"]].dropna()
        if mdf.empty:
            ax.set_title(method)
            continue

        sns.scatterplot(
            data=mdf,
            x=entropy_col,
            y="psnr",
            color=palette[idx % len(palette)],
            ax=ax,
            label="Dados",
            **_SCATTER_KW,
        )

        _add_entropy_psnr_fit(ax, mdf, entropy_col)

        ax.set_xlabel(f"Entropia ({ws}x{ws})")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(method)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.85,
            fontsize=FONT_SIZE - 2,
        )

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
        labels=["baixa", "média", "alta"],
        right=True,
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    sns.boxplot(
        data=valid,
        x="method",
        y="psnr",
        hue="entropy_bin",
        hue_order=["baixa", "média", "alta"],
        palette="Set2",
        ax=ax,
        **_BOXPLOT_KW,
    )
    ax.set_xlabel("Método")
    ax.set_ylabel("PSNR (dB)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Faixa de Entropia", **_LEGEND_KW)

    plt.tight_layout()
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
            hue="method",
            palette="Set2",
            legend=False,
            ax=ax,
            **_BOXPLOT_KW,
        )
    else:
        sns.boxplot(
            data=valid,
            x="method",
            y="psnr",
            hue="noise_level",
            hue_order=hue_order,
            palette="Set2",
            ax=ax,
            **_BOXPLOT_KW,
        )
    ax.set_xlabel("Método")
    ax.set_ylabel("PSNR (dB)")
    ax.tick_params(axis="x", rotation=45)
    if len(hue_order) > 1:
        ax.legend(title="Ruído (dB)", **_LEGEND_KW)

    plt.tight_layout()
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
    preprocessed = _PROJECT_ROOT / "preprocessed"
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

    # Constrain to patches that have preprocessed clean/mask files
    valid = valid[
        valid.apply(
            lambda r: _has_preprocessed_clean(
                preprocessed, str(r["satellite"]), int(r["patch_id"])
            ),
            axis=1,
        )
    ]
    if valid.empty:
        log.info("No patches with preprocessed files for fig5")
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
        nrows,
        ncols,
        figsize=(ncols * 2.5, nrows * 2.5),
        squeeze=False,
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
        frameon=_LEGEND_KW["frameon"],
        framealpha=_LEGEND_KW["framealpha"],
        fontsize=_LEGEND_KW["fontsize"],
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
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

    preprocessed = _PROJECT_ROOT / "preprocessed"
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

    satellites = [
        s for s in sorted(valid["satellite"].unique()) if s == "sentinel2"
    ]
    if not satellites:
        log.warning("No Sentinel-2 data available for fig6")
        return

    for sat in satellites:
        sat_df = valid_with_recon[valid_with_recon["satellite"] == sat]
        if sat_df.empty:
            log.warning(
                "No patches with reconstruction images for fig6 (%s)", sat
            )
            continue

        # Constrain to patches that have preprocessed clean/mask files
        sat_df = sat_df[
            sat_df["patch_id"].apply(
                lambda pid, sat=sat: _has_preprocessed_clean(
                    preprocessed,
                    sat,
                    int(pid),
                )
            )
        ]
        if sat_df.empty:
            log.warning("No patches with preprocessed files for fig6 (%s)", sat)
            continue

        percentiles = {
            "P10 (baixa)": 0.10,
            "P50 (mediana)": 0.50,
            "P90 (alta)": 0.90,
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

        plt.tight_layout()
        _save_figure(fig, output_dir, f"fig6_visual_examples_{sat}")
        plt.close(fig)
        log.info("Saved fig6_visual_examples_%s", sat)

    log.info("Saved fig6_visual_examples (all satellites)")


def fig7_correlation_heatmap(results_dir: Path, output_dir: Path) -> None:
    """Fig 7: Correlation heatmap (method x entropy_window x metric)."""
    from pdi_pipeline.statistics import correlation_matrix

    df = load_results(results_dir)

    entropy_cols = [c for c in df.columns if c.startswith("entropy_")]
    metric_cols = [c for c in ["psnr"] if c in df.columns]

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
            pivot
            .reindex(index=methods)
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
            cmap="RdYlBu_r",
            center=0,
            ax=ax,
            annot_kws={"size": FONT_SIZE - 2},
            vmin=-1,
            vmax=1,
        )
        ax.set_title(f"Spearman $\\rho$: Entropia vs {mcol.upper()}")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

        plt.tight_layout()
        _save_figure(fig, output_dir, f"fig7_corr_heatmap_{mcol}")
        plt.close(fig)

    log.info("Saved fig7_correlation_heatmap(s)")


ALL_FIGURES = {
    1: fig1_entropy_examples,
    2: fig2_entropy_vs_psnr,
    3: fig3_psnr_by_entropy_bin,
    4: fig4_psnr_by_noise,
    5: fig5_lisa_clusters,
    6: fig6_visual_examples,
    7: fig7_correlation_heatmap,
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
        help="Generate only this figure number (1-7).",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Save figures in PNG only (skip PDF). Use for quick validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _SETTINGS["png_only"] = args.png_only
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
        for num, func in ALL_FIGURES.items():
            try:
                func(results_dir, output_dir)
            except Exception:
                log.exception("Error generating figure %d", num)

    log.info("Figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
