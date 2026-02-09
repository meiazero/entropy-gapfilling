"""Generate publication-quality figures from experiment results.

Produces 9 figure types for the journal paper. All figures use
matplotlib + seaborn with consistent styling.

Usage:
    uv run python scripts/generate_figures.py --results results/paper_results
    uv run python scripts/generate_figures.py --results results/paper_results --figure 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pdi_pipeline.aggregation import (
    load_results,
)

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
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


def _setup_style() -> None:
    plt.rcParams.update(STYLE_PARAMS)
    sns.set_palette("colorblind")


def fig1_entropy_examples(results_dir: Path, output_dir: Path) -> None:
    """Fig 1: Entropy map examples at 3 scales, 2 satellites."""
    log.info("Figure 1: entropy map examples (requires preprocessed data)")
    # This figure requires loading actual patches and entropy maps
    # Placeholder structure - needs preprocessed data to render
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID)
    fig.suptitle("Local Shannon Entropy at Multiple Scales")

    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"

    for row_idx, sat in enumerate(["sentinel2", "landsat8"]):
        for col_idx, ws in enumerate([7, 15, 31]):
            ax = axes[row_idx, col_idx]
            # Search for a test patch with entropy
            pattern = f"test/{sat}/*_entropy_{ws}.npy"
            import glob

            files = sorted(glob.glob(str(preprocessed / pattern)))
            if files:
                ent = np.load(files[0])
                im = ax.imshow(ent, cmap="viridis")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{sat} - {ws}x{ws}")
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_entropy_examples.pdf")
    fig.savefig(output_dir / "fig1_entropy_examples.png")
    plt.close(fig)
    log.info("Saved fig1_entropy_examples")


def fig2_entropy_vs_psnr(results_dir: Path, output_dir: Path) -> None:
    """Fig 2: Scatterplot of entropy vs PSNR per method."""
    df = load_results(results_dir)

    methods = sorted(df["method"].unique())
    n_methods = len(methods)
    ncols = min(5, n_methods)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes_flat = np.array(axes).flatten() if n_methods > 1 else [axes]

    for idx, method in enumerate(methods):
        ax = axes_flat[idx]
        mdf = df[df["method"] == method].dropna(subset=["entropy_7", "psnr"])
        if mdf.empty:
            ax.set_title(method)
            continue

        ax.scatter(
            mdf["entropy_7"],
            mdf["psnr"],
            s=1,
            alpha=0.3,
            rasterized=True,
        )
        ax.set_xlabel("Entropy (7x7)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(method)

    # Hide unused axes
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_entropy_vs_psnr.pdf")
    fig.savefig(output_dir / "fig2_entropy_vs_psnr.png")
    plt.close(fig)
    log.info("Saved fig2_entropy_vs_psnr")


def fig3_psnr_by_entropy_bin(results_dir: Path, output_dir: Path) -> None:
    """Fig 3: Boxplot of PSNR per method grouped by entropy bin."""
    df = load_results(results_dir)

    # Add entropy bin column
    valid = df.dropna(subset=["entropy_7", "psnr"])
    if valid.empty:
        log.warning("No valid data for fig3")
        return

    t1 = float(valid["entropy_7"].quantile(1 / 3))
    t2 = float(valid["entropy_7"].quantile(2 / 3))

    def _bin(v: float) -> str:
        if v <= t1:
            return "low"
        if v <= t2:
            return "medium"
        return "high"

    valid = valid.copy()
    valid["entropy_bin"] = valid["entropy_7"].apply(_bin)

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
    fig.savefig(output_dir / "fig3_psnr_by_entropy_bin.pdf")
    fig.savefig(output_dir / "fig3_psnr_by_entropy_bin.png")
    plt.close(fig)
    log.info("Saved fig3_psnr_by_entropy_bin")


def fig4_psnr_by_noise(results_dir: Path, output_dir: Path) -> None:
    """Fig 4: Boxplot of PSNR per method grouped by noise level."""
    df = load_results(results_dir)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    sns.boxplot(
        data=df,
        x="method",
        y="psnr",
        hue="noise_level",
        hue_order=["inf", "40", "30", "20"],
        ax=ax,
        fliersize=0.5,
        linewidth=0.5,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("PSNR (dB)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Noise (dB)", loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_psnr_by_noise.pdf")
    fig.savefig(output_dir / "fig4_psnr_by_noise.png")
    plt.close(fig)
    log.info("Saved fig4_psnr_by_noise")


def fig5_lisa_clusters(results_dir: Path, output_dir: Path) -> None:
    """Fig 5: LISA cluster maps overlaid on error maps.

    Requires running spatial_autocorrelation on individual patches.
    This is a placeholder that generates from precomputed LISA results
    if available.
    """
    log.info("Figure 5: LISA clusters (requires spatial analysis results)")
    lisa_dir = results_dir / "lisa_maps"
    if not lisa_dir.exists():
        log.warning(
            "LISA maps not found at %s. Run spatial analysis first.",
            lisa_dir,
        )
        return

    import glob

    files = sorted(glob.glob(str(lisa_dir / "*.npy")))[:6]
    if not files:
        log.warning("No LISA map files found")
        return

    n = len(files)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes_flat = np.array(axes).flatten()

    cmap = plt.cm.get_cmap("RdYlBu", 5)

    for idx, fpath in enumerate(files):
        ax = axes_flat[idx]
        lisa = np.load(fpath)
        ax.imshow(lisa, cmap=cmap, vmin=0, vmax=4)
        ax.axis("off")
        ax.set_title(Path(fpath).stem, fontsize=6)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig5_lisa_clusters.pdf")
    fig.savefig(output_dir / "fig5_lisa_clusters.png")
    plt.close(fig)
    log.info("Saved fig5_lisa_clusters")


def fig6_visual_examples(results_dir: Path, output_dir: Path) -> None:
    """Fig 6: Visual reconstruction examples.

    Shows clean / degraded / top-4 classical methods for two patches
    at low (10th percentile) and high (90th percentile) entropy.
    Requires preprocessed patches and reconstruction outputs.
    """
    df = load_results(results_dir)

    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"
    recon_dir = results_dir / "reconstructions"

    if not recon_dir.exists():
        log.warning(
            "Reconstructions not found at %s. "
            "Run experiment with --save-reconstructions first.",
            recon_dir,
        )
        return

    # Find representative patches at 10th and 90th entropy percentile
    valid = df.dropna(subset=["entropy_7", "psnr"])
    if valid.empty:
        log.warning("No valid data for fig6")
        return

    p10 = float(valid["entropy_7"].quantile(0.10))
    p90 = float(valid["entropy_7"].quantile(0.90))

    low_ent = valid.iloc[(valid["entropy_7"] - p10).abs().argsort()[:1]]
    high_ent = valid.iloc[(valid["entropy_7"] - p90).abs().argsort()[:1]]

    representative = [
        ("Low Entropy", low_ent.iloc[0]),
        ("High Entropy", high_ent.iloc[0]),
    ]

    available_methods = sorted(recon_dir.iterdir())
    method_names = [m.name for m in available_methods]

    n_show = min(4, len(method_names))
    ncols = 2 + n_show
    nrows = len(representative)

    if ncols < 3:
        log.warning("Not enough reconstructions for fig6")
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (label, ref_row) in enumerate(representative):
        patch_id = int(ref_row["patch_id"])
        satellite = ref_row["satellite"]
        noise_level = ref_row.get("noise_level", "inf")

        # Load clean and degraded
        clean_path = (
            preprocessed / "test" / satellite / f"{patch_id:07d}_clean.npy"
        )
        degraded_path = (
            preprocessed
            / "test"
            / satellite
            / f"{patch_id:07d}_degraded_{noise_level}.npy"
        )

        if not clean_path.exists() or not degraded_path.exists():
            log.warning("Patch files not found for patch_id=%d", patch_id)
            continue

        clean = np.load(clean_path)
        degraded = np.load(degraded_path)

        def _to_display(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            vmin, vmax = float(arr.min()), float(arr.max())
            if vmax - vmin < 1e-8:
                return np.zeros_like(arr)
            return np.clip((arr - vmin) / (vmax - vmin), 0, 1)

        # Rank methods by PSNR for this patch
        patch_df = valid[
            (valid["patch_id"] == patch_id)
            & (valid["noise_level"] == noise_level)
        ]
        ranked = patch_df.sort_values("psnr", ascending=False)

        col = 0
        axes[row_idx, col].imshow(_to_display(clean))
        axes[row_idx, col].set_title("Clean")
        axes[row_idx, col].axis("off")
        if col == 0:
            axes[row_idx, col].set_ylabel(label, fontsize=FONT_SIZE + 1)

        col = 1
        axes[row_idx, col].imshow(_to_display(degraded))
        axes[row_idx, col].set_title("Degraded")
        axes[row_idx, col].axis("off")

        # Top-N methods
        for k in range(n_show):
            col = 2 + k
            if k < len(ranked):
                mname = ranked.iloc[k]["method"]
                recon_path = recon_dir / mname / f"{patch_id:07d}.npy"
                if recon_path.exists():
                    recon = np.load(recon_path)
                    axes[row_idx, col].imshow(_to_display(recon))
                    psnr_val = ranked.iloc[k]["psnr"]
                    axes[row_idx, col].set_title(
                        f"{mname}\n{psnr_val:.1f} dB", fontsize=FONT_SIZE - 1
                    )
                else:
                    axes[row_idx, col].set_title(mname)
            axes[row_idx, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "fig6_visual_examples.pdf")
    fig.savefig(output_dir / "fig6_visual_examples.png")
    plt.close(fig)
    log.info("Saved fig6_visual_examples")


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
        fig.savefig(output_dir / f"fig7_corr_heatmap_{mcol}.pdf")
        fig.savefig(output_dir / f"fig7_corr_heatmap_{mcol}.png")
        plt.close(fig)

    log.info("Saved fig7_correlation_heatmap(s)")


def fig8_dl_vs_classical(results_dir: Path, output_dir: Path) -> None:
    """Fig 8: Bar chart comparing classical top-5 vs DL models."""
    import pandas as pd

    df_classical = load_results(results_dir)

    # Top 5 classical methods by mean PSNR
    classical_means = (
        df_classical.groupby("method")["psnr"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    # Load DL results
    dl_base = results_dir.parent / "dl_eval"
    dl_means: dict[str, float] = {}
    if dl_base.exists():
        for model_dir in sorted(dl_base.iterdir()):
            csv = model_dir / "results.csv"
            if csv.exists():
                dl_df = pd.read_csv(csv)
                ok = dl_df[dl_df["status"] == "ok"]
                if not ok.empty:
                    dl_means[model_dir.name] = float(ok["psnr"].mean())

    if not dl_means:
        log.warning(
            "No DL results found at %s. Skipping fig8.",
            dl_base,
        )
        return

    # Build combined bar data
    names = list(classical_means.index) + list(dl_means.keys())
    values = list(classical_means.values) + list(dl_means.values())
    colors = ["#4878CF"] * len(classical_means) + ["#E24A33"] * len(dl_means)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    bars = ax.barh(range(len(names)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean PSNR (dB)")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#4878CF", label="Classical (top 5)"),
        Patch(facecolor="#E24A33", label="Deep Learning"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=FONT_SIZE - 1,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "fig8_dl_vs_classical.pdf")
    fig.savefig(output_dir / "fig8_dl_vs_classical.png")
    plt.close(fig)
    log.info("Saved fig8_dl_vs_classical")


def fig9_local_metric_maps(results_dir: Path, output_dir: Path) -> None:
    """Fig 9: Aggregated local PSNR and SSIM maps for a representative patch.

    Shows spatial distribution of reconstruction quality to support
    the LISA hotspot analysis (fig5).
    """
    from pdi_pipeline.metrics import local_psnr, local_ssim

    df = load_results(results_dir)
    project_root = Path(__file__).resolve().parent.parent
    preprocessed = project_root / "preprocessed"
    recon_dir = results_dir / "reconstructions"

    if not recon_dir.exists():
        log.warning(
            "Reconstructions not found at %s. Skipping fig9.",
            recon_dir,
        )
        return

    valid = df.dropna(subset=["entropy_7", "psnr"])
    if valid.empty:
        log.warning("No valid data for fig9")
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
        log.warning("Patch files not found for patch_id=%d", patch_id)
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
        log.warning("No method data for patch_id=%d", patch_id)
        return

    nrows = len(top_methods)
    fig, axes = plt.subplots(nrows, 2, figsize=(5.0, nrows * 2.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, method in enumerate(top_methods):
        recon_path = recon_dir / method / f"{patch_id:07d}.npy"
        if not recon_path.exists():
            axes[row_idx, 0].set_visible(False)
            axes[row_idx, 1].set_visible(False)
            continue

        recon = np.load(recon_path)
        lpsnr = local_psnr(clean, recon, mask, window=15)
        lssim = local_ssim(clean, recon, mask, window=15)

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
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_local_metric_maps.pdf")
    fig.savefig(output_dir / "fig9_local_metric_maps.png")
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
    8: fig8_dl_vs_classical,
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
    return parser.parse_args(argv)


def _setup_file_logging(log_path: Path) -> None:
    """Add a file handler to the root logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _setup_style()

    results_dir = args.results
    output_dir = args.output or results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logging(results_dir / "figures.log")

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
