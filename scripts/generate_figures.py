"""Generate publication-quality figures from experiment results.

Produces 7 figure types for the journal paper. All figures use
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
    ncols = min(4, n_methods)
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

    Shows clean / degraded / top-3 methods / worst method side by side.
    Requires preprocessed patches and reconstruction outputs.
    """
    log.info(
        "Figure 6: visual examples (placeholder - needs reconstruction outputs)"
    )


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
        pivot = pivot.reindex(index=methods)

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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _setup_style()

    results_dir = args.results
    output_dir = args.output or results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

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
