"""Generate or copy additional figures required by docs/main.tex.

This repository has two workflows:
- `make paper`: generates assets into paper_assets/ and copies a selected
  subset into docs/.
- `make paper-only`: compiles the paper assuming docs/ already contains all
  referenced tables/figures.

Some figures referenced by the LaTeX sources are not part of the default
selection in scripts/select_paper_figures.py. This script fills that gap by
copying existing generated figures under the expected filenames and by
generating a small number of lightweight plots/montages from existing
paper_assets artifacts.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pdi_pipeline.statistics import spatial_autocorrelation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _normalize_noise(noise: object) -> str:
    s = str(noise)
    if s == "inf":
        return "inf"
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _noise_to_suffix(noise: str) -> str:
    """Match figure naming convention in scripts/generate_figures.py."""
    return noise.replace("inf", "gap_only")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        log.warning("Missing source figure: %s", src)
        return
    shutil.copy2(src, dst)
    log.info("Copied %s -> %s", src.name, dst.name)


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def copy_entropy_window_panels(
    *,
    best_noise: str = "40",
    output_dir: Path,
) -> None:
    """Provide the three-panel entropy-vs-PSNR figure filenames.

    The current LaTeX sources expect:
      - figures/fig2_entropy_vs_psnr_7x7.png
      - figures/fig2_entropy_vs_psnr_15x15.png
      - figures/fig2_entropy_vs_psnr_31x31.png

    The generated assets use fig6_psnr_entropy_{noise}_e{ws}.png.
    """
    figures_dir = PROJECT_ROOT / "paper_assets" / "figures"
    out_dir = output_dir

    suffix = _noise_to_suffix(_normalize_noise(best_noise))

    mapping = {
        figures_dir / f"fig6_psnr_entropy_{suffix}_e7.png": (
            out_dir / "fig2_entropy_vs_psnr_7x7.png"
        ),
        figures_dir / f"fig6_psnr_entropy_{suffix}_e15.png": (
            out_dir / "fig2_entropy_vs_psnr_15x15.png"
        ),
        figures_dir / f"fig6_psnr_entropy_{suffix}_e31.png": (
            out_dir / "fig2_entropy_vs_psnr_31x31.png"
        ),
    }
    for src, dst in mapping.items():
        _copy(src, dst)


def generate_psnr_by_noise(*, output_dir: Path) -> None:
    """Create a compact plot of PSNR vs noise level (classic results)."""
    csv_path = (
        PROJECT_ROOT
        / "paper_assets"
        / "classic"
        / "full_results"
        / "raw_results.csv"
    )
    if not csv_path.exists():
        log.warning("Missing classic raw results: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["noise_level"] = df["noise_level"].apply(_normalize_noise)

    order = ["inf", "40", "30", "20"]
    rows = []
    for noise in [n for n in order if n in set(df["noise_level"])]:
        vals = df[df["noise_level"] == noise]["psnr"].dropna().to_numpy()
        if vals.size == 0:
            continue
        rows.append({
            "noise": noise,
            "median": float(np.median(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
        })
    if not rows:
        log.warning("No PSNR rows for noise plot")
        return

    plot_df = pd.DataFrame(rows)
    x = np.arange(len(plot_df))
    y = plot_df["median"].to_numpy()
    yerr = np.vstack([
        y - plot_df["q25"].to_numpy(),
        plot_df["q75"].to_numpy() - y,
    ])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        linewidth=1.2,
        markersize=4,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([
        "Sem ruído" if n == "inf" else f"{n} dB" for n in plot_df["noise"]
    ])
    ax.set_xlabel("Nível de ruído")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR por nível de ruído (mediana e IQR)")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()

    out_path = output_dir / "fig4_psnr_by_noise.png"
    _save(fig, out_path)


def generate_visual_examples(*, noise: str = "40", output_dir: Path) -> None:
    """Create a montage of reference vs reconstructions for a few patches."""
    root = (
        PROJECT_ROOT
        / "paper_assets"
        / "classic"
        / "full_results"
        / "reconstruction_images"
    )
    ref_dir = root / "_reference"

    patch_ids = ["0000033", "0001101"]
    methods = ["rbf", "kriging", "exemplar_based"]

    # Columns: clean, degraded, mask, then method recons
    col_defs: list[tuple[str, Path]] = [
        ("Clean", ref_dir),
        ("Degraded", ref_dir),
        ("Mask", ref_dir),
        *[(m, root / m) for m in methods],
    ]

    n_rows = len(patch_ids)
    n_cols = len(col_defs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 2.0 * n_rows),
        squeeze=False,
    )

    for r, pid in enumerate(patch_ids):
        for c, (title, base_dir) in enumerate(col_defs):
            ax = axes[r][c]
            ax.axis("off")

            if title == "Clean":
                img_path = base_dir / f"{pid}_clean.png"
            elif title == "Degraded":
                img_path = base_dir / f"{pid}_degraded.png"
            elif title == "Mask":
                img_path = base_dir / f"{pid}_mask.png"
            else:
                img_path = base_dir / f"{pid}.png"

            if not img_path.exists():
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue

            img = plt.imread(img_path)
            ax.imshow(img)

            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.text(
                    -0.05,
                    0.5,
                    f"ID {int(pid)}",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=8,
                )

    fig.suptitle(f"Exemplos qualitativos (ruído {noise} dB)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = output_dir / "fig6_visual_examples_sentinel2.png"
    _save(fig, out_path)


def generate_lisa_clusters(
    *,
    noise: str = "40",
    method: str = "rbf",
    output_dir: Path,
) -> None:
    """Generate a LISA cluster map for one example patch."""
    arr_root = (
        PROJECT_ROOT
        / "paper_assets"
        / "classic"
        / "full_results"
        / "reconstruction_arrays"
        / noise
    )
    ref = arr_root / "_reference"
    recon_dir = arr_root / method
    patch_id = "0000033"

    clean_path = ref / f"{patch_id}_clean.npy"
    mask_path = ref / f"{patch_id}_mask.npy"
    recon_path = recon_dir / f"{patch_id}.npy"
    if not (clean_path.exists() and mask_path.exists() and recon_path.exists()):
        log.warning("Missing arrays for LISA figure (patch %s)", patch_id)
        return

    clean = np.load(clean_path)
    mask = np.load(mask_path)
    recon = np.load(recon_path)

    # Ensure HxW for mask
    mask2d = mask[..., 0] if mask.ndim == 3 else mask
    mask2d = (mask2d > 0.5).astype(np.uint8)

    # Error map: per-pixel MSE averaged over channels
    if clean.ndim == 3 and recon.ndim == 3:
        err = (recon - clean) ** 2
        err2d = err.mean(axis=-1)
    else:
        err2d = (recon - clean) ** 2

    spatial = spatial_autocorrelation(err2d, mask=mask2d)
    labels = spatial.lisa_labels.copy()
    pvals = spatial.lisa_p_values
    labels[pvals >= 0.05] = 0

    # Color mapping for q: 1=HH, 2=LH, 3=LL, 4=HL. 0=not significant.
    cmap = plt.get_cmap("tab10")
    colors = {
        0: (0.85, 0.85, 0.85, 1.0),
        1: cmap(3),  # red-ish
        2: cmap(1),  # orange
        3: cmap(2),  # green
        4: cmap(0),  # blue
    }
    rgb = np.zeros((*labels.shape, 4), dtype=float)
    for k, col in colors.items():
        rgb[labels == k] = col

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(rgb)
    ax.set_title(
        f"LISA (p<0.05) em erro de reconstrução\n{method}, "
        f"patch {int(patch_id)}",
        fontsize=9,
    )
    ax.axis("off")

    legend_items = [
        ("NS", 0),
        ("HH", 1),
        ("LH", 2),
        ("LL", 3),
        ("HL", 4),
    ]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color=colors[k],
            label=lab,
            linestyle="",
            markersize=8,
        )
        for lab, k in legend_items
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()

    out_path = output_dir / "fig5_lisa_clusters.png"
    _save(fig, out_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate additional paper figures used by LaTeX"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper_assets" / "figures",
        help="Output directory for generated/copied figures.",
    )
    parser.add_argument(
        "--noise",
        default="inf",
        help=(
            "Noise level for example-based figures. "
            "Use 'auto' to pick the best median PSNR from classic results. "
            "Default: inf."
        ),
    )
    return parser.parse_args()


def _resolve_noise(*, requested: str) -> str:
    req = str(requested)
    if req != "auto":
        return _normalize_noise(req)

    classic_csv = (
        PROJECT_ROOT
        / "paper_assets"
        / "classic"
        / "full_results"
        / "raw_results.csv"
    )
    best_noise = "40"
    if classic_csv.exists():
        df = pd.read_csv(classic_csv)
        df = df[df["status"] == "ok"].copy()
        df["noise_level"] = df["noise_level"].apply(_normalize_noise)
        med = df.groupby("noise_level", observed=True)["psnr"].median()
        order = ["inf", "40", "30", "20"]
        med = med.reindex([n for n in order if n in med.index])
        if not med.empty:
            best_noise = str(med.idxmax())
    return _normalize_noise(best_noise)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise = _resolve_noise(requested=str(args.noise))
    log.info("Noise used for extra figures: %s", noise)

    copy_entropy_window_panels(best_noise=noise, output_dir=output_dir)
    generate_psnr_by_noise(output_dir=output_dir)
    generate_visual_examples(noise=noise, output_dir=output_dir)
    generate_lisa_clusters(noise=noise, method="rbf", output_dir=output_dir)


if __name__ == "__main__":
    main()
