"""Select figure variants for the article.

By default, this script uses a deterministic, non-cherry-picked
reference scenario:

- Prefer noise level ``inf`` (gap-only) when available.
- Use entropy window 15 when available.

You can override the noise level via ``--noise``.

The script copies the chosen variants from ``paper_assets/figures`` to
``docs/figures`` using stable filenames referenced by the LaTeX sources.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .data_loader import ENTROPY_WINDOWS, NOISE_ORDER, load_combined


def _best_noise(df: pd.DataFrame) -> str:
    """Backwards-compatible: return noise level with highest median PSNR."""
    if "noise_level" not in df.columns:
        return "inf"
    med = (
        df
        .groupby("noise_level", observed=True)["psnr"]
        .median()
        .reindex([n for n in NOISE_ORDER if n in df["noise_level"].unique()])
    )
    if med.empty:
        return "inf"
    return str(med.idxmax())


def _select_noise(df: pd.DataFrame, *, preferred: str) -> str:
    if "noise_level" not in df.columns:
        return "inf"
    available = set(df["noise_level"].dropna().astype(str).unique().tolist())
    if preferred in available:
        return preferred
    # Fall back to previous behavior when the preferred scenario is absent.
    return _best_noise(df)


def _best_entropy_window(df: pd.DataFrame) -> int:
    # PSNR does not depend on the entropy window, so use the middle window
    # when available to avoid overfitting presentation choices.
    if 15 in ENTROPY_WINDOWS and "entropy_15" in df.columns:
        return 15
    return ENTROPY_WINDOWS[0]


def _best_dl_scenario(df: pd.DataFrame) -> str:
    if df.empty or "entropy_scenario" not in df.columns:
        return "entropy_all"
    med = df.groupby("entropy_scenario", observed=True)["psnr"].median()
    if med.empty:
        return "entropy_all"
    return str(med.idxmax())


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"WARN missing figure: {src.name}")
        return False
    shutil.copy2(src, dst)
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select paper figure assets")
    parser.add_argument(
        "--noise",
        default="inf",
        help=(
            "Noise level to use as reference (default: inf). "
            "If not available, falls back to the level with best median PSNR."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent.parent
    figures_dir = root / "paper_assets" / "figures"
    output_dir = root / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous outputs we manage to avoid stale files.
    managed = {
        "fig1_pareto_classic_best.png",
        "fig1_pareto_sentinel2_best.png",
        "fig2_spectral_bar_best.png",
        "fig6_psnr_entropy_best.png",
        "fig7_correlation_heatmap_best.png",
        "fig11_dl_comparison_best.png",
        "fig2_entropy_vs_psnr_7x7.png",
        "fig2_entropy_vs_psnr_15x15.png",
        "fig2_entropy_vs_psnr_31x31.png",
        "fig4_psnr_by_noise.png",
        "fig5_lisa_clusters.png",
        "fig6_visual_examples_sentinel2.png",
    }
    for name in managed:
        (output_dir / name).unlink(missing_ok=True)

    df = load_combined()
    if df.empty:
        print("No data loaded. Skipping selection.")
        return

    best_noise = _select_noise(df, preferred=str(args.noise))
    best_entropy = _best_entropy_window(df)
    dl_df = df[df["type"] == "DL"] if "type" in df.columns else pd.DataFrame()
    best_scenario = _best_dl_scenario(dl_df)

    suffix = best_noise.replace("inf", "gap_only")

    selected = {
        f"fig1_pareto_classic_{suffix}.png": "fig1_pareto_classic_best.png",
        f"fig1_pareto_sentinel2_{suffix}.png": (
            "fig1_pareto_sentinel2_best.png"
        ),
        f"fig2_spectral_bar_{suffix}.png": "fig2_spectral_bar_best.png",
        f"fig6_psnr_entropy_{suffix}_e{best_entropy}.png": (
            "fig6_psnr_entropy_best.png"
        ),
        "fig7_correlation_heatmap.png": "fig7_correlation_heatmap_best.png",
        f"fig11_dl_comparison_{best_scenario}.png": (
            "fig11_dl_comparison_best.png"
        ),
        # Extra figures referenced by LaTeX (stable filenames)
        "fig2_entropy_vs_psnr_7x7.png": "fig2_entropy_vs_psnr_7x7.png",
        "fig2_entropy_vs_psnr_15x15.png": "fig2_entropy_vs_psnr_15x15.png",
        "fig2_entropy_vs_psnr_31x31.png": "fig2_entropy_vs_psnr_31x31.png",
        "fig4_psnr_by_noise.png": "fig4_psnr_by_noise.png",
        "fig5_lisa_clusters.png": "fig5_lisa_clusters.png",
        "fig6_visual_examples_sentinel2.png": "fig6_visual_examples_sentinel2.png",  # noqa: E501
    }

    copied = 0
    for src_name, dst_name in selected.items():
        if _copy_if_exists(figures_dir / src_name, output_dir / dst_name):
            copied += 1

    print("Selected noise:", best_noise)
    print("Selected entropy window:", best_entropy)
    print("Selected DL scenario:", best_scenario)
    print("Copied", copied, "of", len(selected), "figures")


if __name__ == "__main__":
    main()
