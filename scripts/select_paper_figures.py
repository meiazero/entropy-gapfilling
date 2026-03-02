"""Select best figure variants for the article.

Decision rule: pick the noise level and entropy window that maximize
median PSNR over the combined evaluation data. Then copy only the
corresponding figure variants from paper_assets/figures to docs/figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from data_loader import ENTROPY_WINDOWS, NOISE_ORDER, load_combined


def _best_noise(df: pd.DataFrame) -> str:
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


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    figures_dir = root / "paper_assets" / "figures"
    output_dir = root / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous best outputs
    for old in output_dir.glob("*_best.*"):
        old.unlink(missing_ok=True)

    df = load_combined()
    if df.empty:
        print("No data loaded. Skipping selection.")
        return

    best_noise = _best_noise(df)
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
    }

    for src_name, dst_name in selected.items():
        _copy_if_exists(figures_dir / src_name, output_dir / dst_name)

    print("Selected noise:", best_noise)
    print("Selected entropy window:", best_entropy)
    print("Selected DL scenario:", best_scenario)
    print("Copied", len(selected), "main figures")


if __name__ == "__main__":
    main()
