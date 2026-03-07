"""Copy only the curated paper figures into docs/figures."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    source_dir = root / "paper_assets" / "figures"
    output_dir = root / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        "fig_classical_pareto.png",
        "fig_classical_pareto.pdf",
        "fig_classical_spectral_profile.png",
        "fig_classical_spectral_profile.pdf",
        "fig_classical_noise_robustness_baixa.png",
        "fig_classical_noise_robustness_baixa.pdf",
        "fig_classical_noise_robustness_media.png",
        "fig_classical_noise_robustness_media.pdf",
        "fig_classical_noise_robustness_alta.png",
        "fig_classical_noise_robustness_alta.pdf",
        "fig_classical_correlation_heatmap.png",
        "fig_classical_correlation_heatmap.pdf",
        "fig_dl_comparison.png",
        "fig_dl_comparison.pdf",
        "fig_dl_noise_robustness.png",
        "fig_dl_noise_robustness.pdf",
    ]

    copied = 0
    for name in selected:
        src = source_dir / name
        dst = output_dir / name
        if not src.exists():
            print(f"WARN missing figure: {name}")
            continue
        shutil.copy2(src, dst)
        copied += 1

    print("Copied", copied, "of", len(selected), "figures")


if __name__ == "__main__":
    main()
