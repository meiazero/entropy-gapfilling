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

from .data_loader import load_combined
from .figures.common import SETTINGS, configure_settings, setup_style
from .figures.correlation_heatmap import fig7_correlation_heatmap
from .figures.dl_comparison import fig11_dl_comparison
from .figures.dl_components import fig10_components
from .figures.dl_loss import fig8_dl_loss
from .figures.dl_val_metrics import fig9_dl_val_metrics
from .figures.entropy_sensitivity import fig3_sensitivity
from .figures.f1_threshold import fig5_f1_threshold
from .figures.multisensor import fig4_multisensor
from .figures.pareto import fig1_pareto
from .figures.psnr_entropy import fig6_psnr_entropy
from .figures.spectral_bar import fig2_spectral_bar
from .figures.spectral_dotplot import fig2_spectral_dotplot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=int(SETTINGS.bootstrap_samples),
        help="Number of bootstrap resamples for CI estimates.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_settings(
        png_only=args.png_only,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    setup_style()

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
