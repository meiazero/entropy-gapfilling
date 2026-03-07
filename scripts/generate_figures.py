"""Generate publication-quality figures from experiment results.

Data-driven rewrite. Produces 11 figure types with variations per noise
level, entropy window, model, and entropy scenario.

Usage:
    uv run python scripts/generate_figures.py
    uv run python scripts/generate_figures.py --output docs/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.data_loader import load_combined
    from scripts.figures.classical_noise_robustness import (
        fig_classical_noise_robustness,
    )
    from scripts.figures.classical_spectral_profile import (
        fig_classical_spectral_profile,
    )
    from scripts.figures.common import SETTINGS, configure_settings, setup_style
    from scripts.figures.correlation_heatmap import (
        fig_classical_correlation_heatmap,
    )
    from scripts.figures.dl_comparison import fig_dl_comparison
    from scripts.figures.dl_noise_robustness import fig_dl_noise_robustness
    from scripts.figures.pareto import fig1_pareto
else:
    from .data_loader import load_combined
    from .figures.classical_noise_robustness import (
        fig_classical_noise_robustness,
    )
    from .figures.classical_spectral_profile import (
        fig_classical_spectral_profile,
    )
    from .figures.common import SETTINGS, configure_settings, setup_style
    from .figures.correlation_heatmap import fig_classical_correlation_heatmap
    from .figures.dl_comparison import fig_dl_comparison
    from .figures.dl_noise_robustness import fig_dl_noise_robustness
    from .figures.pareto import fig1_pareto

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _remove_stale_raster_outputs(output_dir: Path) -> None:
    stale_files = []
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        stale_files.extend(output_dir.glob(pattern))

    for file_path in stale_files:
        file_path.unlink()

    if stale_files:
        log.info("Removed %d stale raster figure(s)", len(stale_files))


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication figures."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to docs/figures/",
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
        bootstrap_samples=int(args.bootstrap_samples),
    )
    setup_style()

    output_dir = args.output or Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_raster_outputs(output_dir)

    df = load_combined()

    # ── Evaluation data figures ──
    if not df.empty:
        eval_figs = [
            ("Classic Pareto", fig1_pareto),
            ("Classic Spectral Profile", fig_classical_spectral_profile),
            ("Classic Noise Robustness", fig_classical_noise_robustness),
            ("Classic Correlation Heatmap", fig_classical_correlation_heatmap),
        ]
        for name, func in eval_figs:
            try:
                log.info("Generating %s...", name)
                func(df, output_dir)
            except Exception:
                log.exception("Error generating %s", name)
    else:
        log.warning("No evaluation data loaded. Skipping eval figures.")

    dl_figs = [
        ("DL Comparison", fig_dl_comparison),
        ("DL Noise Robustness", fig_dl_noise_robustness),
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
