"""Generate LaTeX tables from experiment results (data-driven rewrite).

Produces 5 table types with variations per noise level, entropy window,
and satellite. Each table is saved as a standalone .tex file.

Usage:
    uv run python scripts/generate_tables.py
    uv run python scripts/generate_tables.py --output docs/tables
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .data_loader import load_combined
from .tables.common import SETTINGS, configure_settings
from .tables.degradation_entropy import table_degradation_entropy
from .tables.global_scoreboard import table_global_scoreboard
from .tables.noise_slope import table_noise_slope
from .tables.overview_dataset import table_overview_dataset
from .tables.spearman_entropy import table_spearman_entropy
from .tables.spectral_decomposition import table_spectral_decomposition
from .tables.speed_summary import table_speed_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to paper_assets/tables/",
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
    configure_settings(bootstrap_samples=int(args.bootstrap_samples))

    df = load_combined()
    if df.empty:
        log.error("No data loaded. Check paper_assets/ paths.")
        return

    output_dir = args.output or Path("paper_assets/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("Table 0: Dataset Overview", table_overview_dataset),
        ("Table 1: Global Scoreboard", table_global_scoreboard),
        ("Table 2: Spectral Decomposition", table_spectral_decomposition),
        ("Table 3: Degradation by Entropy", table_degradation_entropy),
        ("Table 3b: Noise Slope", table_noise_slope),
        ("Table 4: Spearman Correlation", table_spearman_entropy),
        ("Table 5: Speed", table_speed_summary),
    ]

    for name, func in generators:
        try:
            log.info("Generating %s...", name)
            func(df, output_dir)
        except Exception:
            log.exception("Error generating %s", name)

    log.info("All tables saved to: %s", output_dir)


if __name__ == "__main__":
    main()
