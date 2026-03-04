"""CLI helpers for running individual table generators."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from ..data_loader import load_combined
from .common import SETTINGS, configure_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to docs/tables/",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=int(SETTINGS.bootstrap_samples),
        help="Number of bootstrap resamples for CI estimates.",
    )
    return parser.parse_args()


def _prepare_output_dir(output: Path | None) -> Path:
    output_dir = output or Path("docs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_table(
    table_fn: Callable[[object, Path], None],
    description: str,
) -> None:
    args = _parse_args(description)
    configure_settings(bootstrap_samples=int(args.bootstrap_samples))
    output_dir = _prepare_output_dir(args.output)

    df = load_combined()
    if df.empty:
        log.warning("No data loaded. Skipping %s.", description)
        return

    table_fn(df, output_dir)
    log.info("Saved %s to: %s", description, output_dir)
