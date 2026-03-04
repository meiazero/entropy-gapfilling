"""CLI helpers for running individual figure generators."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from ..data_loader import load_combined
from .common import SETTINGS, configure_settings, setup_style

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
    return parser.parse_args()


def _prepare_output_dir(output: Path | None) -> Path:
    output_dir = output or Path("paper_assets/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _configure_runtime(args: argparse.Namespace) -> Path:
    configure_settings(
        png_only=args.png_only,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    setup_style()
    return _prepare_output_dir(args.output)


def run_with_df(
    fig_fn: Callable[[object, Path], None],
    description: str,
) -> None:
    args = _parse_args(description)
    output_dir = _configure_runtime(args)
    df = load_combined()
    if df.empty:
        log.warning("No evaluation data loaded. Skipping %s.", description)
        return
    fig_fn(df, output_dir)
    log.info("Saved %s to: %s", description, output_dir)


def run_no_df(
    fig_fn: Callable[[Path], None],
    description: str,
) -> None:
    args = _parse_args(description)
    output_dir = _configure_runtime(args)
    fig_fn(output_dir)
    log.info("Saved %s to: %s", description, output_dir)
