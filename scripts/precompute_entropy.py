"""Precompute local entropy maps for all clean patches.

For each clean patch in the manifest, computes Shannon entropy at
multiple window sizes and saves the result as NPY files. Also updates
the manifest CSV with mean entropy columns.

Usage:
    uv run python scripts/precompute_entropy.py
    uv run python scripts/precompute_entropy.py --resume
    uv run python scripts/precompute_entropy.py --windows 7 15 31
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pdi_pipeline.entropy import shannon_entropy
from pdi_pipeline.logging_utils import get_project_root, setup_logging

setup_logging()
log = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()
DEFAULT_PREPROCESSED = PROJECT_ROOT / "preprocessed"
DEFAULT_WINDOWS = [7, 15, 31]


def entropy_path(
    base_dir: Path,
    split: str,
    satellite: str,
    patch_id: int,
    window: int,
) -> Path:
    return base_dir / split / satellite / f"{patch_id:07d}_entropy_{window}.npy"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute entropy maps for clean patches.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=DEFAULT_PREPROCESSED,
        help=f"Preprocessed data directory. Default: {DEFAULT_PREPROCESSED}",
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help=f"Entropy window sizes. Default: {DEFAULT_WINDOWS}",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patches whose entropy files already exist.",
    )
    return parser.parse_args(argv)


def _compute_patch_entropy(
    row: dict[str, object],
    preprocessed_dir: Path,
    windows: list[int],
    *,
    resume: bool,
) -> tuple[int, dict[str, float], bool] | None:
    patch_id = int(row["patch_id"])
    split = str(row["split"])
    satellite = str(row["satellite"])
    clean_path = preprocessed_dir / str(row["clean_path"])

    if not clean_path.exists():
        log.warning("Clean patch not found: %s", clean_path)
        return None

    patch_entropy: dict[str, float] = {}
    all_exist = True
    clean: np.ndarray | None = None

    for ws in windows:
        out_path = entropy_path(
            preprocessed_dir, split, satellite, patch_id, ws
        )

        if resume and out_path.exists():
            arr = np.load(out_path)
            patch_entropy[f"mean_entropy_{ws}"] = float(np.mean(arr))
            continue

        all_exist = False
        if clean is None:
            clean = np.load(clean_path).astype(np.float32)

        ent = shannon_entropy(clean, window_size=ws)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, ent)
        patch_entropy[f"mean_entropy_{ws}"] = float(np.mean(ent))

    return patch_id, patch_entropy, all_exist and resume


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preprocessed_dir: Path = args.preprocessed_dir
    windows: list[int] = args.windows
    resume: bool = args.resume

    manifest_path = preprocessed_dir / "manifest.csv"
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return

    manifest_df = pd.read_csv(manifest_path)
    rows = manifest_df.to_dict("records")

    log.info("Loaded %d patches from manifest", len(rows))
    log.info("Entropy windows: %s", windows)

    n_computed = 0
    n_skipped = 0
    t0 = time.monotonic()

    # Track mean entropy values for manifest update
    entropy_means: dict[int, dict[str, float]] = {}

    for row in tqdm(rows, desc="Computing entropy"):
        result = _compute_patch_entropy(
            row,
            preprocessed_dir,
            windows,
            resume=resume,
        )
        if result is None:
            continue

        patch_id, patch_entropy, was_skipped = result
        if was_skipped:
            n_skipped += 1
        else:
            n_computed += 1
        entropy_means[patch_id] = patch_entropy

    # Vectorized manifest update: merge entropy_means into the DataFrame
    entropy_cols = [f"mean_entropy_{ws}" for ws in windows]

    if entropy_means:
        entropy_df = pd.DataFrame.from_dict(entropy_means, orient="index")
        entropy_df.index.name = "patch_id"
        entropy_df = entropy_df.reset_index()
        manifest_df = manifest_df.drop(
            columns=[c for c in entropy_cols if c in manifest_df.columns],
            errors="ignore",
        )
        manifest_df = manifest_df.merge(entropy_df, on="patch_id", how="left")

    for col in entropy_cols:
        if col not in manifest_df.columns:
            manifest_df[col] = float("nan")

    manifest_df.to_csv(manifest_path, index=False)

    elapsed = time.monotonic() - t0
    log.info("--- Summary ---")
    log.info("Computed: %d patches", n_computed)
    log.info("Skipped (resume): %d patches", n_skipped)
    log.info("Manifest updated with columns: %s", entropy_cols)
    log.info("Elapsed: %.1fs", elapsed)


if __name__ == "__main__":
    main()
