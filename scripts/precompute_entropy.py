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
import csv
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pdi_pipeline.entropy import shannon_entropy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preprocessed_dir: Path = args.preprocessed_dir
    windows: list[int] = args.windows
    resume: bool = args.resume

    manifest_path = preprocessed_dir / "manifest.csv"
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return

    with manifest_path.open() as fh:
        rows = list(csv.DictReader(fh))

    log.info("Loaded %d patches from manifest", len(rows))
    log.info("Entropy windows: %s", windows)

    n_computed = 0
    n_skipped = 0
    t0 = time.monotonic()

    # Track mean entropy values for manifest update
    entropy_means: dict[int, dict[str, float]] = {}

    for row in tqdm(rows, desc="Computing entropy"):
        patch_id = int(row["patch_id"])
        split = row["split"]
        satellite = row["satellite"]
        clean_path = preprocessed_dir / row["clean_path"]

        if not clean_path.exists():
            log.warning("Clean patch not found: %s", clean_path)
            continue

        patch_entropy: dict[str, float] = {}
        all_exist = True

        for ws in windows:
            out_path = entropy_path(
                preprocessed_dir, split, satellite, patch_id, ws
            )

            if resume and out_path.exists():
                arr = np.load(out_path)
                patch_entropy[f"mean_entropy_{ws}"] = float(np.mean(arr))
                continue

            all_exist = False

            clean = np.load(clean_path).astype(np.float32)
            ent = shannon_entropy(clean, window_size=ws)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, ent)

            patch_entropy[f"mean_entropy_{ws}"] = float(np.mean(ent))

        if all_exist and resume:
            n_skipped += 1
        else:
            n_computed += 1

        entropy_means[patch_id] = patch_entropy

    # Update manifest with entropy columns
    entropy_cols = [f"mean_entropy_{ws}" for ws in windows]

    for row in rows:
        pid = int(row["patch_id"])
        if pid in entropy_means:
            for col in entropy_cols:
                row[col] = str(
                    round(entropy_means[pid].get(col, float("nan")), 6)
                )
        else:
            for col in entropy_cols:
                row[col] = ""

    # Determine all fieldnames (original + entropy)
    fieldnames = list(rows[0].keys())
    for col in entropy_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.monotonic() - t0
    log.info("--- Summary ---")
    log.info("Computed: %d patches", n_computed)
    log.info("Skipped (resume): %d patches", n_skipped)
    log.info("Manifest updated with columns: %s", entropy_cols)
    log.info("Elapsed: %.1fs", elapsed)


if __name__ == "__main__":
    main()
