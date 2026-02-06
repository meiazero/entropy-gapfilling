"""Preprocess satellite image patches from GeoTIFF to NPY format.

Converts 77,916 GeoTIFF patches into NumPy arrays for fast loading during
experiments. Inverts mask convention from source (1=valid, 0=gap) to pipeline
convention (1=gap, 0=valid). Transposes images from rasterio's (C, H, W) to
(H, W, C) layout expected by BaseMethod._apply_channelwise().

Usage:
    uv run python scripts/preprocess_dataset.py
    uv run python scripts/preprocess_dataset.py --resume
    uv run python scripts/preprocess_dataset.py --limit 10 --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("/opt/datasets/satellite-images")
DEFAULT_SEED = 69
MANIFEST_NAME = "manifest.csv"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "preprocessed"

VARIANTS = [
    "clean",
    "degraded_inf",
    "degraded_40",
    "degraded_30",
    "degraded_20",
    "mask",
]

VARIANT_COL_MAP = {
    "clean": "clean_path",
    "mask": "mask_synthetic_path",
    "degraded_inf": "degraded_inf_path",
    "degraded_40": "degraded_40_path",
    "degraded_30": "degraded_30_path",
    "degraded_20": "degraded_20_path",
}

EXPECTED_IMAGE_SHAPE = (4, 64, 64)
EXPECTED_MASK_SHAPE = (1, 64, 64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess GeoTIFF patches to NPY format.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patches whose 6 output files already exist.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing metadata.parquet and GeoTIFF patches. "
            f"Default: PDI_DATA_ROOT env var or {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N randomly sampled patches per satellite.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for --limit sampling. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for preprocessed NPY files. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    return parser.parse_args(argv)


def resolve_data_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root
    env = os.environ.get("PDI_DATA_ROOT")
    if env:
        return Path(env)
    return DEFAULT_DATA_ROOT


def output_path_for(
    output_dir: Path,
    split: str,
    satellite: str,
    patch_id: int,
    variant: str,
) -> Path:
    return output_dir / split / satellite / f"{patch_id:07d}_{variant}.npy"


def validate_image(arr: np.ndarray, path: Path) -> None:
    if arr.shape != EXPECTED_IMAGE_SHAPE:
        msg = f"{path}: expected shape {EXPECTED_IMAGE_SHAPE}, got {arr.shape}"
        raise ValueError(msg)


def validate_mask(arr: np.ndarray, path: Path) -> None:
    if arr.shape != EXPECTED_MASK_SHAPE:
        msg = f"{path}: expected shape {EXPECTED_MASK_SHAPE}, got {arr.shape}"
        raise ValueError(msg)
    unique = np.unique(arr)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        msg = f"{path}: mask contains values other than 0 and 1: {unique}"
        raise ValueError(msg)


def validate_clean(arr: np.ndarray, path: Path) -> None:
    if np.any(np.isnan(arr)):
        msg = f"{path}: clean image contains NaN values"
        raise ValueError(msg)


def read_tiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def process_patch(
    row: pd.Series,
    data_root: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[dict, bool] | None:
    """Process a single patch.

    Returns (manifest_row, was_skipped) or None on failure.
    """
    patch_id: int = int(row["patch_id"])
    satellite: str = str(row["satellite"])
    split: str = str(row["split"])

    out_paths = {
        v: output_path_for(output_dir, split, satellite, patch_id, v)
        for v in VARIANTS
    }

    if resume and all(p.exists() for p in out_paths.values()):
        return _manifest_row(row, out_paths, output_dir), True

    src_paths = {v: data_root / str(row[VARIANT_COL_MAP[v]]) for v in VARIANTS}

    for _v, sp in src_paths.items():
        if not sp.exists():
            log.warning("patch %d: missing source file %s", patch_id, sp)
            return None

    try:
        arrays = {v: read_tiff(sp) for v, sp in src_paths.items()}
    except (rasterio.errors.RasterioError, OSError) as exc:
        log.warning("patch %d: read error - %s", patch_id, exc)
        return None

    try:
        for v in VARIANTS:
            if v == "mask":
                validate_mask(arrays[v], src_paths[v])
            else:
                validate_image(arrays[v], src_paths[v])
        validate_clean(arrays["clean"], src_paths["clean"])
    except ValueError as exc:
        log.warning("patch %d: validation failed - %s", patch_id, exc)
        return None

    saved_paths: list[Path] = []
    try:
        for v in VARIANTS:
            arr = arrays[v]
            if v == "mask":
                # (1, H, W) -> (H, W), invert: source 1=valid -> 1=gap
                processed = 1.0 - arr.squeeze(0)
            else:
                # (C, H, W) -> (H, W, C)
                processed = np.transpose(arr, (1, 2, 0))

            out_paths[v].parent.mkdir(parents=True, exist_ok=True)
            np.save(out_paths[v], processed)
            saved_paths.append(out_paths[v])
    except OSError as exc:
        log.warning("patch %d: write error - %s", patch_id, exc)
        _cleanup(saved_paths)
        return None

    return _manifest_row(row, out_paths, output_dir), False


def _manifest_row(
    row: pd.Series,
    out_paths: dict[str, Path],
    output_dir: Path,
) -> dict:
    mask_arr = np.load(out_paths["mask"])
    clean_arr = np.load(out_paths["clean"])

    gap_fraction = float(np.mean(mask_arr))
    height, width = mask_arr.shape
    n_bands = clean_arr.shape[2] if clean_arr.ndim == 3 else 1

    rel_paths = {
        v: str(p.relative_to(output_dir)) for v, p in out_paths.items()
    }

    return {
        "patch_id": int(row["patch_id"]),
        "satellite": str(row["satellite"]),
        "split": str(row["split"]),
        "source_file": str(row["source_file"]),
        "acquisition_date": str(row["acquisition_date"]),
        "bands": str(row["bands"]),
        "crs": str(row["crs"]),
        "col_off": int(row["col_off"]),
        "row_off": int(row["row_off"]),
        "height": height,
        "width": width,
        "n_bands": n_bands,
        "clean_path": rel_paths["clean"],
        "mask_path": rel_paths["mask"],
        "degraded_inf_path": rel_paths["degraded_inf"],
        "degraded_40_path": rel_paths["degraded_40"],
        "degraded_30_path": rel_paths["degraded_30"],
        "degraded_20_path": rel_paths["degraded_20"],
        "gap_fraction": round(gap_fraction, 6),
    }


def _cleanup(paths: list[Path]) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            p.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = resolve_data_root(args.data_root)
    output_dir = (
        args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    )

    log.info("Data root: %s", data_root)
    log.info("Output dir: %s", output_dir)

    parquet_path = data_root / "metadata.parquet"
    if not parquet_path.exists():
        log.error("metadata.parquet not found at %s", parquet_path)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    log.info("Loaded %d patches from metadata.parquet", len(df))

    if args.limit is not None:
        rng = np.random.default_rng(args.seed)
        sampled = []
        for _sat, group in df.groupby("satellite"):
            n = min(args.limit, len(group))
            idx = rng.choice(len(group), size=n, replace=False)
            sampled.append(group.iloc[sorted(idx)])
        df = pd.concat(sampled).reset_index(drop=True)
        log.info(
            "Sampled %d patches (%d per satellite, seed=%d)",
            len(df),
            args.limit,
            args.seed,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    failed_ids: list[int] = []
    n_skipped = 0
    n_processed = 0

    t0 = time.monotonic()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        result = process_patch(row, data_root, output_dir, resume=args.resume)
        if result is None:
            failed_ids.append(int(row["patch_id"]))
        else:
            manifest_row, was_skipped = result
            manifest_rows.append(manifest_row)
            if was_skipped:
                n_skipped += 1
            else:
                n_processed += 1

    elapsed = time.monotonic() - t0

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_df.to_csv(manifest_path, index=False)

    log.info("--- Summary ---")
    log.info("Total patches in selection: %d", len(df))
    log.info("Processed: %d", n_processed)
    log.info("Skipped (resume): %d", n_skipped)
    log.info("Failed: %d", len(failed_ids))
    log.info("Manifest written to: %s", manifest_path)
    log.info("Elapsed: %.1fs", elapsed)

    if failed_ids:
        log.warning("Failed patch IDs: %s", failed_ids)


if __name__ == "__main__":
    main()
