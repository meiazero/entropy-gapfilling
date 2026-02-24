"""Preprocess satellite image patches from GeoTIFF to NPY format.

Converts GeoTIFF patches into NumPy arrays for fast loading during
experiments. Inverts mask convention from source (1=valid, 0=gap) to pipeline
convention (1=gap, 0=valid). Transposes images from rasterio's (C, H, W) to
(H, W, C) layout expected by BaseMethod._apply_channelwise().

When ``--config`` is given, simulates PatchDataset's deterministic selection
across all experiment seeds and writes the per-seed selections to
``patch_selections.json``. The preprocessing step always targets the shared
manifest so classical and DL pipelines use the same NPY dataset.

Usage:
    uv run python scripts/preprocess_dataset.py
    uv run python scripts/preprocess_dataset.py \
        --config config/paper_results.yaml --resume
    uv run python scripts/preprocess_dataset.py \
        --config config/quick_validation.yaml --resume
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
from tqdm import tqdm

from pdi_pipeline.logging_utils import get_project_root, setup_logging

setup_logging()
log = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path("/opt/datasets/satellite-images")
MANIFEST_NAME = "manifest.csv"

PROJECT_ROOT = get_project_root()
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("PDI_PREPROCESSED_DIR", str(PROJECT_ROOT / "preprocessed"))
)

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

SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}
SPLIT_SEED = 42


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
        "--config",
        type=Path,
        default=None,
        help=(
            "Experiment YAML config. When provided, preprocesses only the "
            "patches required by the experiment (satellites, seeds, "
            "max_patches). Saves per-seed patch selections to JSON."
        ),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel worker threads for I/O-bound TIFF reading. "
            "Default: min(cpu_count, 8)."
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
    invalid = np.any((arr != 0.0) & (arr != 1.0))
    if invalid:
        unique = np.unique(arr)
        msg = f"{path}: mask contains values other than 0 and 1: {unique}"
        raise ValueError(msg)


def validate_clean(arr: np.ndarray, path: Path) -> None:
    if np.any(np.isnan(arr)):
        msg = f"{path}: clean image contains NaN values"
        raise ValueError(msg)


def read_tiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(out_dtype=np.float32, masked=False)


def _resume_manifest_row(
    row: tuple,
    row_fields: dict[str, int],
    out_paths: dict[str, Path],
    output_dir: Path,
) -> tuple[dict, bool]:
    mask_arr = np.load(out_paths["mask"])
    clean_arr = np.load(out_paths["clean"])
    manifest_row = _manifest_row(
        row,
        row_fields,
        out_paths,
        output_dir,
        mask_arr=mask_arr,
        clean_arr=clean_arr,
    )
    return manifest_row, True


def _load_and_validate_sources(
    patch_id: int,
    src_paths: dict[str, Path],
) -> dict[str, np.ndarray] | None:
    for source_path in src_paths.values():
        if not source_path.exists():
            log.warning(
                "patch %d: missing source file %s", patch_id, source_path
            )
            return None

    try:
        arrays = {
            variant: read_tiff(path) for variant, path in src_paths.items()
        }
    except (rasterio.errors.RasterioError, OSError) as exc:
        log.warning("patch %d: read error - %s", patch_id, exc)
        return None

    try:
        for variant in VARIANTS:
            if variant == "mask":
                validate_mask(arrays[variant], src_paths[variant])
            else:
                validate_image(arrays[variant], src_paths[variant])
        validate_clean(arrays["clean"], src_paths["clean"])
    except ValueError as exc:
        log.warning("patch %d: validation failed - %s", patch_id, exc)
        return None

    return arrays


def _transform_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    processed: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        arr = arrays[variant]
        if variant == "mask":
            processed[variant] = 1.0 - arr.squeeze(0)
        else:
            processed[variant] = np.transpose(arr, (1, 2, 0))
    return processed


def _save_processed_arrays(
    patch_id: int,
    out_paths: dict[str, Path],
    processed: dict[str, np.ndarray],
) -> bool:
    saved_paths: list[Path] = []
    try:
        for variant in VARIANTS:
            out_paths[variant].parent.mkdir(parents=True, exist_ok=True)
            np.save(out_paths[variant], processed[variant])
            saved_paths.append(out_paths[variant])
    except OSError as exc:
        log.warning("patch %d: write error - %s", patch_id, exc)
        _cleanup(saved_paths)
        return False
    return True


def process_patch(
    row: tuple,
    row_fields: dict[str, int],
    data_root: Path,
    output_dir: Path,
    *,
    resume: bool,
) -> tuple[dict, bool] | None:
    """Process a single patch.

    Args:
        row: A namedtuple-like row from itertuples().
        row_fields: Mapping of column name to attribute index.
        data_root: Root directory for source GeoTIFF files.
        output_dir: Output directory for NPY files.
        resume: Skip patches whose output files already exist.

    Returns (manifest_row, was_skipped) or None on failure.
    """
    patch_id: int = int(row[row_fields["patch_id"]])
    satellite: str = str(row[row_fields["satellite"]])
    split: str = str(row[row_fields["split"]])

    out_paths = {
        v: output_path_for(output_dir, split, satellite, patch_id, v)
        for v in VARIANTS
    }

    if resume and all(p.exists() for p in out_paths.values()):
        return _resume_manifest_row(row, row_fields, out_paths, output_dir)

    src_paths = {
        v: data_root / str(row[row_fields[VARIANT_COL_MAP[v]]])
        for v in VARIANTS
    }

    arrays = _load_and_validate_sources(patch_id, src_paths)
    if arrays is None:
        return None

    processed = _transform_arrays(arrays)
    if not _save_processed_arrays(patch_id, out_paths, processed):
        return None

    mrow = _manifest_row(
        row,
        row_fields,
        out_paths,
        output_dir,
        mask_arr=processed["mask"],
        clean_arr=processed["clean"],
    )
    return mrow, False


def _manifest_row(
    row: tuple,
    row_fields: dict[str, int],
    out_paths: dict[str, Path],
    output_dir: Path,
    *,
    mask_arr: np.ndarray,
    clean_arr: np.ndarray,
) -> dict:
    gap_fraction = float(np.mean(mask_arr))
    height, width = mask_arr.shape[:2]
    n_bands = clean_arr.shape[2] if clean_arr.ndim == 3 else 1

    rel_paths = {
        v: str(p.relative_to(output_dir)) for v, p in out_paths.items()
    }

    return {
        "patch_id": int(row[row_fields["patch_id"]]),
        "satellite": str(row[row_fields["satellite"]]),
        "split": str(row[row_fields["split"]]),
        "source_file": str(row[row_fields["source_file"]]),
        "acquisition_date": str(row[row_fields["acquisition_date"]]),
        "bands": str(row[row_fields["bands"]]),
        "crs": str(row[row_fields["crs"]]),
        "col_off": int(row[row_fields["col_off"]]),
        "row_off": int(row[row_fields["row_off"]]),
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


def _stable_seed(base_seed: int, salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16)


def _assign_splits(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    total_ratio = sum(ratios.values())
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        msg = f"Split ratios must sum to 1.0, got {total_ratio:.6f}"
        raise ValueError(msg)

    df = df.copy()
    df["split"] = ""

    for sat in sorted(df["satellite"].unique()):
        sat_idx = df.index[df["satellite"] == sat].to_numpy().copy()
        rng = np.random.default_rng(_stable_seed(seed, sat))
        rng.shuffle(sat_idx)

        n = len(sat_idx)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        train_idx = sat_idx[:n_train]
        val_idx = sat_idx[n_train : n_train + n_val]
        test_idx = sat_idx[n_train + n_val :]

        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    if (df["split"] == "").any():
        msg = "Split assignment incomplete - some rows are unassigned"
        raise RuntimeError(msg)

    return df


def _split_ratios_ok(
    splits: pd.Series,
    ratios: dict[str, float],
    tol: float = 0.01,
) -> bool:
    counts = splits.value_counts()
    total = int(counts.sum())
    if total == 0:
        return False
    for name, expected in ratios.items():
        actual = counts.get(name, 0) / total
        if abs(actual - expected) > tol:
            return False
    return True


def _raise_split_ratio_mismatch() -> None:
    exc = ValueError("split ratios mismatch")
    raise exc


def _simulate_selection(
    df: pd.DataFrame,
    satellites: list[str],
    seeds: list[int],
    max_patches: int | None,
) -> dict[str, dict[int, list[int]]]:
    """Replicate PatchDataset's deterministic patch selection on raw metadata.

    For each (satellite, seed) pair, applies the same sort-then-sample
    logic as PatchDataset.__init__ on the train split.

    Returns:
        Nested dict ``{satellite: {seed: [sorted_patch_ids]}}``.
    """
    selections: dict[str, dict[int, list[int]]] = {}

    for sat in satellites:
        sat_df = df[(df["split"] == "train") & (df["satellite"] == sat)]
        sat_df = sat_df.sort_values("patch_id").reset_index(drop=True)
        all_ids = sat_df["patch_id"].tolist()
        n = len(all_ids)

        selections[sat] = {}
        for seed in seeds:
            if max_patches is not None and n > max_patches:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n, size=max_patches, replace=False)
                idx.sort()
                selected = [int(all_ids[i]) for i in idx]
            else:
                selected = [int(pid) for pid in all_ids]
            selections[sat][seed] = selected

    return selections


def _save_patch_selections(
    selections: dict[str, dict[int, list[int]]],
    output_dir: Path,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save per-seed patch selections to JSON for the experiment runner."""
    serializable = {
        sat: {str(seed): ids for seed, ids in seed_map.items()}
        for sat, seed_map in selections.items()
    }
    payload: dict[str, object]
    if metadata is None:
        payload = serializable
    else:
        payload = {"metadata": metadata, "selections": serializable}
    path = output_dir / "patch_selections.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Patch selections saved to: %s", path)


def _try_resume_fast_path(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
) -> bool:
    manifest_path = output_dir / MANIFEST_NAME
    if not args.resume or not manifest_path.exists():
        return False

    try:
        existing = pd.read_csv(manifest_path, usecols=["split"], dtype=str)
        existing_splits = set(existing["split"].unique())
        expected_splits = {"train", "val", "test"}
        if not expected_splits.issubset(existing_splits):
            return False

        if not _split_ratios_ok(existing["split"], SPLIT_RATIOS):
            log.info(
                "Manifest split ratios differ from expected %s; reprocessing.",
                SPLIT_RATIOS,
            )
            _raise_split_ratio_mismatch()

        n_existing = len(existing)
        n_expected = len(df) if args.config is None else None
        if n_expected is not None and n_existing < n_expected:
            return False

        log.info(
            "Manifest already complete (%d rows, splits=%s). "
            "Skipping preprocessing.",
            n_existing,
            sorted(existing_splits),
        )
    except Exception:
        log.debug("Could not read existing manifest, will re-process")
        return False
    else:
        return True


def _apply_config_selection(
    args: argparse.Namespace,
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    if args.config is None:
        return df

    from pdi_pipeline.config import load_config

    cfg = load_config(args.config)
    log.info(
        "Config: %s (satellites=%s, seeds=%d, max_patches=%s)",
        cfg.name,
        cfg.satellites,
        len(cfg.seeds),
        cfg.max_patches,
    )

    filtered = df[df["satellite"].isin(cfg.satellites)].reset_index(drop=True)
    log.info(
        "Selection satellites %s: %d patches",
        cfg.satellites,
        len(filtered),
    )

    if cfg.max_patches is None:
        return df

    selections = _simulate_selection(
        filtered,
        cfg.satellites,
        cfg.seeds,
        cfg.max_patches,
    )
    union_ids: set[int] = set()
    for sat_sel in selections.values():
        for patch_ids in sat_sel.values():
            union_ids.update(patch_ids)

    log.info(
        "Patch selection: %d unique patches across %d seeds"
        " (from %d candidates)",
        len(union_ids),
        len(cfg.seeds),
        len(filtered),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": cfg.name,
        "satellites": cfg.satellites,
        "seeds": cfg.seeds,
        "max_patches": cfg.max_patches,
    }
    _save_patch_selections(selections, output_dir, metadata)
    return df


def _parallel_process_rows(
    rows_list: list[tuple],
    row_fields: dict[str, int],
    data_root: Path,
    output_dir: Path,
    *,
    resume: bool,
    workers: int,
) -> tuple[list[dict], list[int], int, int]:
    manifest_rows: list[dict] = []
    failed_ids: list[int] = []
    n_skipped = 0
    n_processed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_patch,
                row,
                row_fields,
                data_root,
                output_dir,
                resume=resume,
            ): row
            for row in rows_list
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Preprocessing"
        ):
            row = futures[future]
            result = future.result()
            if result is None:
                failed_ids.append(int(row[row_fields["patch_id"]]))
                continue

            manifest_row, was_skipped = result
            manifest_rows.append(manifest_row)
            if was_skipped:
                n_skipped += 1
            else:
                n_processed += 1

    return manifest_rows, failed_ids, n_skipped, n_processed


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

    _needed_cols = [
        "patch_id",
        "satellite",
        "source_file",
        "acquisition_date",
        "bands",
        "crs",
        "col_off",
        "row_off",
        *VARIANT_COL_MAP.values(),
    ]
    df = pd.read_parquet(parquet_path, columns=_needed_cols)
    df["patch_id"] = df["patch_id"].astype(np.int32)
    df["satellite"] = df["satellite"].astype("category")
    log.info("Loaded %d patches from metadata.parquet", len(df))

    df = _assign_splits(df, SPLIT_RATIOS, SPLIT_SEED)
    log.info(
        "Assigned splits with ratios %s (seed=%d)",
        SPLIT_RATIOS,
        SPLIT_SEED,
    )

    df = _apply_config_selection(args, df, output_dir)

    if _try_resume_fast_path(args, output_dir, df):
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # Build column index map for itertuples() access (index offset +1
    # because itertuples prepends the Index element at position 0).
    row_fields = {col: i + 1 for i, col in enumerate(df.columns)}

    n_workers = args.workers if args.workers is not None else os.cpu_count() - 1
    log.info("Using %d worker threads for parallel TIFF I/O", n_workers)

    rows_list = list(df.itertuples())
    manifest_rows, failed_ids, n_skipped, n_processed = _parallel_process_rows(
        rows_list,
        row_fields,
        data_root,
        output_dir,
        resume=args.resume,
        workers=n_workers,
    )

    elapsed = time.monotonic() - t0

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / MANIFEST_NAME

    if manifest_path.exists():
        existing_df = pd.read_csv(manifest_path)
        extra_cols = [
            col for col in existing_df.columns if col not in manifest_df.columns
        ]
        if extra_cols:
            manifest_df = manifest_df.merge(
                existing_df[["patch_id", *extra_cols]],
                on="patch_id",
                how="left",
            )
            log.info(
                "Preserved %d extra manifest columns: %s",
                len(extra_cols),
                extra_cols,
            )

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
