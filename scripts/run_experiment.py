"""Run the gap-filling experiment pipeline.

Samples patches using the configured seeds, then evaluates each selected
patch once across noise levels and methods. Results are written to a
Parquet file with checkpointing for resumable execution.

Usage:
    uv run python scripts/run_experiment.py --config config/paper_results.yaml
    uv run python scripts/run_experiment.py --quick
    uv run python scripts/run_experiment.py --quick --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from pdi_pipeline.config import ExperimentConfig, load_config
from pdi_pipeline.dataset import PatchDataset
from pdi_pipeline.experiment_artifacts import (
    ReconstructionManager,
    compute_entropy_extremes,
    write_entropy_extremes_manifest,
)
from pdi_pipeline.logging_utils import (
    get_project_root,
    setup_file_logging,
    setup_logging,
)
from pdi_pipeline.methods.registry import get_interpolator

setup_logging()
log = logging.getLogger(__name__)

PROJECT_ROOT = get_project_root()
DEFAULT_MANIFEST = PROJECT_ROOT / "preprocessed" / "manifest.csv"


def _load_entropy(
    base_dir: Path,
    split: str,
    satellite: str,
    patch_id: int,
    windows: list[int],
    clean: np.ndarray | None = None,
) -> dict[str, float]:
    """Load precomputed entropy values for a patch.

    Falls back to computing entropy on-the-fly from the clean image
    when precomputed files are not available.

    Returns a dict like {"entropy_7": 3.45, "entropy_15": 3.21, ...}.
    Returns NaN only when both precomputed file and clean image are
    unavailable.
    """
    from pdi_pipeline.entropy import shannon_entropy

    result: dict[str, float] = {}
    for ws in windows:
        key = f"entropy_{ws}"
        path = base_dir / split / satellite / f"{patch_id:07d}_entropy_{ws}.npy"
        if path.exists():
            arr = np.load(path)
            result[key] = float(np.mean(arr))
        elif clean is not None:
            ent_map = shannon_entropy(clean, window_size=ws)
            result[key] = float(np.mean(ent_map))
        else:
            result[key] = float("nan")
    return result


def _load_completed(output_path: Path) -> set[tuple[int, str, str, int]]:
    """Load already-completed (seed, noise, method, patch_id) tuples."""
    csv_path = output_path / "raw_results.csv"
    if not csv_path.exists():
        return set()
    df = pd.read_csv(
        csv_path,
        usecols=["seed", "noise_level", "method", "patch_id"],
        dtype={"noise_level": str},
    )
    return set(
        zip(
            df["seed"].astype(int),
            df["noise_level"].astype(str),
            df["method"],
            df["patch_id"].astype(int),
            strict=True,
        )
    )


def _ensure_entropy_precomputed(
    preprocessed_dir: Path,
    entropy_windows: list[int],
) -> str:
    manifest_path = preprocessed_dir / "manifest.csv"
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return "missing"

    cols = set(pd.read_csv(manifest_path, nrows=0).columns)
    missing = [ws for ws in entropy_windows if f"mean_entropy_{ws}" not in cols]
    if not missing:
        return "complete"

    log.info(
        "Missing mean entropy columns for windows %s. "
        "Running precompute_entropy with --resume.",
        missing,
    )

    import importlib

    precompute_entropy = importlib.import_module("scripts.precompute_entropy")

    args = [
        "--preprocessed-dir",
        str(preprocessed_dir),
        "--resume",
        "--windows",
        *[str(ws) for ws in entropy_windows],
    ]

    precompute_entropy.main(args)
    return "ran"


def _save_checkpoint(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Append rows to the CSV file in O(len(rows)) time."""
    if not rows:
        return
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "raw_results.csv"
    new_df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    new_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def _load_patch_selections(
    preprocessed_dir: Path,
) -> dict[str, dict[int, list[int]]] | None:
    """Load pre-computed patch selections from JSON.

    Returns ``{satellite: {seed: [patch_ids]}}`` or None if not found.
    """
    path = preprocessed_dir / "patch_selections.json"
    if not path.exists():
        return None
    with path.open() as fh:
        raw = json.load(fh)
    # Convert string keys back to int
    return {
        sat: {int(seed): ids for seed, ids in seed_map.items()}
        for sat, seed_map in raw.items()
    }


def _seed_sequences(
    seeds: list[int],
    satellites: list[str],
) -> dict[str, np.random.SeedSequence]:
    base = np.random.SeedSequence(seeds)
    spawned = base.spawn(len(satellites))
    return dict(zip(satellites, spawned, strict=True))


def _build_patch_plan(
    manifest_path: Path,
    config: ExperimentConfig,
    selections: dict[str, dict[int, list[int]]] | None,
) -> tuple[dict[str, list[int]] | None, dict[str, dict[int, int]]]:
    if not config.satellites:
        return None, {}

    seed_sequences = _seed_sequences(config.seeds, config.satellites)
    seed_map: dict[str, dict[int, int]] = {}

    if selections is not None:
        selected_by_sat: dict[str, list[int]] = {}
        for sat in config.satellites:
            ordered_ids: list[int] = []
            sat_seed_map: dict[int, int] = {}
            for seed in config.seeds:
                for patch_id in selections.get(sat, {}).get(seed, []):
                    if patch_id not in sat_seed_map:
                        sat_seed_map[patch_id] = seed
                        ordered_ids.append(patch_id)

            if (
                config.max_patches is not None
                and len(ordered_ids) > config.max_patches
            ):
                rng = np.random.default_rng(seed_sequences[sat])
                idx = rng.choice(
                    len(ordered_ids),
                    size=config.max_patches,
                    replace=False,
                )
                idx.sort()
                ordered_ids = [ordered_ids[i] for i in idx]
                sat_seed_map = {
                    patch_id: sat_seed_map[patch_id] for patch_id in ordered_ids
                }

            selected_by_sat[sat] = ordered_ids
            seed_map[sat] = sat_seed_map
        return selected_by_sat, seed_map

    selected_by_sat = {}
    seed_map = {}
    seed_default = config.seeds[0] if config.seeds else 0
    manifest = pd.read_csv(
        manifest_path,
        usecols=["patch_id", "satellite", "split"],
    )
    for sat in config.satellites:
        sat_rows = manifest[
            (manifest["split"] == "train") & (manifest["satellite"] == sat)
        ]
        patch_ids = sat_rows["patch_id"].astype(int).sort_values().tolist()
        if (
            config.max_patches is not None
            and len(patch_ids) > config.max_patches
        ):
            rng = np.random.default_rng(seed_sequences[sat])
            idx = rng.choice(
                len(patch_ids),
                size=config.max_patches,
                replace=False,
            )
            idx.sort()
            patch_ids = [patch_ids[i] for i in idx]
        selected_by_sat[sat] = patch_ids
        seed_map[sat] = dict.fromkeys(patch_ids, seed_default)

    return selected_by_sat, seed_map


def _dataset_for_combo(
    manifest_path: Path,
    config: ExperimentConfig,
    selected_ids: list[int] | None,
    *,
    seed: int,
    noise_level: str,
    satellite: str,
) -> PatchDataset:
    return PatchDataset(
        manifest_path,
        split="train",
        satellite=satellite,
        noise_level=noise_level,
        max_patches=config.max_patches,
        seed=seed,
        selected_ids=selected_ids,
    )


def _compute_total_work(
    manifest_path: Path,
    config: ExperimentConfig,
    selected_by_sat: dict[str, list[int]] | None,
) -> int:
    total = 0
    seed = config.seeds[0] if config.seeds else 0
    for noise_level in config.noise_levels:
        for satellite in config.satellites:
            selected_ids = (
                selected_by_sat.get(satellite) if selected_by_sat else None
            )
            dataset = _dataset_for_combo(
                manifest_path,
                config,
                selected_ids,
                seed=seed,
                noise_level=noise_level,
                satellite=satellite,
            )
            total += len(dataset) * len(config.methods)
    return total


def _build_method_instances(
    methods: list,
) -> dict[str, object]:
    instances: dict[str, object] = {}
    for method_cfg in methods:
        instances[method_cfg.name] = get_interpolator(
            method_cfg.name,
            **method_cfg.params,
        )
        log.info("Loaded method: %s", method_cfg.name)
    return instances


def _entropy_for_patch(
    patch: Any,
    entropy_keys: set[str],
    config: ExperimentConfig,
    preprocessed_dir: Path,
) -> dict[str, float]:
    if patch.mean_entropy is not None:
        entropy = {
            key: value
            for key, value in patch.mean_entropy.items()
            if key in entropy_keys
        }
        missing = [
            ws
            for ws in config.entropy_windows
            if f"entropy_{ws}" not in entropy
        ]
        if missing:
            entropy.update(
                _load_entropy(
                    preprocessed_dir,
                    patch.split,
                    patch.satellite,
                    patch.patch_id,
                    missing,
                    clean=patch.clean,
                )
            )
        return entropy

    return _load_entropy(
        preprocessed_dir,
        patch.split,
        patch.satellite,
        patch.patch_id,
        config.entropy_windows,
        clean=patch.clean,
    )


def _evaluate_patch(
    method: object,
    patch: Any,
    *,
    seed: int,
    noise_level: str,
    method_name: str,
    method_category: str,
    metrics_module: object,
    recon_manager: ReconstructionManager,
    preprocessed_dir: Path,
    config: ExperimentConfig,
    entropy_keys: set[str],
) -> dict[str, Any]:
    t_start = time.perf_counter()
    try:
        result = method.apply(patch.degraded, patch.mask)
        scores = metrics_module.compute_all(patch.clean, result, patch.mask)
        status = "ok"
        error_msg = ""

        recon_manager.maybe_save_entropy_extreme(
            noise_level,
            method_name,
            patch.patch_id,
            result,
            patch.clean,
            patch.degraded,
            patch.mask,
        )
        recon_manager.maybe_save_first(
            seed,
            noise_level,
            method_name,
            patch.patch_id,
            result,
        )
    except Exception as exc:
        log.exception(
            "Error: seed=%d noise=%s method=%s patch=%d",
            seed,
            noise_level,
            method_name,
            patch.patch_id,
        )
        scores = {
            "psnr": float("nan"),
            "ssim": float("nan"),
            "rmse": float("nan"),
            "sam": float("nan"),
        }
        status = "error"
        error_msg = str(exc)

    elapsed_s = time.perf_counter() - t_start
    entropy = _entropy_for_patch(patch, entropy_keys, config, preprocessed_dir)

    row: dict[str, Any] = {
        "seed": seed,
        "noise_level": noise_level,
        "method": method_name,
        "method_category": method_category,
        "patch_id": patch.patch_id,
        "satellite": patch.satellite,
        "gap_fraction": patch.gap_fraction,
        "status": status,
        "error_msg": error_msg,
        "elapsed_s": elapsed_s,
    }
    row.update(entropy)
    row.update(scores)
    return row


def _run_evaluation_loop(
    *,
    config: ExperimentConfig,
    manifest_path: Path,
    selected_by_sat: dict[str, list[int]] | None,
    seed_map: dict[str, dict[int, int]],
    methods: list[Any],
    method_instances: dict[str, Any],
    completed: set[tuple[Any, ...]],
    metrics_module: Any,
    recon_manager: ReconstructionManager,
    preprocessed_dir: Path,
    entropy_keys: set[str],
    output_path: Path,
    total_work: int,
    checkpoint_interval: int = 500,
) -> tuple[int, int]:
    """Run the inner evaluation loop. Returns (n_done, n_skipped)."""
    buffer: list[dict[str, Any]] = []
    n_done = 0
    n_skipped = 0

    with tqdm(total=total_work, desc="Experiment") as pbar:
        seed_default = config.seeds[0] if config.seeds else 0
        for noise_level in config.noise_levels:
            for sat in config.satellites:
                selected_ids = (
                    selected_by_sat.get(sat) if selected_by_sat else None
                )
                ds = _dataset_for_combo(
                    manifest_path,
                    config,
                    selected_ids,
                    seed=seed_default,
                    noise_level=noise_level,
                    satellite=sat,
                )

                for mc in methods:
                    method = method_instances[mc.name]

                    for patch in ds:
                        seed = seed_map.get(sat, {}).get(
                            patch.patch_id,
                            seed_default,
                        )
                        key = (
                            seed,
                            noise_level,
                            mc.name,
                            patch.patch_id,
                        )
                        if key in completed:
                            n_skipped += 1
                            pbar.update(1)
                            continue
                        row = _evaluate_patch(
                            method,
                            patch,
                            seed=seed,
                            noise_level=noise_level,
                            method_name=mc.name,
                            method_category=mc.category,
                            metrics_module=metrics_module,
                            recon_manager=recon_manager,
                            preprocessed_dir=preprocessed_dir,
                            config=config,
                            entropy_keys=entropy_keys,
                        )

                        buffer.append(row)
                        n_done += 1
                        pbar.update(1)

                        if len(buffer) >= checkpoint_interval:
                            _save_checkpoint(buffer, output_path)
                            buffer.clear()

    _save_checkpoint(buffer, output_path)
    buffer.clear()
    return n_done, n_skipped


def _log_failure_summary(output_path: Path) -> None:
    """Log a summary of failed evaluations."""
    csv_path = output_path / "raw_results.csv"
    if not csv_path.exists():
        return

    all_df = pd.read_csv(csv_path)
    if "status" not in all_df.columns:
        return

    errors = all_df[all_df["status"] == "error"]
    if errors.empty:
        return

    log.warning("--- Failure Summary: %d errors ---", len(errors))
    for method, group in errors.groupby("method"):
        log.warning("  %s: %d failures", method, len(group))


def run_experiment(
    config: ExperimentConfig,
    dry_run: bool = False,
    save_reconstructions: int = 0,
    save_entropy_top_k: int = 5,
) -> None:
    """Execute the full experiment loop."""
    from pdi_pipeline import metrics as m

    output_path = config.output_path
    setup_file_logging(output_path, name="experiment")
    manifest_path = DEFAULT_MANIFEST

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return

    # Load pre-computed patch selections (written by preprocess_dataset.py)
    preprocessed_dir = manifest_path.parent
    selections = _load_patch_selections(preprocessed_dir)
    if selections is not None:
        log.info(
            "Using pre-computed patch selections from patch_selections.json"
        )

    log.info(
        "Entropy precompute: start (windows=%s)",
        config.entropy_windows,
    )
    entropy_status = _ensure_entropy_precomputed(
        preprocessed_dir,
        config.entropy_windows,
    )
    if entropy_status == "ran":
        log.info("Entropy precompute: completed")
    elif entropy_status == "complete":
        log.info("Entropy precompute: already complete")
    else:
        log.info("Entropy precompute: skipped")

    selected_by_sat, seed_map = _build_patch_plan(
        manifest_path,
        config,
        selections,
    )

    methods = config.methods
    total_work = _compute_total_work(manifest_path, config, selected_by_sat)

    log.info("Experiment: %s", config.name)
    log.info("Seeds: %d", len(config.seeds))
    log.info("Seed usage: sampling only (no seed multiplication)")
    log.info("Noise levels: %s", config.noise_levels)
    log.info("Methods: %d (%s)", len(methods), [m.name for m in methods])
    log.info("Total evaluations: %d", total_work)
    log.info("Output: %s", output_path)

    if selected_by_sat is not None:
        log.info("Selection strategy: union across seeds, capped per satellite")

    if selected_by_sat is not None:
        for sat, patch_ids in selected_by_sat.items():
            log.info(
                "Selection: %s -> %d patches (max=%s)",
                sat,
                len(patch_ids),
                config.max_patches,
            )

    if dry_run:
        log.info("DRY RUN -- exiting without execution.")
        return

    # Load completed work for resuming
    completed = _load_completed(output_path)
    if completed:
        log.info("Resuming: %d evaluations already completed", len(completed))

    method_instances = _build_method_instances(methods)

    entropy_extremes = compute_entropy_extremes(
        manifest_path,
        config,
        selections,
        top_k=save_entropy_top_k,
    )
    write_entropy_extremes_manifest(output_path, entropy_extremes)

    recon_manager = ReconstructionManager(
        output_dir=output_path,
        entropy_extremes=entropy_extremes,
        save_first_n=save_reconstructions,
        first_seed=config.seeds[0] if config.seeds else None,
    )

    t0 = time.monotonic()
    entropy_keys = {f"entropy_{ws}" for ws in config.entropy_windows}

    n_done, n_skipped = _run_evaluation_loop(
        config=config,
        manifest_path=manifest_path,
        selected_by_sat=selected_by_sat,
        seed_map=seed_map,
        methods=methods,
        method_instances=method_instances,
        completed=completed,
        metrics_module=m,
        recon_manager=recon_manager,
        preprocessed_dir=preprocessed_dir,
        entropy_keys=entropy_keys,
        output_path=output_path,
        total_work=total_work,
    )

    _log_failure_summary(output_path)

    elapsed = time.monotonic() - t0
    log.info("--- Experiment Complete ---")
    log.info("Evaluations computed: %d", n_done)
    log.info("Evaluations skipped (resume): %d", n_skipped)
    log.info("Elapsed: %.1fs", elapsed)
    log.info("Results: %s", output_path / "raw_results.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the gap-filling experiment pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick_validation.yaml config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without executing.",
    )
    parser.add_argument(
        "--save-reconstructions",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Save first N reconstruction arrays per method "
            "per noise level (first seed)."
        ),
    )
    parser.add_argument(
        "--save-entropy-top-k",
        type=int,
        default=5,
        metavar="K",
        help=(
            "Save top-K high and low entropy patches per noise level "
            "(K=0 disables)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config_dir = PROJECT_ROOT / "config"

    if args.quick:
        config_path = config_dir / "quick_validation.yaml"
    elif args.config is not None:
        config_path = args.config
    else:
        config_path = config_dir / "paper_results.yaml"

    log.info("Loading config: %s", config_path)
    config = load_config(config_path)
    run_experiment(
        config,
        dry_run=args.dry_run,
        save_reconstructions=args.save_reconstructions,
        save_entropy_top_k=args.save_entropy_top_k,
    )


if __name__ == "__main__":
    main()
