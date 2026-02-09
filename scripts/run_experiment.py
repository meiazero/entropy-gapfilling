"""Run the gap-filling experiment pipeline.

Iterates over seeds x noise levels x methods x patches, computes metrics
on each reconstruction, and writes results to a Parquet file with
checkpointing for resumable execution.

Usage:
    uv run python scripts/run_experiment.py --config config/paper_results.yaml
    uv run python scripts/run_experiment.py --quick
    uv run python scripts/run_experiment.py --quick --dry-run
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from pdi_pipeline.config import ExperimentConfig, MethodConfig, load_config
from pdi_pipeline.dataset import PatchDataset
from pdi_pipeline.methods.base import BaseMethod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "preprocessed" / "manifest.csv"

# Multi-temporal methods excluded from this paper
EXCLUDED_METHODS = {
    "temporal_spline",
    "temporal_fourier",
    "kriging_spacetime",
    "dineof",
}

# Registry: config name -> (class, default kwargs)
_METHOD_REGISTRY: dict[str, type[BaseMethod]] = {}


def _build_registry() -> dict[str, type[BaseMethod]]:
    """Lazy-build the method registry from the methods package."""
    if _METHOD_REGISTRY:
        return _METHOD_REGISTRY

    from pdi_pipeline.methods import (
        BicubicInterpolator,
        BilinearInterpolator,
        DCTInpainting,
        ExemplarBasedInterpolator,
        IDWInterpolator,
        KrigingInterpolator,
        L1DCTInpainting,
        L1WaveletInpainting,
        LanczosInterpolator,
        NearestInterpolator,
        NonLocalMeansInterpolator,
        RBFInterpolator,
        SplineInterpolator,
        TVInpainting,
        WaveletInpainting,
    )

    classes: list[type[BaseMethod]] = [
        NearestInterpolator,
        BilinearInterpolator,
        BicubicInterpolator,
        LanczosInterpolator,
        IDWInterpolator,
        RBFInterpolator,
        SplineInterpolator,
        KrigingInterpolator,
        DCTInpainting,
        WaveletInpainting,
        TVInpainting,
        L1DCTInpainting,
        L1WaveletInpainting,
        NonLocalMeansInterpolator,
        ExemplarBasedInterpolator,
    ]

    # Map both class.name and config aliases
    config_aliases = {
        "l1_dct": "cs_dct",
        "l1_wavelet": "cs_wavelet",
        "non_local": "non_local",
        "exemplar_based": "exemplar_based",
    }

    for cls in classes:
        _METHOD_REGISTRY[cls.name] = cls

    for alias, real_name in config_aliases.items():
        if real_name in _METHOD_REGISTRY:
            _METHOD_REGISTRY[alias] = _METHOD_REGISTRY[real_name]

    return _METHOD_REGISTRY


def instantiate_method(cfg: MethodConfig) -> BaseMethod:
    """Instantiate a method from its config."""
    registry = _build_registry()
    name = cfg.name
    if name not in registry:
        msg = f"Unknown method: {name!r}. Available: {sorted(registry.keys())}"
        raise ValueError(msg)
    cls = registry[name]
    return cls(**cfg.params)


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
        csv_path, usecols=["seed", "noise_level", "method", "patch_id"]
    )
    return {
        (int(r["seed"]), r["noise_level"], r["method"], int(r["patch_id"]))
        for _, r in df.iterrows()
    }


def _save_checkpoint(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Append rows to the CSV file."""
    if not rows:
        return
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "raw_results.csv"
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)


def _setup_file_logging(log_path: Path) -> None:
    """Add a file handler to the root logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


def run_experiment(
    config: ExperimentConfig,
    dry_run: bool = False,
    save_reconstructions: int = 0,
) -> None:
    """Execute the full experiment loop."""
    from pdi_pipeline import metrics as m

    output_path = config.output_path
    _setup_file_logging(output_path / "experiment.log")
    manifest_path = DEFAULT_MANIFEST

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        return

    # Filter out multi-temporal methods
    methods = [mc for mc in config.methods if mc.name not in EXCLUDED_METHODS]

    # Count total work
    total_patches = 0
    for noise_level in config.noise_levels:
        for sat in config.satellites:
            ds = PatchDataset(
                manifest_path,
                split="test",
                satellite=sat,
                noise_level=noise_level,
                max_patches=config.max_patches,
            )
            total_patches += len(ds)

    total_work = len(config.seeds) * len(methods) * total_patches

    log.info("Experiment: %s", config.name)
    log.info("Seeds: %d", len(config.seeds))
    log.info("Noise levels: %s", config.noise_levels)
    log.info("Methods: %d (%s)", len(methods), [m.name for m in methods])
    log.info("Total patches per noise level: %d", total_patches)
    log.info("Total evaluations: %d", total_work)
    log.info("Output: %s", output_path)

    if dry_run:
        log.info("DRY RUN -- exiting without execution.")
        return

    # Load completed work for resuming
    completed = _load_completed(output_path)
    if completed:
        log.info("Resuming: %d evaluations already completed", len(completed))

    # Instantiate all methods
    method_instances = {}
    for mc in methods:
        method_instances[mc.name] = instantiate_method(mc)
        log.info("Loaded method: %s", mc.name)

    checkpoint_interval = 500
    buffer: list[dict[str, Any]] = []
    n_done = 0
    n_skipped = 0
    t0 = time.monotonic()

    preprocessed_dir = manifest_path.parent

    with tqdm(total=total_work, desc="Experiment") as pbar:
        for seed in config.seeds:
            rng = np.random.default_rng(seed)
            _ = rng  # seed reserved for future stochastic methods

            for noise_level in config.noise_levels:
                for sat in config.satellites:
                    ds = PatchDataset(
                        manifest_path,
                        split="test",
                        satellite=sat,
                        noise_level=noise_level,
                        max_patches=config.max_patches,
                        seed=seed,
                    )

                    for mc in methods:
                        method = method_instances[mc.name]

                        for patch in ds:
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

                            try:
                                result = method.apply(
                                    patch.degraded, patch.mask
                                )
                                scores = m.compute_all(
                                    patch.clean, result, patch.mask
                                )
                                status = "ok"
                                error_msg = ""

                                # Save reconstructions (first seed, no noise only)
                                if (
                                    save_reconstructions > 0
                                    and seed == config.seeds[0]
                                    and noise_level == "inf"
                                ):
                                    recon_dir = (
                                        output_path
                                        / "reconstructions"
                                        / mc.name
                                    )
                                    existing = (
                                        len(list(recon_dir.glob("*.npy")))
                                        if recon_dir.exists()
                                        else 0
                                    )
                                    if existing < save_reconstructions:
                                        recon_dir.mkdir(
                                            parents=True, exist_ok=True
                                        )
                                        np.save(
                                            recon_dir
                                            / f"{patch.patch_id:07d}.npy",
                                            result,
                                        )

                            except Exception as exc:
                                log.exception(
                                    "Error: seed=%d noise=%s method=%s "
                                    "patch=%d",
                                    seed,
                                    noise_level,
                                    mc.name,
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

                            entropy = _load_entropy(
                                preprocessed_dir,
                                patch.split,
                                patch.satellite,
                                patch.patch_id,
                                config.entropy_windows,
                                clean=patch.clean,
                            )

                            row: dict[str, Any] = {
                                "seed": seed,
                                "noise_level": noise_level,
                                "method": mc.name,
                                "method_category": mc.category,
                                "patch_id": patch.patch_id,
                                "satellite": patch.satellite,
                                "gap_fraction": patch.gap_fraction,
                                "status": status,
                                "error_msg": error_msg,
                            }
                            row.update(entropy)
                            row.update(scores)

                            buffer.append(row)
                            n_done += 1
                            pbar.update(1)

                            if len(buffer) >= checkpoint_interval:
                                _save_checkpoint(buffer, output_path)
                                buffer.clear()

    # Final flush
    _save_checkpoint(buffer, output_path)
    buffer.clear()

    # Log failure summary
    csv_path = output_path / "raw_results.csv"
    if csv_path.exists():
        all_df = pd.read_csv(csv_path)
        if "status" in all_df.columns:
            errors = all_df[all_df["status"] == "error"]
            if not errors.empty:
                log.warning("--- Failure Summary: %d errors ---", len(errors))
                for method, group in errors.groupby("method"):
                    log.warning("  %s: %d failures", method, len(group))

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
        help="Save first N reconstructed arrays per method (first seed, no noise).",
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
    )


if __name__ == "__main__":
    main()
