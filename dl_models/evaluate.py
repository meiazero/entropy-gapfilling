"""Standalone evaluation script for DL inpainting models.

Loads a trained model checkpoint, runs inference on the test set,
computes the full metric set (PSNR, SSIM, RMSE, SAM, ERGAS, per-band
RMSE, pixel accuracy and F1 at three thresholds), joins entropy values
from the manifest, and saves per-patch results to a CSV.

The output schema matches raw_results.csv from the classical pipeline so
both can be fed into aggregate_results.py without transformation.

Usage:
    uv run python -m dl_models.evaluate \
        --model ae \
        --checkpoint results/dl_models/checkpoints/ae_best.pth \
        --manifest preprocessed/manifest.csv \
        --output results/dl_eval/ \
        --noise-level inf \
        --satellite sentinel2

Available models: ae, vae, gan, unet, vit
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dl_models.shared import metrics as m
from dl_models.shared.dataset import InpaintingDataset
from dl_models.shared.trainer import setup_file_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "ae": ("dl_models.ae.model", "AEInpainting"),
    "vae": ("dl_models.vae.model", "VAEInpainting"),
    "gan": ("dl_models.gan.model", "GANInpainting"),
    "unet": ("dl_models.unet.model", "UNetInpainting"),
    "vit": ("dl_models.vit.model", "ViTInpainting"),
}

# DL architecture category labels (for combined comparison tables)
MODEL_ARCHITECTURE: dict[str, str] = {
    "ae": "bottleneck",
    "vae": "bottleneck",
    "gan": "skip_connection",
    "unet": "skip_connection",
    "vit": "attention",
}

# Noise-level key -> manifest column
_NOISE_PATH_COLS: dict[str, str] = {
    "inf": "degraded_inf_path",
    "40": "degraded_40_path",
    "30": "degraded_30_path",
    "20": "degraded_20_path",
}

_ALL_METRIC_KEYS: list[str] = [
    "psnr",
    "ssim",
    "rmse",
    "sam",
    "ergas",
    "rmse_b0",
    "rmse_b1",
    "rmse_b2",
    "rmse_b3",
    "pixel_acc_002",
    "pixel_acc_005",
    "pixel_acc_01",
    "f1_002",
    "f1_005",
    "f1_01",
]


def _load_model(
    model_key: str,
    checkpoint: Path,
    device: str | None,
) -> Any:
    """Dynamically load and instantiate a DL model."""
    if model_key not in MODEL_REGISTRY:
        msg = (
            f"Unknown model: {model_key!r}. "
            f"Available: {sorted(MODEL_REGISTRY.keys())}"
        )
        raise ValueError(msg)

    module_path, class_name = MODEL_REGISTRY[model_key]

    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    kwargs: dict[str, Any] = {"checkpoint_path": str(checkpoint)}
    if device:
        kwargs["device"] = device

    return cls(**kwargs)


def _load_entropy_map(
    manifest_path: Path,
    satellite: str,
) -> pd.DataFrame:
    """Load patch_id -> entropy + gap_fraction lookup from manifest."""
    cols = {"patch_id", "satellite", "gap_fraction"}
    entropy_candidates = [f"mean_entropy_{ws}" for ws in (7, 15, 31)]
    all_cols_raw = pd.read_csv(manifest_path, nrows=0).columns.tolist()
    entropy_cols = [c for c in entropy_candidates if c in all_cols_raw]
    usecols = sorted(cols | set(entropy_cols))

    df = pd.read_csv(manifest_path, usecols=usecols)
    df = df[df["satellite"] == satellite].set_index("patch_id")
    return df


def _build_error_row() -> dict[str, float]:
    """Return NaN row for all metrics."""
    return {k: float("nan") for k in _ALL_METRIC_KEYS}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a DL inpainting model on the test set.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dl_eval"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--satellite",
        type=str,
        default="sentinel2",
        help="Satellite filter for the test set.",
    )
    parser.add_argument(
        "--noise-level",
        type=str,
        default="inf",
        choices=sorted(_NOISE_PATH_COLS.keys()),
        help="Noise level variant to evaluate (inf = no noise).",
    )
    parser.add_argument(
        "--save-reconstructions",
        type=int,
        default=0,
        metavar="N",
        help="Save first N reconstructed arrays as .npy files.",
    )
    return parser


def _save_reconstruction(
    output_dir: Path,
    patch_id: int,
    result: np.ndarray,
) -> None:
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    np.save(recon_dir / f"{patch_id:07d}.npy", result)


def _log_summary(df: pd.DataFrame, model_name: str) -> None:
    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        log.info("--- Results for %s ---", model_name)
        for metric in ["psnr", "ssim", "rmse", "sam", "ergas"]:
            if metric in ok_df.columns:
                finite = ok_df[metric].dropna()
                if not finite.empty:
                    log.info(
                        "  %s: %.4f +/- %.4f",
                        metric.upper(),
                        finite.mean(),
                        finite.std(),
                    )
        if "elapsed_s" in ok_df.columns:
            log.info(
                "  TIME: %.4fs +/- %.4fs",
                ok_df["elapsed_s"].mean(),
                ok_df["elapsed_s"].std(),
            )

    n_errors = len(df[df["status"] == "error"])
    if n_errors > 0:
        log.warning("Errors: %d / %d", n_errors, len(df))


def main() -> None:  # noqa: C901
    args = _build_parser().parse_args()
    if not args.checkpoint.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        return

    output_dir = args.output / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(output_dir / "evaluate.log")

    log.info("Loading model: %s", args.model)
    model = _load_model(args.model, args.checkpoint, args.device)
    log.info(
        "Model loaded: %s | noise_level=%s | satellite=%s",
        model.name,
        args.noise_level,
        args.satellite,
    )

    # Load entropy + gap_fraction lookup keyed by patch_id.
    entropy_df = _load_entropy_map(args.manifest, args.satellite)
    has_entropy_7 = "mean_entropy_7" in entropy_df.columns
    has_entropy_15 = "mean_entropy_15" in entropy_df.columns
    has_entropy_31 = "mean_entropy_31" in entropy_df.columns

    test_ds = InpaintingDataset(
        args.manifest,
        split="test",
        satellite=args.satellite,
        max_patches=args.max_patches,
    )
    log.info("Test patches: %d", len(test_ds))

    architecture = MODEL_ARCHITECTURE.get(args.model, "unknown")
    rows: list[dict[str, Any]] = []
    n_saved = 0
    t0 = time.monotonic()

    for idx in range(len(test_ds)):
        x, clean_t, mask_t = test_ds[idx]
        clean_np = clean_t.permute(1, 2, 0).numpy()
        mask_np = mask_t.numpy()
        channels = clean_t.shape[0]
        degraded_np = x[:channels].permute(1, 2, 0).numpy()
        patch_id = int(test_ds.patch_ids[idx])

        # Compute gap_fraction from mask (fraction of pixels that are gaps).
        gap_fraction = float(np.mean(mask_np > 0.5))

        # Join entropy from manifest.
        entropy_7 = entropy_15 = entropy_31 = float("nan")
        if patch_id in entropy_df.index:
            row_meta = entropy_df.loc[patch_id]
            if has_entropy_7:
                entropy_7 = float(row_meta["mean_entropy_7"])
            if has_entropy_15:
                entropy_15 = float(row_meta["mean_entropy_15"])
            if has_entropy_31:
                entropy_31 = float(row_meta["mean_entropy_31"])
            # Override gap_fraction with manifest value if available.
            if "gap_fraction" in row_meta.index and not pd.isna(
                row_meta["gap_fraction"]
            ):
                gap_fraction = float(row_meta["gap_fraction"])

        t_start = time.perf_counter()
        should_save = (
            args.save_reconstructions > 0
            and n_saved < args.save_reconstructions
        )

        try:
            result = model.apply(degraded_np, mask_np)
            scores = m.compute_all(clean_np, result, mask_np)
            status = "ok"
            error_msg = ""
            if should_save:
                _save_reconstruction(output_dir, patch_id, result)
                n_saved += 1
        except Exception as exc:
            log.exception("Error on patch %d", patch_id)
            scores = _build_error_row()
            status = "error"
            error_msg = str(exc)

        elapsed_s = time.perf_counter() - t_start

        row: dict[str, Any] = {
            "model": model.name,
            "architecture": architecture,
            "satellite": args.satellite,
            "noise_level": args.noise_level,
            "patch_id": patch_id,
            "gap_fraction": gap_fraction,
            "entropy_7": entropy_7,
            "entropy_15": entropy_15,
            "entropy_31": entropy_31,
            "status": status,
            "error_msg": error_msg,
            "elapsed_s": elapsed_s,
        }
        row.update(scores)
        rows.append(row)

        if (idx + 1) % 100 == 0:
            log.info("Processed %d / %d patches", idx + 1, len(test_ds))

    elapsed = time.monotonic() - t0
    df = pd.DataFrame(rows)

    # Use noise_level suffix so results for different noise levels don't
    # overwrite each other within the same model output directory.
    csv_path = output_dir / f"eval_{args.noise_level}.csv"
    df.to_csv(csv_path, index=False)

    _log_summary(df, model.name)
    log.info("Elapsed: %.1fs | saved to: %s", elapsed, csv_path)


if __name__ == "__main__":
    main()
