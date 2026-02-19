"""Standalone evaluation script for DL inpainting models.

Loads a trained model checkpoint, runs inference on the test set,
computes quality metrics, and saves results. Fully isolated from
pdi_pipeline.

Usage:
    uv run python -m dl_models.evaluate \
        --model ae \
        --checkpoint dl_models/checkpoints/ae_best.pth \
        --manifest preprocessed/manifest.csv \
        --output results/dl_eval/

Available models: ae, vae, gan, unet, transformer
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
    "transformer": ("dl_models.transformer.model", "TransformerInpainting"),
    "unet_jax": ("dl_models.unet_jax.model", "UNetInpaintingJAX"),
}


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


def main() -> None:
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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--satellite", type=str, default="sentinel2")
    parser.add_argument(
        "--save-reconstructions",
        type=int,
        default=0,
        metavar="N",
        help="Save first N reconstructed arrays.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        return

    output_dir = args.output / args.model
    setup_file_logging(output_dir / "evaluate.log")

    log.info("Loading model: %s", args.model)
    model = _load_model(args.model, args.checkpoint, args.device)
    log.info("Model loaded: %s", model.name)

    test_ds = InpaintingDataset(
        args.manifest,
        split="test",
        satellite=args.satellite,
        max_patches=args.max_patches,
    )
    log.info("Test patches: %d", len(test_ds))

    output_dir = args.output / model.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_saved = 0
    t0 = time.monotonic()

    for idx in range(len(test_ds)):
        x, clean_t, mask_t = test_ds[idx]

        # Reconstruct numpy arrays for apply()
        # clean_t: (C, H, W), mask_t: (H, W)
        clean_np = clean_t.permute(1, 2, 0).numpy()
        mask_np = mask_t.numpy()
        # Input has C+1 channels; first C are degraded*~mask
        c = clean_t.shape[0]
        degraded_np = x[:c].permute(1, 2, 0).numpy()

        patch_id = int(test_ds.patch_ids[idx])

        t_start = time.perf_counter()
        try:
            result = model.apply(degraded_np, mask_np)
            scores = m.compute_all(clean_np, result, mask_np)
            status = "ok"
            error_msg = ""

            if (
                args.save_reconstructions > 0
                and n_saved < args.save_reconstructions
            ):
                recon_dir = output_dir / "reconstructions"
                recon_dir.mkdir(parents=True, exist_ok=True)
                np.save(recon_dir / f"{patch_id:07d}.npy", result)
                n_saved += 1

        except Exception as exc:
            log.exception("Error on patch %d", patch_id)
            scores = {
                "psnr": float("nan"),
                "ssim": float("nan"),
                "rmse": float("nan"),
            }
            status = "error"
            error_msg = str(exc)
        elapsed_s = time.perf_counter() - t_start

        row: dict[str, Any] = {
            "model": model.name,
            "patch_id": patch_id,
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
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        log.info("--- Results for %s ---", model.name)
        for metric in ["psnr", "ssim", "rmse"]:
            if metric in ok_df.columns:
                log.info(
                    "  %s: %.4f +/- %.4f",
                    metric.upper(),
                    ok_df[metric].mean(),
                    ok_df[metric].std(),
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

    log.info("Elapsed: %.1fs", elapsed)
    log.info("Results saved to: %s", csv_path)


if __name__ == "__main__":
    main()
