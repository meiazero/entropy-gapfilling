"""Standalone evaluation script for DL inpainting models.

Loads a trained model checkpoint, runs inference on the test set,
computes quality metrics, and saves results. Works independently
from the main experiment pipeline.

Usage:
    uv run python src/dl-models/evaluate.py \
        --model ae \
        --checkpoint src/dl-models/checkpoints/ae_best.pt \
        --manifest preprocessed/manifest.csv \
        --output results/dl_eval/

Available models: ae, vae, gan, transformer
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

_DL_ROOT = Path(__file__).resolve().parent
if str(_DL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DL_ROOT))

# Also add src/ for pdi_pipeline imports
_SRC_ROOT = _DL_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import numpy as np
import pandas as pd
from shared.dataset import InpaintingDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "ae": ("ae.model", "AEInpainting"),
    "vae": ("vae.model", "VAEInpainting"),
    "gan": ("gan.model", "GANInpainting"),
    "transformer": ("transformer.model", "TransformerInpainting"),
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

    log.info("Loading model: %s", args.model)
    model = _load_model(args.model, args.checkpoint, args.device)
    log.info("Model loaded: %s", model.name)

    from pdi_pipeline import metrics as m

    test_ds = InpaintingDataset(
        args.manifest,
        split="test",
        max_patches=args.max_patches,
    )
    log.info("Test patches: %d", len(test_ds))

    output_dir = args.output / model.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    n_saved = 0
    t0 = time.monotonic()

    for idx in range(len(test_ds)):
        sample = test_ds._ds[idx]

        try:
            result = model.apply(sample.degraded, sample.mask)
            scores = m.compute_all(sample.clean, result, sample.mask)
            status = "ok"
            error_msg = ""

            if (
                args.save_reconstructions > 0
                and n_saved < args.save_reconstructions
            ):
                recon_dir = output_dir / "reconstructions"
                recon_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    recon_dir / f"{sample.patch_id:07d}.npy",
                    result,
                )
                n_saved += 1

        except Exception as exc:
            log.exception("Error on patch %d", sample.patch_id)
            scores = {
                "psnr": float("nan"),
                "ssim": float("nan"),
                "rmse": float("nan"),
                "sam": float("nan"),
            }
            status = "error"
            error_msg = str(exc)

        row: dict[str, Any] = {
            "model": model.name,
            "patch_id": sample.patch_id,
            "satellite": sample.satellite,
            "gap_fraction": sample.gap_fraction,
            "status": status,
            "error_msg": error_msg,
        }
        row.update(scores)
        rows.append(row)

        if (idx + 1) % 100 == 0:
            log.info("Processed %d / %d patches", idx + 1, len(test_ds))

    elapsed = time.monotonic() - t0
    df = pd.DataFrame(rows)
    parquet_path = output_dir / "results.parquet"
    df.to_parquet(parquet_path, index=False)

    # Summary
    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        log.info("--- Results for %s ---", model.name)
        for metric in ["psnr", "ssim", "rmse", "sam"]:
            if metric in ok_df.columns:
                log.info(
                    "  %s: %.4f +/- %.4f",
                    metric.upper(),
                    ok_df[metric].mean(),
                    ok_df[metric].std(),
                )

    n_errors = len(df[df["status"] == "error"])
    if n_errors > 0:
        log.warning("Errors: %d / %d", n_errors, len(df))

    log.info("Elapsed: %.1fs", elapsed)
    log.info("Results saved to: %s", parquet_path)


if __name__ == "__main__":
    main()
