"""Grid search runner for DL inpainting models.

Runs training and evaluation for each hyperparameter combination and
writes a consolidated CSV with execution time and quality metrics.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pdi_pipeline.logging_utils import setup_file_logging, setup_logging

log = logging.getLogger(__name__)

MODEL_TRAIN_MODULES = {
    "ae": "dl_models.ae.train",
    "vae": "dl_models.vae.train",
    "gan": "dl_models.gan.train",
    "unet": "dl_models.unet.train",
    "vit": "dl_models.vit.train",
}

MODEL_ALLOWED_ARGS = {
    "ae": {
        "manifest",
        "output",
        "epochs",
        "batch_size",
        "lr",
        "device",
        "patience",
        "satellite",
        "num_workers",
    },
    "vae": {
        "manifest",
        "output",
        "epochs",
        "batch_size",
        "lr",
        "beta",
        "device",
        "patience",
        "satellite",
        "num_workers",
    },
    "gan": {
        "manifest",
        "output",
        "epochs",
        "batch_size",
        "lr",
        "device",
        "patience",
        "lambda_l1",
        "lambda_adv",
        "satellite",
        "num_workers",
    },
    "unet": {
        "manifest",
        "output",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "device",
        "patience",
        "satellite",
        "num_workers",
    },
    "vit": {
        "manifest",
        "output",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "device",
        "patience",
        "satellite",
        "num_workers",
    },
}


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "run"


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _expand_grid(params: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not params:
        return [{}]
    keys = list(params.keys())
    values: list[list[Any]] = [params[k] for k in keys]
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo, strict=True)))
    return combos


def _merge_grids(
    common: dict[str, list[Any]] | None,
    specific: dict[str, list[Any]] | None,
) -> dict[str, list[Any]]:
    merged: dict[str, list[Any]] = {}
    if common:
        merged.update(common)
    if specific:
        merged.update(specific)
    return merged


def _args_to_argv(args_map: dict[str, Any]) -> list[str]:
    argv = ["train"]
    for key, value in args_map.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        argv.extend([flag, str(value)])
    return argv


@contextmanager
def _patched_argv(argv: list[str]) -> Any:
    original = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Grid config not found: {path}"
        raise FileNotFoundError(msg)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        msg = "Grid config must be a mapping"
        raise TypeError(msg)
    return data


def _load_grid_from_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Config not found: {path}"
        raise FileNotFoundError(msg)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        msg = "Config must be a mapping"
        raise TypeError(msg)
    grid = data.get("dl_grid_search")
    if not isinstance(grid, dict):
        msg = "Config missing 'dl_grid_search' mapping"
        raise TypeError(msg)
    return grid


def _resolve_grid(args: argparse.Namespace) -> dict[str, Any]:
    if args.grid:
        return _load_yaml(args.grid)
    if args.config is None:
        msg = "--config is required when --grid is not provided"
        raise ValueError(msg)
    return _load_grid_from_config(args.config)


def _run_grid_search(grid: dict[str, Any], args: argparse.Namespace) -> None:
    manifest = Path(grid.get("manifest", "preprocessed/manifest.csv"))
    output_root = Path(grid.get("output_dir", "results/grid_search/dl"))
    device = grid.get("device")
    satellite = grid.get("satellite", "sentinel2")
    num_workers = grid.get("num_workers")
    max_patches = grid.get("max_patches")

    models_grid = grid.get("models")
    if not isinstance(models_grid, dict) or not models_grid:
        msg = "Grid config must include a non-empty 'models' mapping"
        raise ValueError(msg)

    common_grid = grid.get("common")

    for model_name, model_grid in models_grid.items():
        if model_name not in MODEL_TRAIN_MODULES:
            msg = f"Unknown model in grid: {model_name}"
            raise ValueError(msg)

        merged = _merge_grids(common_grid, model_grid)
        param_combos = _expand_grid(merged)

        for params in param_combos:
            param_parts = [
                f"{key}={_format_value(value)}" for key, value in params.items()
            ]
            tag = _slugify("_".join(param_parts)) if param_parts else "default"
            run_name = _slugify(f"{model_name}_{tag}")

            run_dir = output_root / model_name / run_name
            checkpoints_dir = run_dir / "checkpoints"
            checkpoint_path = checkpoints_dir / f"{model_name}_best.pth"

            if args.skip_existing and checkpoint_path.exists():
                log.info("Skip existing run: %s", run_name)
                continue

            setup_file_logging(run_dir, name="grid_search")

            args_map: dict[str, Any] = {
                "manifest": manifest,
                "output": checkpoint_path,
                "device": device,
                "satellite": satellite,
                "num_workers": num_workers,
            }
            args_map.update(params)
            _validate_args(model_name, args_map)

            argv = _args_to_argv(args_map)
            log.info("Training %s: %s", model_name, run_name)
            log.info("Args: %s", args_map)

            t0 = time.monotonic()
            train_module = importlib.import_module(
                MODEL_TRAIN_MODULES[model_name]
            )
            with _patched_argv(argv):
                train_module.main()
            train_time_s = time.monotonic() - t0

            status = "ok" if checkpoint_path.exists() else "error"

            eval_time_s = 0.0
            eval_summary: dict[str, Any] = {}
            if status == "ok" and not args.skip_eval:
                eval_dir = run_dir / "eval"
                eval_module = importlib.import_module("dl_models.evaluate")
                eval_argv = [
                    "evaluate",
                    "--model",
                    model_name,
                    "--checkpoint",
                    str(checkpoint_path),
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(eval_dir),
                    "--satellite",
                    str(satellite),
                ]
                if device:
                    eval_argv.extend(["--device", str(device)])
                if max_patches is not None:
                    eval_argv.extend(["--max-patches", str(max_patches)])

                t1 = time.monotonic()
                with _patched_argv(eval_argv):
                    eval_module.main()
                eval_time_s = time.monotonic() - t1

                eval_results_path = eval_dir / model_name / "results.csv"
                eval_summary = _summarize_eval(eval_results_path)

            history = _load_history(run_dir, model_name)
            best = _best_epoch(history) if history else None

            summary: dict[str, Any] = {
                "run_name": run_name,
                "model": model_name,
                "params": json.dumps(params, sort_keys=True),
                "status": status,
                "train_time_s": float(train_time_s),
                "eval_time_s": float(eval_time_s),
                "total_time_s": float(train_time_s + eval_time_s),
            }

            if best:
                summary.update({
                    "best_epoch": int(best.get("epoch", 0)),
                    "best_val_loss": float(best.get("val_loss", float("nan"))),
                    "best_val_psnr": float(best.get("val_psnr", float("nan"))),
                    "best_val_ssim": float(best.get("val_ssim", float("nan"))),
                    "best_val_rmse": float(best.get("val_rmse", float("nan"))),
                    "best_val_pixel_acc_002": float(
                        best.get("val_pixel_acc_002", float("nan"))
                    ),
                    "best_val_f1_002": float(
                        best.get("val_f1_002", float("nan"))
                    ),
                    "best_val_pixel_acc_005": float(
                        best.get("val_pixel_acc_005", float("nan"))
                    ),
                    "best_val_f1_005": float(
                        best.get("val_f1_005", float("nan"))
                    ),
                    "best_val_pixel_acc_01": float(
                        best.get("val_pixel_acc_01", float("nan"))
                    ),
                    "best_val_f1_01": float(
                        best.get("val_f1_01", float("nan"))
                    ),
                })

            summary.update(eval_summary)
            _append_summary(output_root, summary)

    log.info(
        "Grid search complete. Summary: %s",
        output_root / "grid_results.csv",
    )


def _validate_args(model: str, args_map: dict[str, Any]) -> None:
    allowed = MODEL_ALLOWED_ARGS[model]
    unknown = [k for k in args_map if k not in allowed]
    if unknown:
        msg = f"Unsupported args for {model}: {sorted(unknown)}"
        raise ValueError(msg)


def _load_history(run_dir: Path, model: str) -> dict[str, Any] | None:
    history_path = run_dir / f"{model}_history.json"
    if not history_path.exists():
        return None
    return json.loads(history_path.read_text())


def _best_epoch(history: dict[str, Any]) -> dict[str, Any] | None:
    epochs = history.get("epochs", []) if history else []
    if not epochs:
        return None
    return min(epochs, key=lambda row: row.get("val_loss", float("inf")))


def _summarize_eval(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {
            "eval_n_rows": 0,
            "eval_psnr_mean": float("nan"),
            "eval_ssim_mean": float("nan"),
            "eval_rmse_mean": float("nan"),
            "eval_elapsed_mean": float("nan"),
        }
    df = pd.read_csv(results_path)
    ok_df = df[df["status"] == "ok"] if "status" in df.columns else df
    if ok_df.empty:
        return {
            "eval_n_rows": len(df),
            "eval_psnr_mean": float("nan"),
            "eval_ssim_mean": float("nan"),
            "eval_rmse_mean": float("nan"),
            "eval_elapsed_mean": float("nan"),
        }
    return {
        "eval_n_rows": len(df),
        "eval_psnr_mean": float(ok_df["psnr"].mean())
        if "psnr" in ok_df.columns
        else float("nan"),
        "eval_ssim_mean": float(ok_df["ssim"].mean())
        if "ssim" in ok_df.columns
        else float("nan"),
        "eval_rmse_mean": float(ok_df["rmse"].mean())
        if "rmse" in ok_df.columns
        else float("nan"),
        "eval_elapsed_mean": float(ok_df["elapsed_s"].mean())
        if "elapsed_s" in ok_df.columns
        else float("nan"),
    }


def _append_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "grid_results.csv"
    df = pd.DataFrame([summary])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search for DL inpainting models.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Experiment config YAML (uses dl_grid_search section).",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        required=False,
        help="Optional grid search YAML (overrides config dl_grid_search).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs with existing checkpoints.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip test evaluation after training.",
    )
    return parser


def main() -> None:
    setup_logging()
    args = _build_parser().parse_args()
    grid = _resolve_grid(args)
    _run_grid_search(grid, args)


if __name__ == "__main__":
    main()
