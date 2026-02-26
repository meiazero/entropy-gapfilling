"""Launch a single DL model training run from YAML config.

Reads the ``dl_training`` section of an experiment config and dispatches
to the appropriate ``dl_models.<model>.train`` module. Optionally runs
evaluation on the test set after training completes.

Usage::

    python scripts/slurm/train_model.py --config config/paper_results.yaml \
        --model ae
    python scripts/slurm/train_model.py --config config/quick_validation.yaml \
        --model vae --skip-eval
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_MODELS = ("ae", "vae", "gan", "unet", "vit")

MODEL_TRAIN_MODULES: dict[str, str] = {
    "ae": "dl_models.ae.train",
    "vae": "dl_models.vae.train",
    "gan": "dl_models.gan.train",
    "unet": "dl_models.unet.train",
    "vit": "dl_models.vit.train",
}

MODEL_ALLOWED_ARGS: dict[str, set[str]] = {
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
        "entropy_window",
        "entropy_buckets",
        "entropy_quantiles",
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
        "entropy_window",
        "entropy_buckets",
        "entropy_quantiles",
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
        "entropy_window",
        "entropy_buckets",
        "entropy_quantiles",
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
        "entropy_window",
        "entropy_buckets",
        "entropy_quantiles",
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
        "entropy_window",
        "entropy_buckets",
        "entropy_quantiles",
    },
}


def _args_to_argv(args_map: dict[str, Any]) -> list[str]:
    """Convert a dict of parameters to a ``sys.argv``-style list."""
    argv = ["train"]
    for key, value in args_map.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        argv.extend([flag, str(value)])
    return argv


@contextmanager
def _patched_argv(argv: list[str]) -> Any:
    """Temporarily replace ``sys.argv``."""
    original = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def _load_training_config(
    config_path: Path,
    model: str,
) -> dict[str, Any]:
    """Read ``dl_training`` from *config_path* and return config for *model*."""
    if not config_path.exists():
        msg = f"Config not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open() as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        msg = "Config must be a YAML mapping"
        raise TypeError(msg)

    training = data.get("dl_training")
    if not isinstance(training, dict):
        msg = (
            f"Config {config_path} missing 'dl_training' section. "
            "Add a dl_training mapping with scalar hyperparameters."
        )
        raise TypeError(msg)

    models = training.get("models")
    if not isinstance(models, dict) or model not in models:
        msg = f"Model '{model}' not found in dl_training.models"
        raise ValueError(msg)

    entropy_filter = data.get("entropy_filter", {})
    if entropy_filter is None:
        entropy_filter = {}

    return {
        "manifest": training.get("manifest", "preprocessed/manifest.csv"),
        "output_dir": training.get("output_dir", "dl_models/checkpoints"),
        "satellite": training.get("satellite", "sentinel2"),
        "device": training.get("device"),
        "num_workers": training.get("num_workers"),
        "eval_after_train": training.get("eval_after_train", True),
        "max_patches": training.get("max_patches"),
        "entropy_filter": entropy_filter,
        "model_params": models[model],
    }


def _validate_args(model: str, args_map: dict[str, Any]) -> None:
    allowed = MODEL_ALLOWED_ARGS[model]
    unknown = [k for k in args_map if k not in allowed]
    if unknown:
        msg = f"Unsupported args for {model}: {sorted(unknown)}"
        raise ValueError(msg)


def _run_training(model: str, train_args: dict[str, Any]) -> None:
    _validate_args(model, train_args)
    argv = _args_to_argv(train_args)

    log.info("Training %s", model)
    log.info("Args: %s", train_args)

    train_module = importlib.import_module(MODEL_TRAIN_MODULES[model])
    with _patched_argv(argv):
        train_module.main()


def _run_evaluation(
    model: str,
    checkpoint: Path,
    cfg: dict[str, Any],
) -> None:
    log.info("Evaluating %s", model)

    eval_dir = checkpoint.parent.parent / "eval"
    eval_argv = [
        "evaluate",
        "--model",
        model,
        "--checkpoint",
        str(checkpoint),
        "--manifest",
        str(cfg["manifest"]),
        "--output",
        str(eval_dir),
        "--satellite",
        str(cfg["satellite"]),
    ]
    if cfg["device"]:
        eval_argv.extend(["--device", str(cfg["device"])])
    if cfg["max_patches"] is not None:
        eval_argv.extend(["--max-patches", str(cfg["max_patches"])])

    entropy_filter = cfg.get("entropy_filter", {})
    if entropy_filter:
        window = entropy_filter.get("window")
        if window is not None:
            eval_argv.extend(["--entropy-window", str(window)])
        buckets = entropy_filter.get("eval_buckets")
        if buckets:
            eval_argv.extend(["--entropy-buckets", ",".join(buckets)])
        quantiles = entropy_filter.get("quantiles")
        if quantiles and len(quantiles) == 2:
            eval_argv.extend([
                "--entropy-quantiles",
                f"{quantiles[0]},{quantiles[1]}",
            ])

    eval_module = importlib.import_module("dl_models.evaluate")
    with _patched_argv(eval_argv):
        eval_module.main()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a single DL model from YAML config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML config (must contain dl_training section).",
    )
    parser.add_argument(
        "--model",
        choices=VALID_MODELS,
        required=True,
        help="Model to train.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip test evaluation after training.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_training_config(args.config, args.model)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{args.model}_best.pth"

    train_args: dict[str, Any] = {
        "manifest": cfg["manifest"],
        "output": str(checkpoint),
        "device": cfg["device"],
        "satellite": cfg["satellite"],
        "num_workers": cfg["num_workers"],
    }
    train_args.update(cfg["model_params"])

    entropy_filter = cfg.get("entropy_filter", {})
    if entropy_filter:
        window = entropy_filter.get("window")
        if window is not None:
            train_args["entropy_window"] = window
        buckets = entropy_filter.get("train_buckets")
        if buckets:
            train_args["entropy_buckets"] = ",".join(buckets)
        quantiles = entropy_filter.get("quantiles")
        if quantiles and len(quantiles) == 2:
            train_args["entropy_quantiles"] = f"{quantiles[0]},{quantiles[1]}"

    _run_training(args.model, train_args)

    if not args.skip_eval and cfg["eval_after_train"] and checkpoint.exists():
        _run_evaluation(args.model, checkpoint, cfg)

    log.info("Done: %s", args.model)


if __name__ == "__main__":
    main()
