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
        "noise_levels": training.get("noise_levels", ["inf", "40", "30", "20"]),
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
    eval_base_dir: Path,
    scenario_name: str | None = None,
    eval_buckets: list[str] | None = None,
    noise_level: str = "inf",
) -> None:
    """Run evaluation for one (scenario, noise_level) combination.

    Args:
        model: Model key (ae, vae, gan, unet, vit).
        checkpoint: Path to the trained checkpoint.
        cfg: Parsed training config dict.
        eval_base_dir: Root directory for evaluation outputs.
        scenario_name: Subdirectory name for this evaluation scenario.
            If None, results are written directly to the base eval dir.
        eval_buckets: Entropy buckets to include for this scenario.
            Overrides entropy_filter.eval_buckets from cfg.
        noise_level: Noise level variant to evaluate (inf/40/30/20).
    """
    eval_dir = eval_base_dir / scenario_name if scenario_name else eval_base_dir

    log.info(
        "Evaluating %s | scenario=%s | noise=%s",
        model,
        scenario_name or "default",
        noise_level,
    )

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
        "--noise-level",
        noise_level,
    ]
    if cfg["device"]:
        eval_argv.extend(["--device", str(cfg["device"])])
    if cfg["max_patches"] is not None:
        eval_argv.extend(["--max-patches", str(cfg["max_patches"])])

    entropy_filter = cfg.get("entropy_filter") or {}
    if entropy_filter:
        window = entropy_filter.get("window")
        if window is not None:
            eval_argv.extend(["--entropy-window", str(window)])
        # Explicit eval_buckets for this scenario take priority
        #  over config default.
        buckets = (
            eval_buckets
            if eval_buckets is not None
            else entropy_filter.get("eval_buckets")
        )
        if buckets:
            eval_argv.extend(["--entropy-buckets", ",".join(buckets)])
        quantiles = entropy_filter.get("quantiles")
        if quantiles and len(quantiles) == 2:
            eval_argv.extend([
                "--entropy-quantiles",
                f"{quantiles[0]},{quantiles[1]}",
            ])

    if scenario_name:
        eval_argv.extend(["--scenario-name", scenario_name])

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


def _build_base_train_args(
    cfg: dict[str, Any], checkpoint: Path
) -> dict[str, Any]:
    """Build the base training args dict from config,
    then merge model params."""
    train_args: dict[str, Any] = {
        "manifest": cfg["manifest"],
        "output": str(checkpoint),
        "device": cfg["device"],
        "satellite": cfg["satellite"],
        "num_workers": cfg["num_workers"],
    }
    train_args.update(cfg["model_params"])
    return train_args


def _apply_entropy_to_train_args(
    train_args: dict[str, Any],
    entropy_filter: dict[str, Any],
    buckets: list[str] | None,
) -> None:
    """Inject entropy-filter keys into *train_args* in-place."""
    window = entropy_filter.get("window")
    if window is not None:
        train_args["entropy_window"] = window
    if buckets:
        train_args["entropy_buckets"] = ",".join(buckets)
    quantiles = entropy_filter.get("quantiles")
    if quantiles and len(quantiles) == 2:
        train_args["entropy_quantiles"] = f"{quantiles[0]},{quantiles[1]}"


def _run_scenario_pass(
    args: argparse.Namespace,
    scenario: dict[str, Any],
    cfg: dict[str, Any],
    base_output_dir: Path,
    eval_base_dir: Path,
    noise_levels: list[str],
) -> None:
    """Train and optionally evaluate one entropy scenario."""
    scenario_name: str = scenario["name"]
    scenario_buckets: list[str] = scenario["buckets"]

    scenario_dir = base_output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = scenario_dir / f"{args.model}_best.pth"

    entropy_filter: dict[str, Any] = cfg.get("entropy_filter") or {}
    train_args = _build_base_train_args(cfg, checkpoint)
    if entropy_filter:
        _apply_entropy_to_train_args(
            train_args, entropy_filter, scenario_buckets
        )

    log.info(
        "--- Scenario %s: training on buckets %s",
        scenario_name,
        scenario_buckets,
    )
    _run_training(args.model, train_args)

    if not args.skip_eval and cfg["eval_after_train"] and checkpoint.exists():
        log.info(
            "--- Scenario %s: evaluating %d noise level(s)",
            scenario_name,
            len(noise_levels),
        )
        for noise_level in noise_levels:
            _run_evaluation(
                args.model,
                checkpoint,
                cfg,
                eval_base_dir=eval_base_dir,
                scenario_name=scenario_name,
                eval_buckets=scenario_buckets,
                noise_level=noise_level,
            )


def _run_single_pass(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    base_output_dir: Path,
    eval_base_dir: Path,
    noise_levels: list[str],
) -> None:
    """Single training + evaluation pass (no scenario splitting)."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = base_output_dir / f"{args.model}_best.pth"

    entropy_filter: dict[str, Any] = cfg.get("entropy_filter") or {}
    train_args = _build_base_train_args(cfg, checkpoint)
    if entropy_filter:
        buckets: list[str] | None = entropy_filter.get("train_buckets")
        _apply_entropy_to_train_args(train_args, entropy_filter, buckets)

    _run_training(args.model, train_args)

    if not args.skip_eval and cfg["eval_after_train"] and checkpoint.exists():
        for noise_level in noise_levels:
            _run_evaluation(
                args.model,
                checkpoint,
                cfg,
                eval_base_dir=eval_base_dir,
                noise_level=noise_level,
            )


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_training_config(args.config, args.model)

    base_output_dir = Path(cfg["output_dir"])
    entropy_filter: dict[str, Any] = cfg.get("entropy_filter") or {}
    scenarios: list[dict[str, Any]] | None = entropy_filter.get("scenarios")
    noise_levels: list[str] = cfg.get("noise_levels", ["inf", "40", "30", "20"])
    # eval outputs always go to a sibling "eval" dir relative to checkpoints
    eval_base_dir = base_output_dir.parent / "eval"

    if scenarios:
        log.info(
            "Training %s across %d scenario(s)", args.model, len(scenarios)
        )
        for scenario in scenarios:
            _run_scenario_pass(
                args,
                scenario,
                cfg,
                base_output_dir,
                eval_base_dir,
                noise_levels,
            )
    else:
        # No scenarios: single training + evaluation pass (backward compat).
        _run_single_pass(
            args, cfg, base_output_dir, eval_base_dir, noise_levels
        )

    log.info("Done: %s", args.model)


if __name__ == "__main__":
    main()
