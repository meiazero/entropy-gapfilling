"""Grid search runner for classical interpolation methods.

Expands per-method parameter grids and executes the classical experiment
pipeline for each configuration. Writes a consolidated CSV with
execution time and metric summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pdi_pipeline.aggregation import load_results
from pdi_pipeline.config import ExperimentConfig, MethodConfig, load_config
from pdi_pipeline.logging_utils import setup_file_logging, setup_logging
from pdi_pipeline.methods.registry import list_aliases, list_categories
from scripts.run_experiment import run_experiment

log = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return text.strip("_") or "run"


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _resolve_method_category(method_name: str) -> str:
    aliases = list_aliases()
    canonical = aliases.get(method_name, method_name)
    categories = list_categories()
    for category, methods in categories.items():
        if canonical in methods:
            return category
    msg = f"Unknown method category for {method_name!r}"
    raise ValueError(msg)


def _expand_grid(params: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not params:
        return [{}]
    keys = list(params.keys())
    values: list[list[Any]] = [params[k] for k in keys]
    combos = []
    for combo in product(*values):
        combos.append(dict(zip(keys, combo, strict=True)))
    return combos


def _apply_overrides(
    config: ExperimentConfig,
    overrides: dict[str, Any] | None,
) -> ExperimentConfig:
    if not overrides:
        return config

    valid_keys = {
        "name",
        "seeds",
        "noise_levels",
        "satellites",
        "entropy_windows",
        "max_patches",
        "workers",
        "output_dir",
        "metrics",
    }
    unknown = [k for k in overrides if k not in valid_keys]
    if unknown:
        msg = f"Unknown experiment overrides: {sorted(unknown)}"
        raise ValueError(msg)

    return replace(config, **overrides)


def _summarize_results(
    df: pd.DataFrame,
    metrics: Iterable[str],
    wall_time_s: float,
) -> dict[str, Any]:
    ok_df = df[df["status"] == "ok"] if "status" in df.columns else df
    summary: dict[str, Any] = {
        "n_rows": len(df),
        "n_ok": len(ok_df),
        "n_error": int(len(df) - len(ok_df)),
        "error_rate": float((len(df) - len(ok_df)) / len(df))
        if len(df) > 0
        else 0.0,
        "wall_time_s": float(wall_time_s),
    }

    if "elapsed_s" in ok_df.columns and not ok_df.empty:
        summary["mean_elapsed_s"] = float(ok_df["elapsed_s"].mean())
        summary["p95_elapsed_s"] = float(ok_df["elapsed_s"].quantile(0.95))
        summary["throughput_rows_per_s"] = (
            float(len(ok_df)) / wall_time_s if wall_time_s > 0 else 0.0
        )
    else:
        summary["mean_elapsed_s"] = 0.0
        summary["p95_elapsed_s"] = 0.0
        summary["throughput_rows_per_s"] = 0.0

    for metric in metrics:
        if metric in ok_df.columns and not ok_df.empty:
            summary[f"{metric}_mean"] = float(ok_df[metric].mean())
            summary[f"{metric}_std"] = float(ok_df[metric].std())
        else:
            summary[f"{metric}_mean"] = float("nan")
            summary[f"{metric}_std"] = float("nan")

    return summary


def _append_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "grid_results.csv"
    df = pd.DataFrame([summary])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid search for classical interpolation methods.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Experiment config YAML (uses grid_search section by default).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=False,
        help="Deprecated alias for --config.",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        required=False,
        help="Optional grid search YAML (overrides config grid_search).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs with existing raw_results.csv.",
    )
    return parser


def _load_grid(path: Path) -> dict[str, Any]:
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
    grid = data.get("grid_search")
    if not isinstance(grid, dict):
        msg = "Config missing 'grid_search' mapping"
        raise TypeError(msg)
    return grid


def main() -> None:
    setup_logging()
    args = _build_parser().parse_args()

    config_path = args.config or args.base_config
    if config_path is None:
        msg = "--config is required"
        raise ValueError(msg)

    base_config = load_config(config_path)
    grid = (
        _load_grid(args.grid)
        if args.grid
        else _load_grid_from_config(config_path)
    )

    output_dir = Path(grid.get("output_dir", "results/grid_search/classical"))
    overrides = grid.get("experiment_overrides")
    methods_grid = grid.get("methods")

    if not isinstance(methods_grid, dict) or not methods_grid:
        msg = "Grid config must include a non-empty 'methods' mapping"
        raise ValueError(msg)

    config = _apply_overrides(base_config, overrides)

    grid_csv_dir = output_dir
    grid_csv_dir.mkdir(parents=True, exist_ok=True)

    run_total = 0
    for method_name, param_grid in methods_grid.items():
        param_combos = _expand_grid(param_grid)
        category = _resolve_method_category(method_name)

        for params in param_combos:
            run_total += 1
            param_parts = [
                f"{key}={_format_value(value)}" for key, value in params.items()
            ]
            tag = _slugify("_".join(param_parts)) if param_parts else "default"
            run_name = _slugify(f"{method_name}_{tag}")

            output_path = output_dir / run_name
            raw_results = output_path / "raw_results.csv"
            if args.skip_existing and raw_results.exists():
                log.info("Skip existing run: %s", run_name)
                continue

            setup_file_logging(output_path, name="grid_search")

            method_cfg = MethodConfig(
                name=method_name,
                category=category,
                params=params,
            )
            run_config = replace(
                config,
                name=run_name,
                output_dir=str(output_dir),
                methods=[method_cfg],
            )

            log.info("Run %d: %s", run_total, run_name)
            log.info("Params: %s", params)

            t0 = time.monotonic()
            run_experiment(run_config)
            wall_time_s = time.monotonic() - t0

            results_df = load_results(output_path)
            summary = {
                "run_name": run_name,
                "method": method_name,
                "params": json.dumps(params, sort_keys=True),
            }
            summary.update(
                _summarize_results(results_df, run_config.metrics, wall_time_s)
            )

            _append_summary(grid_csv_dir, summary)

    log.info(
        "Grid search complete. Summary: %s", grid_csv_dir / "grid_results.csv"
    )


if __name__ == "__main__":
    main()
