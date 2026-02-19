"""Experiment configuration loader.

Reads YAML config files and exposes them as frozen dataclasses. Two
built-in configs ship with the project:

- ``config/paper_results.yaml``  -- full experiment (10 seeds, 4 noise levels)
- ``config/quick_validation.yaml`` -- quick smoke test (1 seed, 3 methods)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pdi_pipeline.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodConfig:
    """Configuration for a single interpolation method."""

    name: str
    category: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    seeds: list[int]
    noise_levels: list[str]
    satellites: list[str]
    entropy_windows: list[int]
    max_patches: int | None
    output_dir: str
    methods: list[MethodConfig]
    metrics: list[str]

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.name


VALID_CATEGORIES = frozenset({
    "spatial",
    "kernel",
    "geostatistical",
    "transform",
    "compressive",
    "patch_based",
})


def _ensure_top_level_keys(raw: dict[str, Any]) -> None:
    for key in ("experiment", "methods"):
        if key not in raw:
            msg = f"Config missing required top-level key: {key!r}"
            raise ConfigError(msg)


def _validate_list_of(
    value: Any,
    item_type: type,
    field_name: str,
) -> None:
    """Ensure *value* is a list whose items are all *item_type*.

    Raises:
        ConfigError: If validation fails.
    """
    if not isinstance(value, list) or not all(
        isinstance(v, item_type) for v in value
    ):
        type_name = item_type.__name__
        msg = f"experiment.{field_name} must be a list of {type_name}s"
        raise ConfigError(msg)


def _validate_experiment_section(exp: Any) -> None:
    if not isinstance(exp, dict):
        msg = "Config 'experiment' must be a mapping"
        raise ConfigError(msg)

    required_exp_keys = (
        "name",
        "seeds",
        "noise_levels",
        "satellites",
        "entropy_windows",
    )
    for key in required_exp_keys:
        if key not in exp:
            msg = f"Config 'experiment' missing required key: {key!r}"
            raise ConfigError(msg)

    if not isinstance(exp["name"], str) or not exp["name"]:
        msg = "experiment.name must be a non-empty string"
        raise ConfigError(msg)

    _validate_list_of(exp["seeds"], int, "seeds")
    _validate_list_of(exp["noise_levels"], str, "noise_levels")
    _validate_list_of(exp["satellites"], str, "satellites")
    _validate_list_of(exp["entropy_windows"], int, "entropy_windows")


def _validate_methods_section(methods_raw: Any) -> None:
    if not isinstance(methods_raw, dict):
        msg = "Config 'methods' must be a mapping of category -> method list"
        raise ConfigError(msg)

    for category, items in methods_raw.items():
        if category not in VALID_CATEGORIES:
            msg = (
                f"Unknown method category: {category!r}. "
                f"Valid: {sorted(VALID_CATEGORIES)}"
            )
            raise ConfigError(msg)

        if not isinstance(items, list):
            msg = f"Methods in category {category!r} must be a list"
            raise ConfigError(msg)

        for item in items:
            if not isinstance(item, dict):
                msg = (
                    f"Each method in {category!r} must be a mapping, "
                    f"got {type(item).__name__}"
                )
                raise ConfigError(msg)
            if "name" not in item:
                msg = f"Method in {category!r} missing required 'name' key"
                raise ConfigError(msg)
            if not isinstance(item["name"], str) or not item["name"]:
                msg = f"Method name in {category!r} must be a non-empty string"
                raise ConfigError(msg)


def _validate_raw(raw: dict[str, Any]) -> None:
    """Validate the raw YAML structure before dataclass construction.

    Raises:
        ConfigError: If required keys are missing or types are wrong.
    """
    _ensure_top_level_keys(raw)
    _validate_experiment_section(raw["experiment"])
    _validate_methods_section(raw["methods"])


def _parse_methods(raw: dict[str, list[dict]]) -> list[MethodConfig]:
    """Flatten category -> method list into a flat list of MethodConfig."""
    methods: list[MethodConfig] = []
    for category, items in raw.items():
        for item in items:
            methods.append(
                MethodConfig(
                    name=item["name"],
                    category=category,
                    params=item.get("params", {}),
                )
            )
    return methods


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed ExperimentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If required keys are missing or types are wrong.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    _validate_raw(raw)

    exp = raw["experiment"]
    methods = _parse_methods(raw["methods"])

    return ExperimentConfig(
        name=exp["name"],
        seeds=exp["seeds"],
        noise_levels=exp["noise_levels"],
        satellites=exp["satellites"],
        entropy_windows=exp["entropy_windows"],
        max_patches=exp.get("max_patches"),
        output_dir=exp.get("output_dir", "results/"),
        methods=methods,
        metrics=raw.get("metrics", ["psnr", "ssim", "rmse", "sam"]),
    )
