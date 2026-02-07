"""Experiment configuration loader.

Reads YAML config files and exposes them as frozen dataclasses. Two
built-in configs ship with the project:

- ``config/paper_results.yaml``  -- full experiment (10 seeds, 4 noise levels)
- ``config/quick_validation.yaml`` -- quick smoke test (1 seed, 3 methods)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
        KeyError: If required keys are missing.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as fh:
        raw = yaml.safe_load(fh)

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
