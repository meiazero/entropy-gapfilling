"""Unit tests for the configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdi_pipeline.config import ExperimentConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class TestLoadConfig:
    def test_load_paper_results(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "paper_results"
        assert len(cfg.seeds) == 10
        assert len(cfg.noise_levels) == 4
        assert cfg.max_patches is None

    def test_load_quick_validation(self) -> None:
        cfg = load_config(CONFIG_DIR / "quick_validation.yaml")
        assert cfg.name == "quick_validation"
        assert len(cfg.seeds) == 1
        assert cfg.max_patches == 50

    def test_methods_are_parsed(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        assert len(cfg.methods) == 16
        names = [m.name for m in cfg.methods]
        assert "nearest" in names
        assert "kriging" in names

    def test_method_categories(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        categories = {m.category for m in cfg.methods}
        assert "spatial" in categories
        assert "kernel" in categories
        assert "geostatistical" in categories

    def test_method_params_parsed(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        idw = next(m for m in cfg.methods if m.name == "idw")
        assert idw.params["power"] == 2.0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_output_path_property(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        assert cfg.output_path == Path("results/paper_results")

    def test_metrics_list(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        assert "psnr" in cfg.metrics
        assert "sam" in cfg.metrics

    def test_entropy_windows(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        assert cfg.entropy_windows == [7, 15, 31]
