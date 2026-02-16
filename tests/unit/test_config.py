"""Unit tests for the configuration module.

All tests use synthetic YAML via tmp_path - no dependency on real config files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pdi_pipeline.config import ExperimentConfig, load_config
from pdi_pipeline.exceptions import ConfigError


def _write_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.dump(data, default_flow_style=False))


def _minimal_config() -> dict:
    return {
        "experiment": {
            "name": "test",
            "seeds": [42],
            "noise_levels": ["inf"],
            "satellites": ["sentinel2"],
            "entropy_windows": [7],
        },
        "methods": {
            "spatial": [{"name": "nearest"}],
        },
    }


def _full_config() -> dict:
    """A richer config with multiple categories and params."""
    return {
        "experiment": {
            "name": "full_test",
            "seeds": [1, 2, 3],
            "noise_levels": ["inf", "30dB"],
            "satellites": ["sentinel2", "landsat8"],
            "entropy_windows": [7, 15, 31],
            "max_patches": 50,
            "output_dir": "results/",
        },
        "methods": {
            "spatial": [
                {"name": "nearest"},
                {"name": "bilinear"},
            ],
            "kernel": [
                {"name": "idw", "params": {"power": 2.0}},
            ],
            "geostatistical": [
                {"name": "kriging"},
            ],
        },
        "metrics": ["psnr", "ssim", "rmse", "sam"],
    }


# -- Loading and parsing ------------------------------------------------------


class TestLoadConfig:
    def test_load_minimal(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "minimal.yaml"
        _write_yaml(_minimal_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == "test"

    def test_load_full(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.name == "full_test"
        assert len(cfg.seeds) == 3
        assert cfg.max_patches == 50

    def test_methods_are_parsed(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert len(cfg.methods) == 4
        names = [m.name for m in cfg.methods]
        assert "nearest" in names
        assert "kriging" in names

    def test_method_categories(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        categories = {m.category for m in cfg.methods}
        assert "spatial" in categories
        assert "kernel" in categories
        assert "geostatistical" in categories

    def test_method_params_parsed(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        idw = next(m for m in cfg.methods if m.name == "idw")
        assert idw.params["power"] == 2.0

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_output_path_property(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.output_path == Path("results/") / "full_test"

    def test_metrics_list(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert "psnr" in cfg.metrics
        assert "sam" in cfg.metrics

    def test_entropy_windows(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "full.yaml"
        _write_yaml(_full_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.entropy_windows == [7, 15, 31]

    def test_max_patches_defaults_to_none(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "minimal.yaml"
        _write_yaml(_minimal_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.max_patches is None


# -- Validation ----------------------------------------------------------------


class TestConfigValidation:
    """Tests for the _validate_raw schema validation using ConfigError."""

    def test_valid_minimal_config(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "valid.yaml"
        _write_yaml(_minimal_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.name == "test"

    def test_missing_experiment_key(self, tmp_path: Path) -> None:
        data = _minimal_config()
        del data["experiment"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="experiment"):
            load_config(cfg_path)

    def test_missing_methods_key(self, tmp_path: Path) -> None:
        data = _minimal_config()
        del data["methods"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="methods"):
            load_config(cfg_path)

    def test_missing_experiment_name(self, tmp_path: Path) -> None:
        data = _minimal_config()
        del data["experiment"]["name"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="name"):
            load_config(cfg_path)

    def test_seeds_must_be_list_of_int(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["experiment"]["seeds"] = ["abc"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="seeds"):
            load_config(cfg_path)

    def test_unknown_category_rejected(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["methods"]["bogus_category"] = [{"name": "foo"}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="Unknown method category"):
            load_config(cfg_path)

    def test_method_missing_name(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["methods"]["spatial"] = [{"params": {}}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="name"):
            load_config(cfg_path)

    def test_method_empty_name(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["methods"]["spatial"] = [{"name": ""}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="non-empty string"):
            load_config(cfg_path)

    def test_deep_learning_category_rejected(self, tmp_path: Path) -> None:
        """deep_learning is not a valid pipeline category."""
        data = _minimal_config()
        data["methods"]["deep_learning"] = [{"name": "ae_inpainting"}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="Unknown method category"):
            load_config(cfg_path)

    def test_config_error_is_also_value_error(self, tmp_path: Path) -> None:
        """ConfigError inherits ValueError, so callers can catch either."""
        data = _minimal_config()
        del data["experiment"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="experiment"):
            load_config(cfg_path)

    def test_noise_levels_must_be_strings(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["experiment"]["noise_levels"] = [30, 40]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="noise_levels"):
            load_config(cfg_path)

    def test_entropy_windows_must_be_ints(self, tmp_path: Path) -> None:
        data = _minimal_config()
        data["experiment"]["entropy_windows"] = ["seven"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ConfigError, match="entropy_windows"):
            load_config(cfg_path)
