"""Unit tests for the configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
        assert len(cfg.methods) == 15
        names = [m.name for m in cfg.methods]
        assert "nearest" in names
        assert "kriging" in names

    def test_dineof_removed(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        names = [m.name for m in cfg.methods]
        assert "dineof" not in names

    def test_dl_methods_not_in_pipeline(self) -> None:
        """DL models are isolated and not in the experiment config."""
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        names = [m.name for m in cfg.methods]
        for dl_name in [
            "ae_inpainting",
            "vae_inpainting",
            "gan_inpainting",
            "transformer_inpainting",
        ]:
            assert dl_name not in names

    def test_method_categories(self) -> None:
        cfg = load_config(CONFIG_DIR / "paper_results.yaml")
        categories = {m.category for m in cfg.methods}
        assert "spatial" in categories
        assert "kernel" in categories
        assert "geostatistical" in categories
        assert "deep_learning" not in categories

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


def _write_yaml(data: dict, path: Path) -> None:
    path.write_text(yaml.dump(data, default_flow_style=False))


class TestConfigValidation:
    """Tests for the _validate_raw schema validation."""

    def _minimal_config(self) -> dict:
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

    def test_valid_minimal_config(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "valid.yaml"
        _write_yaml(self._minimal_config(), cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.name == "test"

    def test_missing_experiment_key(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        del data["experiment"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="experiment"):
            load_config(cfg_path)

    def test_missing_methods_key(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        del data["methods"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="methods"):
            load_config(cfg_path)

    def test_missing_experiment_name(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        del data["experiment"]["name"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="name"):
            load_config(cfg_path)

    def test_seeds_must_be_list_of_int(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        data["experiment"]["seeds"] = ["abc"]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="seeds"):
            load_config(cfg_path)

    def test_unknown_category_rejected(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        data["methods"]["bogus_category"] = [{"name": "foo"}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="Unknown method category"):
            load_config(cfg_path)

    def test_method_missing_name(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        data["methods"]["spatial"] = [{"params": {}}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="name"):
            load_config(cfg_path)

    def test_method_empty_name(self, tmp_path: Path) -> None:
        data = self._minimal_config()
        data["methods"]["spatial"] = [{"name": ""}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="non-empty string"):
            load_config(cfg_path)

    def test_deep_learning_category_rejected(self, tmp_path: Path) -> None:
        """deep_learning is not a valid pipeline category."""
        data = self._minimal_config()
        data["methods"]["deep_learning"] = [{"name": "ae_inpainting"}]
        cfg_path = tmp_path / "bad.yaml"
        _write_yaml(data, cfg_path)
        with pytest.raises(ValueError, match="Unknown method category"):
            load_config(cfg_path)
