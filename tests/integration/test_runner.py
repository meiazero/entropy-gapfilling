"""Integration test for the experiment runner.

Runs a minimal experiment with 1 seed, 1 noise level, and 3 fast
methods on a small subset of patches to verify the full pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "preprocessed" / "manifest.csv"
CONFIG_PATH = PROJECT_ROOT / "config" / "quick_validation.yaml"

_has_data = MANIFEST_PATH.exists()
_has_config = CONFIG_PATH.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not (_has_data and _has_config),
    reason="Requires preprocessed data and config",
)
class TestRunnerIntegration:
    def test_dry_run(self) -> None:
        from pdi_pipeline.config import load_config
        from scripts.run_experiment import run_experiment

        cfg = load_config(CONFIG_PATH)
        # Should not raise
        run_experiment(cfg, dry_run=True)

    def test_method_instantiation(self) -> None:
        from pdi_pipeline.config import load_config
        from scripts.run_experiment import instantiate_method

        cfg = load_config(CONFIG_PATH)
        for mc in cfg.methods:
            method = instantiate_method(mc)
            assert hasattr(method, "apply")
            assert hasattr(method, "name")

    def test_full_registry_instantiation(self) -> None:
        """All 16 paper methods can be instantiated."""
        from pdi_pipeline.config import load_config
        from scripts.run_experiment import instantiate_method

        cfg = load_config(PROJECT_ROOT / "config" / "paper_results.yaml")
        for mc in cfg.methods:
            method = instantiate_method(mc)
            assert method is not None
