"""Integration test for the experiment runner.

Uses synthetic data and an in-memory config so the full pipeline can
be tested without access to real satellite imagery or YAML configs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pdi_pipeline.config import ExperimentConfig, MethodConfig
from pdi_pipeline.methods.base import BaseMethod

# ---------------------------------------------------------------------------
# Helpers: build a minimal synthetic dataset on disk
# ---------------------------------------------------------------------------
_H, _W, _C = 16, 16, 4


def _make_patch_npy(
    base_dir: Path,
    split: str,
    satellite: str,
    patch_id: int,
) -> dict[str, str]:
    sat_dir = base_dir / split / satellite
    sat_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(patch_id)
    clean = rng.random((_H, _W, _C)).astype(np.float32)
    mask = np.zeros((_H, _W), dtype=np.float32)
    mask[:8, :8] = 1.0

    degraded = clean.copy()
    degraded[mask.astype(bool)] = 0.0

    paths: dict[str, str] = {}
    for variant, arr in [
        ("clean", clean),
        ("mask", mask),
        ("degraded_inf", degraded),
        ("degraded_40", degraded),
        ("degraded_30", degraded),
        ("degraded_20", degraded),
    ]:
        fname = f"{patch_id:07d}_{variant}.npy"
        np.save(sat_dir / fname, arr)
        paths[variant] = str(Path(split) / satellite / fname)

    return paths


def _create_test_dataset(
    base_dir: Path,
    n_patches: int = 3,
    satellite: str = "sentinel2",
) -> Path:
    rows = []
    for pid in range(1, n_patches + 1):
        npy_paths = _make_patch_npy(base_dir, "test", satellite, pid)
        rows.append({
            "patch_id": str(pid),
            "satellite": satellite,
            "split": "test",
            "bands": "B2,B3,B4,B8",
            "gap_fraction": "0.25",
            "acquisition_date": "2024-01-01",
            "clean_path": npy_paths["clean"],
            "mask_path": npy_paths["mask"],
            "degraded_inf_path": npy_paths["degraded_inf"],
            "degraded_40_path": npy_paths["degraded_40"],
            "degraded_30_path": npy_paths["degraded_30"],
            "degraded_20_path": npy_paths["degraded_20"],
        })

    manifest = base_dir / "manifest.csv"
    fieldnames = list(rows[0].keys())
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestRunnerIntegration:
    def test_method_instantiation(self) -> None:
        """All 15 classical methods can be instantiated via the registry."""
        from pdi_pipeline.methods.registry import get_interpolator

        method_names = [
            "nearest",
            "bilinear",
            "bicubic",
            "lanczos",
            "idw",
            "rbf",
            "spline",
            "kriging",
            "dct",
            "wavelet",
            "tv",
            "cs_dct",
            "cs_wavelet",
            "non_local",
            "exemplar_based",
        ]
        for name in method_names:
            method = get_interpolator(name)
            assert isinstance(method, BaseMethod)
            assert hasattr(method, "apply")

    def test_dry_run_with_synthetic_data(self, tmp_path: Path) -> None:
        """Dry-run of experiment loop with synthetic dataset."""
        from scripts.run_experiment import run_experiment

        manifest = _create_test_dataset(tmp_path, n_patches=2)

        # Monkey-patch the default manifest location
        import scripts.run_experiment as runner_mod

        original_manifest = runner_mod.DEFAULT_MANIFEST
        runner_mod.DEFAULT_MANIFEST = manifest

        try:
            config = ExperimentConfig(
                name="test_dry_run",
                seeds=[42],
                noise_levels=["inf"],
                satellites=["sentinel2"],
                max_patches=2,
                methods=[
                    MethodConfig(name="nearest", category="spatial", params={}),
                ],
                output_dir=str(tmp_path),
                metrics=["psnr", "ssim", "rmse"],
                entropy_windows=[7],
            )
            # Should complete without error
            run_experiment(config, dry_run=True)
        finally:
            runner_mod.DEFAULT_MANIFEST = original_manifest

    def test_single_method_run(self, tmp_path: Path) -> None:
        """Run a real (non-dry) experiment with one fast method."""
        from scripts.run_experiment import run_experiment

        manifest = _create_test_dataset(tmp_path, n_patches=2)

        import scripts.run_experiment as runner_mod

        original_manifest = runner_mod.DEFAULT_MANIFEST
        runner_mod.DEFAULT_MANIFEST = manifest

        try:
            config = ExperimentConfig(
                name="test_real_run",
                seeds=[42],
                noise_levels=["inf"],
                satellites=["sentinel2"],
                max_patches=2,
                methods=[
                    MethodConfig(name="nearest", category="spatial", params={}),
                ],
                output_dir=str(tmp_path),
                metrics=["psnr", "ssim", "rmse"],
                entropy_windows=[7],
            )
            output_dir = config.output_path
            run_experiment(config, dry_run=False)

            # Verify CSV was produced
            csv_path = output_dir / "raw_results.csv"
            assert csv_path.exists()

            import pandas as pd

            df = pd.read_csv(csv_path)
            assert len(df) == 2  # 2 patches
            assert set(df["method"]) == {"nearest"}
            assert all(df["status"] == "ok")
        finally:
            runner_mod.DEFAULT_MANIFEST = original_manifest

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        """Re-running the same experiment skips already-completed rows."""
        from scripts.run_experiment import run_experiment

        manifest = _create_test_dataset(tmp_path, n_patches=2)

        import scripts.run_experiment as runner_mod

        original_manifest = runner_mod.DEFAULT_MANIFEST
        runner_mod.DEFAULT_MANIFEST = manifest

        try:
            config = ExperimentConfig(
                name="test_resume",
                seeds=[42],
                noise_levels=["inf"],
                satellites=["sentinel2"],
                max_patches=2,
                methods=[
                    MethodConfig(name="nearest", category="spatial", params={}),
                ],
                output_dir=str(tmp_path),
                metrics=["psnr", "ssim", "rmse"],
                entropy_windows=[7],
            )
            output_dir = config.output_path
            run_experiment(config, dry_run=False)
            # Run again - should skip everything
            run_experiment(config, dry_run=False)

            import pandas as pd

            df = pd.read_csv(output_dir / "raw_results.csv")
            # Should still have exactly 2 rows (no duplicates)
            assert len(df) == 2
        finally:
            runner_mod.DEFAULT_MANIFEST = original_manifest
