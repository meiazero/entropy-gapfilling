"""Unit tests for the dataset module.

Uses tmp_path with synthetic manifest CSV and NPY files so tests
run without access to real satellite imagery.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pdi_pipeline.dataset import PatchDataset, PatchSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_H, _W, _C = 8, 8, 4


def _make_patch_npy(
    base_dir: Path,
    split: str,
    satellite: str,
    patch_id: int,
) -> dict[str, str]:
    """Create synthetic NPY files for one patch and return manifest row paths."""
    sat_dir = base_dir / split / satellite
    sat_dir.mkdir(parents=True, exist_ok=True)

    clean = (
        np.random.default_rng(patch_id).random((_H, _W, _C)).astype(np.float32)
    )
    mask = np.zeros((_H, _W), dtype=np.float32)
    mask[:4, :4] = 1.0  # 25% gap

    degraded = clean.copy()
    degraded[mask.astype(bool)] = 0.0

    paths: dict[str, str] = {}
    for variant, arr in [
        ("clean", clean),
        ("mask", mask),
        ("degraded_inf", degraded),
        ("degraded_40", degraded + 0.01),
        ("degraded_30", degraded + 0.02),
        ("degraded_20", degraded + 0.03),
    ]:
        fname = f"{patch_id:07d}_{variant}.npy"
        np.save(sat_dir / fname, arr)
        paths[variant] = str(Path(split) / satellite / fname)

    return paths


def _write_manifest(
    base_dir: Path,
    rows: list[dict[str, str]],
) -> Path:
    """Write a manifest CSV and return its path."""
    manifest_path = base_dir / "manifest.csv"
    fieldnames = [
        "patch_id",
        "satellite",
        "split",
        "bands",
        "gap_fraction",
        "acquisition_date",
        "clean_path",
        "mask_path",
        "degraded_inf_path",
        "degraded_40_path",
        "degraded_30_path",
        "degraded_20_path",
    ]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _create_dataset_dir(
    base_dir: Path,
    n_patches: int = 5,
    split: str = "test",
    satellite: str = "sentinel2",
) -> Path:
    """Create a complete synthetic dataset directory and return manifest path."""
    manifest_rows = []
    for pid in range(1, n_patches + 1):
        npy_paths = _make_patch_npy(base_dir, split, satellite, pid)
        manifest_rows.append({
            "patch_id": str(pid),
            "satellite": satellite,
            "split": split,
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
    return _write_manifest(base_dir, manifest_rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPatchDataset:
    def test_len_is_non_negative(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=3)
        ds = PatchDataset(manifest, split="test")
        assert len(ds) == 3

    def test_filter_by_satellite(self, tmp_path: Path) -> None:
        # Create patches for two satellites
        rows = []
        for sat in ("sentinel2", "landsat8"):
            for pid in range(1, 4):
                offset = 0 if sat == "sentinel2" else 100
                actual_pid = pid + offset
                npy_paths = _make_patch_npy(tmp_path, "test", sat, actual_pid)
                rows.append({
                    "patch_id": str(actual_pid),
                    "satellite": sat,
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
        manifest = _write_manifest(tmp_path, rows)

        ds_all = PatchDataset(manifest, split="test")
        ds_s2 = PatchDataset(manifest, split="test", satellite="sentinel2")
        assert len(ds_s2) <= len(ds_all)
        assert len(ds_s2) == 3

    def test_max_patches_truncates(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=5)
        ds = PatchDataset(manifest, split="test", max_patches=2)
        assert len(ds) == 2

    def test_getitem_returns_patch_sample(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=2)
        ds = PatchDataset(manifest, split="test")
        sample = ds[0]
        assert isinstance(sample, PatchSample)
        assert sample.clean.ndim == 3
        assert sample.clean.shape == (_H, _W, _C)
        assert sample.mask.ndim == 2
        assert sample.degraded.shape == sample.clean.shape
        assert sample.noise_level == "inf"

    def test_invalid_noise_level_raises(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=1)
        with pytest.raises(ValueError, match="Invalid noise_level"):
            PatchDataset(manifest, split="test", noise_level="999")

    def test_missing_manifest_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            PatchDataset("/nonexistent/path.csv", split="test")

    def test_index_out_of_range_raises(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=1)
        ds = PatchDataset(manifest, split="test")
        with pytest.raises(IndexError):
            ds[999]

    def test_negative_index_raises(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=1)
        ds = PatchDataset(manifest, split="test")
        with pytest.raises(IndexError):
            ds[-1]

    def test_iteration(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=3)
        ds = PatchDataset(manifest, split="test")
        count = 0
        for sample in ds:
            count += 1
            assert hasattr(sample, "patch_id")
        assert count == len(ds)

    def test_patch_ids_property(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=3)
        ds = PatchDataset(manifest, split="test")
        ids = ds.patch_ids
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)

    def test_split_filtering(self, tmp_path: Path) -> None:
        """Only patches matching the requested split are returned."""
        rows = []
        for split in ("train", "test"):
            for pid in range(1, 3):
                offset = 0 if split == "train" else 10
                actual_pid = pid + offset
                npy_paths = _make_patch_npy(
                    tmp_path, split, "sentinel2", actual_pid
                )
                rows.append({
                    "patch_id": str(actual_pid),
                    "satellite": "sentinel2",
                    "split": split,
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
        manifest = _write_manifest(tmp_path, rows)

        ds_train = PatchDataset(manifest, split="train")
        ds_test = PatchDataset(manifest, split="test")
        assert len(ds_train) == 2
        assert len(ds_test) == 2

    def test_noise_level_40(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=1)
        ds = PatchDataset(manifest, split="test", noise_level="40")
        sample = ds[0]
        assert sample.noise_level == "40"

    def test_selected_ids(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=5)
        ds = PatchDataset(manifest, split="test", selected_ids=[2, 4])
        assert len(ds) == 2
        assert set(ds.patch_ids) == {2, 4}

    def test_clean_normalized_to_01(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=1)
        ds = PatchDataset(manifest, split="test")
        sample = ds[0]
        assert sample.clean.min() >= 0.0
        assert sample.clean.max() <= 1.0
        assert sample.clean.dtype == np.float32

    def test_mask_is_2d(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=1)
        ds = PatchDataset(manifest, split="test")
        sample = ds[0]
        assert sample.mask.ndim == 2

    def test_satellites_property(self, tmp_path: Path) -> None:
        rows = []
        for sat in ("landsat8", "sentinel2"):
            npy_paths = _make_patch_npy(tmp_path, "test", sat, hash(sat) % 1000)
            rows.append({
                "patch_id": str(hash(sat) % 1000),
                "satellite": sat,
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
        manifest = _write_manifest(tmp_path, rows)
        ds = PatchDataset(manifest, split="test")
        assert ds.satellites == ["landsat8", "sentinel2"]

    def test_empty_split_returns_empty_dataset(self, tmp_path: Path) -> None:
        manifest = _create_dataset_dir(tmp_path, n_patches=3, split="test")
        ds = PatchDataset(manifest, split="val")
        assert len(ds) == 0
