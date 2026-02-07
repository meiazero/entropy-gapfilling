"""Unit tests for the dataset module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdi_pipeline.dataset import PatchDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "preprocessed" / "manifest.csv"

_has_manifest = MANIFEST_PATH.exists()


@pytest.mark.skipif(not _has_manifest, reason="No manifest.csv available")
class TestPatchDataset:
    def test_len_is_non_negative(self) -> None:
        ds = PatchDataset(MANIFEST_PATH, split="test")
        assert len(ds) >= 0

    def test_filter_by_satellite(self) -> None:
        ds_all = PatchDataset(MANIFEST_PATH, split="test")
        ds_s2 = PatchDataset(MANIFEST_PATH, split="test", satellite="sentinel2")
        # Filtered should be <= total
        assert len(ds_s2) <= len(ds_all)

    def test_max_patches_truncates(self) -> None:
        ds = PatchDataset(MANIFEST_PATH, split="test", max_patches=3)
        assert len(ds) <= 3

    def test_getitem_returns_patch_sample(self) -> None:
        ds = PatchDataset(MANIFEST_PATH, split="test", max_patches=1)
        if len(ds) == 0:
            pytest.skip("No test patches available")
        sample = ds[0]
        assert sample.clean.ndim in (2, 3)
        assert sample.mask.ndim == 2
        assert sample.degraded.shape == sample.clean.shape
        assert sample.noise_level == "inf"

    def test_invalid_noise_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid noise_level"):
            PatchDataset(MANIFEST_PATH, split="test", noise_level="999")

    def test_missing_manifest_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            PatchDataset("/nonexistent/path.csv", split="test")

    def test_index_out_of_range_raises(self) -> None:
        ds = PatchDataset(MANIFEST_PATH, split="test", max_patches=1)
        with pytest.raises(IndexError):
            ds[999999]

    def test_iteration(self) -> None:
        ds = PatchDataset(MANIFEST_PATH, split="test", max_patches=2)
        count = 0
        for sample in ds:
            count += 1
            assert hasattr(sample, "patch_id")
        assert count == len(ds)

    def test_patch_ids_property(self) -> None:
        ds = PatchDataset(MANIFEST_PATH, split="test", max_patches=3)
        ids = ds.patch_ids
        assert len(ids) == len(ds)
        assert all(isinstance(i, int) for i in ids)
