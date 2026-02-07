"""Shared fixtures for the pdi_pipeline test suite.

Fixtures load real preprocessed satellite patches from the ``preprocessed/``
directory.  The data is normalised to [0, 1] before being handed to tests so
that every method's ``_finalize`` clip-range is compatible.

If the preprocessed directory is absent or empty the integration tests that
depend on real data are skipped automatically via ``pytest.importorskip``-style
guards inside each fixture.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"
MANIFEST_PATH = PREPROCESSED_DIR / "manifest.csv"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchSample:
    """A single satellite patch with all noise variants."""

    patch_id: int
    satellite: str
    split: str
    bands: str
    gap_fraction: float
    clean: np.ndarray  # (H, W, C) float32, normalised [0, 1]
    mask: np.ndarray  # (H, W) float32, 1=gap 0=valid
    degraded_inf: np.ndarray  # (H, W, C) float32, normalised
    degraded_40: np.ndarray
    degraded_30: np.ndarray
    degraded_20: np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Min-max normalise *arr* to [0, 1] using global (vmin, vmax)."""
    span = max(vmax - vmin, 1e-8)
    out = (arr - vmin) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open() as fh:
        return list(csv.DictReader(fh))


def _load_patch(row: dict[str, str]) -> PatchSample | None:
    """Load a single row from the manifest and return a PatchSample."""
    clean_path = PREPROCESSED_DIR / row["clean_path"]
    mask_path = PREPROCESSED_DIR / row["mask_path"]
    if not clean_path.exists() or not mask_path.exists():
        return None

    clean = np.load(clean_path)
    mask = np.load(mask_path)

    variants: dict[str, np.ndarray] = {}
    for key in ("degraded_inf", "degraded_40", "degraded_30", "degraded_20"):
        col = f"{key}_path"
        p = PREPROCESSED_DIR / row[col]
        if not p.exists():
            return None
        arr = np.load(p)
        # Replace NaN in degraded arrays with 0 before normalising
        arr = np.nan_to_num(arr, nan=0.0)
        variants[key] = arr

    # Compute global min / max from clean for consistent normalisation
    vmin = float(clean.min())
    vmax = float(clean.max())
    if vmax - vmin < 1e-8:
        vmax = vmin + 1.0

    return PatchSample(
        patch_id=int(row["patch_id"]),
        satellite=row["satellite"],
        split=row["split"],
        bands=row.get("bands", ""),
        gap_fraction=float(row["gap_fraction"]),
        clean=_normalise(clean, vmin, vmax),
        mask=mask,
        degraded_inf=_normalise(variants["degraded_inf"], vmin, vmax),
        degraded_40=_normalise(variants["degraded_40"], vmin, vmax),
        degraded_30=_normalise(variants["degraded_30"], vmin, vmax),
        degraded_20=_normalise(variants["degraded_20"], vmin, vmax),
    )


# ---------------------------------------------------------------------------
# Module-level cache (loaded once per session)
# ---------------------------------------------------------------------------
_ALL_PATCHES: list[PatchSample] | None = None


def _get_all_patches() -> list[PatchSample]:
    global _ALL_PATCHES
    if _ALL_PATCHES is None:
        rows = _load_manifest()
        _ALL_PATCHES = []
        for row in rows:
            sample = _load_patch(row)
            if sample is not None:
                _ALL_PATCHES.append(sample)
    return _ALL_PATCHES


# ---------------------------------------------------------------------------
# Skip helper
# ---------------------------------------------------------------------------
_has_real_data: bool | None = None


def _check_real_data() -> bool:
    global _has_real_data
    if _has_real_data is None:
        _has_real_data = len(_get_all_patches()) > 0
    return _has_real_data


requires_real_data = pytest.mark.skipif(
    "not config.stash.get('has_real_data', False)",
    reason="Preprocessed satellite patches not found",
)


def pytest_configure(config: pytest.Config) -> None:
    config.stash["has_real_data"] = _check_real_data()


# ---------------------------------------------------------------------------
# Fixtures -- single patches
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def all_patches() -> list[PatchSample]:
    """All available preprocessed patches (session-scoped, loaded once)."""
    patches = _get_all_patches()
    if not patches:
        pytest.skip("No preprocessed patches available")
    return patches


@pytest.fixture(scope="session")
def sentinel2_patch(all_patches: list[PatchSample]) -> PatchSample:
    """A single Sentinel-2 patch for quick tests."""
    for p in all_patches:
        if p.satellite == "sentinel2":
            return p
    pytest.skip("No Sentinel-2 patch available")


@pytest.fixture(scope="session")
def landsat8_patch(all_patches: list[PatchSample]) -> PatchSample:
    """A single Landsat-8 patch."""
    for p in all_patches:
        if p.satellite == "landsat8":
            return p
    pytest.skip("No Landsat-8 patch available")


@pytest.fixture(scope="session")
def landsat9_patch(all_patches: list[PatchSample]) -> PatchSample:
    """A single Landsat-9 patch."""
    for p in all_patches:
        if p.satellite == "landsat9":
            return p
    pytest.skip("No Landsat-9 patch available")


@pytest.fixture(scope="session")
def modis_patch(all_patches: list[PatchSample]) -> PatchSample:
    """A single MODIS patch."""
    for p in all_patches:
        if p.satellite == "modis":
            return p
    pytest.skip("No MODIS patch available")


# ---------------------------------------------------------------------------
# Fixtures -- grouped by satellite
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def patches_by_satellite(
    all_patches: list[PatchSample],
) -> dict[str, list[PatchSample]]:
    """Patches grouped by satellite name."""
    grouped: dict[str, list[PatchSample]] = {}
    for p in all_patches:
        grouped.setdefault(p.satellite, []).append(p)
    return grouped


# ---------------------------------------------------------------------------
# Noise-variant iteration helpers (used by parametrised integration tests)
# ---------------------------------------------------------------------------
NOISE_VARIANTS = ("inf", "40", "30", "20")


def degraded_for_variant(patch: PatchSample, variant: str) -> np.ndarray:
    """Return the degraded array for a named noise variant."""
    return {
        "inf": patch.degraded_inf,
        "40": patch.degraded_40,
        "30": patch.degraded_30,
        "20": patch.degraded_20,
    }[variant]


# ---------------------------------------------------------------------------
# Metric helpers (available to all test modules)
# ---------------------------------------------------------------------------
def psnr(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two images."""
    mse = float(np.mean((reference - reconstructed) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def rmse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root Mean Squared Error between two images."""
    return float(np.sqrt(np.mean((reference - reconstructed) ** 2)))
