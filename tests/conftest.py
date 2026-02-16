"""Shared fixtures for the pdi_pipeline test suite.

Provides synthetic image fixtures that require no real data files,
enabling the full test suite to run in CI without access to the
preprocessed satellite imagery dataset.

Real-data fixtures are still available for integration tests that
require actual satellite patches, but they gracefully skip when data
is absent.
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
# Synthetic image generators
# ---------------------------------------------------------------------------
def make_checkerboard(
    height: int = 32,
    width: int = 32,
    channels: int = 4,
    block_size: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Generate a checkerboard pattern in [0, 1].

    Args:
        height: Image height.
        width: Image width.
        channels: Number of bands.  Use 0 for a 2D (H, W) output.
        block_size: Side length of each checker square.
        seed: Unused (deterministic), kept for API consistency.

    Returns:
        float32 array with shape (H, W, C) or (H, W).
    """
    rows = np.arange(height) // block_size
    cols = np.arange(width) // block_size
    board = ((rows[:, None] + cols[None, :]) % 2).astype(np.float32)
    if channels == 0:
        return board
    return np.stack(
        [board * (0.3 + 0.1 * c) for c in range(channels)], axis=-1
    ).astype(np.float32)


def make_gradient(
    height: int = 32,
    width: int = 32,
    channels: int = 4,
    direction: str = "horizontal",
) -> np.ndarray:
    """Generate a smooth gradient image in [0, 1].

    Args:
        height: Image height.
        width: Image width.
        channels: Number of bands.  Use 0 for a 2D (H, W) output.
        direction: ``"horizontal"`` or ``"radial"``.

    Returns:
        float32 array.
    """
    if direction == "radial":
        y = np.linspace(-1, 1, height, dtype=np.float32)
        x = np.linspace(-1, 1, width, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        grad = np.sqrt(xx**2 + yy**2)
        grad = grad / max(grad.max(), 1e-8)
    else:
        grad = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(
            height, axis=0
        )
    grad = np.clip(grad, 0.0, 1.0)
    if channels == 0:
        return grad
    return np.stack([grad] * channels, axis=-1)


def make_random_mask(
    height: int = 32,
    width: int = 32,
    gap_fraction: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate a random boolean gap mask.

    Args:
        height: Image height.
        width: Image width.
        gap_fraction: Fraction of pixels to mark as gaps.
        seed: Random seed for reproducibility.

    Returns:
        2D bool array where True marks gap pixels.
    """
    rng = np.random.default_rng(seed)
    return rng.random((height, width)) < gap_fraction


def make_degraded(
    clean: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Apply a gap mask to a clean image.

    Args:
        clean: Reference image, any shape.
        mask: 2D boolean gap mask.
        fill_value: Value to assign to gap pixels.

    Returns:
        Degraded copy of *clean* with gaps set to *fill_value*.
    """
    degraded = clean.copy()
    if degraded.ndim == 3:
        degraded[mask] = fill_value
    else:
        degraded[mask] = fill_value
    return degraded


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def synthetic_clean_3d() -> np.ndarray:
    """A 32x32x4 smooth gradient image in [0, 1]."""
    return make_gradient(32, 32, 4, direction="horizontal")


@pytest.fixture()
def synthetic_clean_2d() -> np.ndarray:
    """A 32x32 smooth gradient image in [0, 1]."""
    return make_gradient(32, 32, 0, direction="horizontal")


@pytest.fixture()
def synthetic_mask() -> np.ndarray:
    """A 32x32 random gap mask with ~30% missing."""
    return make_random_mask(32, 32, gap_fraction=0.3, seed=42)


@pytest.fixture()
def synthetic_mask_all_gap() -> np.ndarray:
    """A 32x32 mask where every pixel is a gap."""
    return np.ones((32, 32), dtype=bool)


@pytest.fixture()
def synthetic_mask_no_gap() -> np.ndarray:
    """A 32x32 mask with no gaps (all valid)."""
    return np.zeros((32, 32), dtype=bool)


@pytest.fixture()
def synthetic_mask_single_pixel() -> np.ndarray:
    """A 32x32 mask with exactly one gap pixel at center."""
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 16] = True
    return mask


@pytest.fixture()
def synthetic_degraded_3d(
    synthetic_clean_3d: np.ndarray,
    synthetic_mask: np.ndarray,
) -> np.ndarray:
    """Degraded version of the 3D gradient image."""
    return make_degraded(synthetic_clean_3d, synthetic_mask)


@pytest.fixture()
def synthetic_degraded_2d(
    synthetic_clean_2d: np.ndarray,
    synthetic_mask: np.ndarray,
) -> np.ndarray:
    """Degraded version of the 2D gradient image."""
    return make_degraded(synthetic_clean_2d, synthetic_mask)


@pytest.fixture()
def checkerboard_3d() -> np.ndarray:
    """A 32x32x4 checkerboard pattern."""
    return make_checkerboard(32, 32, 4)


@pytest.fixture()
def checkerboard_2d() -> np.ndarray:
    """A 32x32 checkerboard pattern."""
    return make_checkerboard(32, 32, 0)


# ---------------------------------------------------------------------------
# Metric helpers (pure numpy, no dependency on production code)
# ---------------------------------------------------------------------------
def compute_psnr(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB) between two images."""
    mse = float(np.mean((reference - reconstructed) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compute_rmse(reference: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root Mean Squared Error between two images."""
    return float(np.sqrt(np.mean((reference - reconstructed) ** 2)))


# ---------------------------------------------------------------------------
# Real-data fixtures (graceful skip when data is absent)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchSample:
    """A single satellite patch with all noise variants."""

    patch_id: int
    satellite: str
    split: str
    bands: str
    gap_fraction: float
    clean: np.ndarray
    mask: np.ndarray
    degraded_inf: np.ndarray
    degraded_40: np.ndarray
    degraded_30: np.ndarray
    degraded_20: np.ndarray


def _normalise(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(vmax - vmin, 1e-8)
    out = (arr - vmin) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _load_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open() as fh:
        return list(csv.DictReader(fh))


def _load_patch(row: dict[str, str]) -> PatchSample | None:
    clean_path = PREPROCESSED_DIR / row["clean_path"]
    mask_path = PREPROCESSED_DIR / row["mask_path"]
    if not clean_path.exists() or not mask_path.exists():
        return None

    clean = np.load(clean_path)
    mask_arr = np.load(mask_path)

    variants: dict[str, np.ndarray] = {}
    for key in (
        "degraded_inf",
        "degraded_40",
        "degraded_30",
        "degraded_20",
    ):
        col = f"{key}_path"
        p = PREPROCESSED_DIR / row[col]
        if not p.exists():
            return None
        arr = np.load(p)
        arr = np.nan_to_num(arr, nan=0.0)
        variants[key] = arr

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
        mask=mask_arr,
        degraded_inf=_normalise(variants["degraded_inf"], vmin, vmax),
        degraded_40=_normalise(variants["degraded_40"], vmin, vmax),
        degraded_30=_normalise(variants["degraded_30"], vmin, vmax),
        degraded_20=_normalise(variants["degraded_20"], vmin, vmax),
    )


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


@pytest.fixture(scope="session")
def all_patches() -> list[PatchSample]:
    """All available preprocessed patches (skip if absent)."""
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


# Keep backward compatibility for existing test references
NOISE_VARIANTS = ("inf", "40", "30", "20")


def degraded_for_variant(patch: PatchSample, variant: str) -> np.ndarray:
    """Return the degraded array for a named noise variant."""
    return {
        "inf": patch.degraded_inf,
        "40": patch.degraded_40,
        "30": patch.degraded_30,
        "20": patch.degraded_20,
    }[variant]


# Backward-compatible aliases
psnr = compute_psnr
rmse = compute_rmse
