"""Lazy-loading dataset for preprocessed satellite patches.

PatchDataset reads the manifest CSV once at construction and loads NPY
arrays on demand via __getitem__. This keeps memory usage bounded even
for the full 77k-patch corpus (~4.7 GB if loaded at once).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pdi_pipeline.preprocessing import (
    compute_normalize_range,
    ensure_mask_2d,
    normalize_image,
    replace_nan,
)


@dataclass(frozen=True)
class PatchSample:
    """A single satellite patch with its degraded variant and metadata."""

    patch_id: int
    satellite: str
    split: str
    bands: str
    gap_fraction: float
    acquisition_date: str
    clean: np.ndarray  # (H, W, C) float32 [0, 1]
    mask: np.ndarray  # (H, W) float32, 1=gap
    degraded: np.ndarray  # (H, W, C) float32 [0, 1]
    noise_level: str  # "inf", "40", "30", "20"
    mean_entropy: dict[str, float] | None = None  # e.g. {"entropy_7": 3.45}


_NOISE_COL = {
    "inf": "degraded_inf_path",
    "40": "degraded_40_path",
    "30": "degraded_30_path",
    "20": "degraded_20_path",
}


class PatchDataset:
    """Lazy-loading dataset over preprocessed satellite patches.

    Filters rows from the manifest at init time by split, satellite, and
    noise level. Individual patches are loaded from disk only when
    accessed via __getitem__.

    Args:
        manifest_path: Path to the manifest CSV file.
        split: Dataset split to use (e.g. "test", "train", "val").
        satellite: Satellite filter (e.g. "sentinel2", "landsat8").
            Use None to include all satellites.
        noise_level: Noise variant to load ("inf", "40", "30", "20").
        max_patches: If set, truncate to this many patches.
        seed: Random seed for reproducible truncation ordering.
        selected_ids: Pre-computed list of patch IDs to use. When
            provided, overrides ``max_patches``/``seed`` sampling and
            filters to exactly these IDs.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        satellite: str | None = None,
        noise_level: str = "inf",
        max_patches: int | None = None,
        seed: int = 42,
        selected_ids: list[int] | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            msg = f"Manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)

        if noise_level not in _NOISE_COL:
            msg = f"Invalid noise_level: {noise_level!r}"
            raise ValueError(msg)

        self._base_dir = manifest_path.parent
        self._noise_level = noise_level
        self._noise_col = _NOISE_COL[noise_level]

        with manifest_path.open() as fh:
            rows = list(csv.DictReader(fh))

        # Filter by split
        rows = [r for r in rows if r["split"] == split]

        # Filter by satellite
        if satellite is not None:
            rows = [r for r in rows if r["satellite"] == satellite]

        # Deterministic ordering by patch_id
        rows.sort(key=lambda r: int(r["patch_id"]))

        # Use pre-computed selection or sample
        if selected_ids is not None:
            id_set = set(selected_ids)
            rows = [r for r in rows if int(r["patch_id"]) in id_set]
        elif max_patches is not None and len(rows) > max_patches:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(rows), size=max_patches, replace=False)
            idx.sort()
            rows = [rows[i] for i in idx]

        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> PatchSample:
        if idx < 0 or idx >= len(self._rows):
            msg = f"Index {idx} out of range for dataset of size {len(self)}"
            raise IndexError(msg)

        row = self._rows[idx]
        clean_path = self._base_dir / row["clean_path"]
        mask_path = self._base_dir / row["mask_path"]
        degraded_path = self._base_dir / row[self._noise_col]

        clean = np.load(clean_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        degraded = np.load(degraded_path).astype(np.float32)

        # Replace NaN in degraded with 0 before normalizing
        degraded = replace_nan(degraded)

        norm_range = compute_normalize_range(clean)
        clean = normalize_image(clean, norm_range)
        degraded = normalize_image(degraded, norm_range)

        mask = ensure_mask_2d(mask)

        # Extract precomputed mean entropy values from manifest
        entropy_dict: dict[str, float] = {}
        for key, val in row.items():
            if key.startswith("mean_entropy_") and val:
                try:
                    ws = key.replace("mean_entropy_", "")
                    entropy_dict[f"entropy_{ws}"] = float(val)
                except (ValueError, TypeError):
                    pass

        return PatchSample(
            patch_id=int(row["patch_id"]),
            satellite=row["satellite"],
            split=row["split"],
            bands=row.get("bands", ""),
            gap_fraction=float(row["gap_fraction"]),
            acquisition_date=row.get("acquisition_date", ""),
            clean=clean,
            mask=mask,
            degraded=degraded,
            noise_level=self._noise_level,
            mean_entropy=entropy_dict if entropy_dict else None,
        )

    def __iter__(self) -> Iterator[PatchSample]:
        for i in range(len(self)):
            yield self[i]

    @property
    def patch_ids(self) -> list[int]:
        """All patch IDs in this dataset, in order."""
        return [int(r["patch_id"]) for r in self._rows]

    @property
    def satellites(self) -> list[str]:
        """Unique satellite names in this dataset."""
        return sorted({r["satellite"] for r in self._rows})
