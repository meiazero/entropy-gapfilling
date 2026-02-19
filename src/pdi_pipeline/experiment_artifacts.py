"""Utilities for selecting and saving reconstruction artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pdi_pipeline.config import ExperimentConfig
from pdi_pipeline.visualization import save_array_as_png

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntropyExtremes:
    """Top and bottom entropy patch IDs for a noise level."""

    high_ids: set[int]
    low_ids: set[int]
    top_k: int

    @property
    def all_ids(self) -> set[int]:
        return self.high_ids | self.low_ids


def compute_entropy_extremes(
    manifest_path: Path,
    config: ExperimentConfig,
    selections: dict[str, dict[int, list[int]]] | None,
    top_k: int,
) -> dict[str, EntropyExtremes]:
    """Compute top-k high/low entropy patch IDs per noise level.

    Uses mean entropy values from the manifest. When multiple entropy
    windows are available, the mean across windows is used.
    """
    if top_k <= 0:
        return {}

    entropy_cols = [
        f"mean_entropy_{ws}"
        for ws in config.entropy_windows
        if f"mean_entropy_{ws}" in _manifest_columns(manifest_path)
    ]
    if not entropy_cols:
        logger.warning(
            "No mean entropy columns found in manifest. "
            "Run scripts/precompute_entropy.py to populate mean_entropy_* "
            "before selecting entropy extremes."
        )
        return {}

    usecols = ["patch_id", "satellite", "split", *entropy_cols]
    manifest = pd.read_csv(manifest_path, usecols=usecols)
    manifest = manifest[manifest["split"] == "test"]
    manifest = manifest[manifest["satellite"].isin(config.satellites)]
    manifest = manifest.sort_values("patch_id").reset_index(drop=True)

    patch_ids_by_noise: dict[str, set[int]] = {
        noise: set() for noise in config.noise_levels
    }
    for seed in config.seeds:
        for noise_level in config.noise_levels:
            for satellite in config.satellites:
                selected_ids = (
                    selections.get(satellite, {}).get(seed)
                    if selections
                    else None
                )
                patch_ids = _select_patch_ids(
                    manifest,
                    satellite,
                    seed,
                    config.max_patches,
                    selected_ids,
                )
                patch_ids_by_noise[noise_level].update(patch_ids)

    manifest["entropy_score"] = manifest[entropy_cols].mean(axis=1, skipna=True)

    extremes_by_noise: dict[str, EntropyExtremes] = {}
    for noise_level, patch_ids in patch_ids_by_noise.items():
        if not patch_ids:
            continue
        subset = manifest[manifest["patch_id"].isin(patch_ids)]
        subset = subset.dropna(subset=["entropy_score"])
        if subset.empty:
            continue

        high = (
            subset.nlargest(top_k, "entropy_score")["patch_id"]
            .astype(int)
            .tolist()
        )
        low = (
            subset.nsmallest(top_k, "entropy_score")["patch_id"]
            .astype(int)
            .tolist()
        )
        extremes_by_noise[noise_level] = EntropyExtremes(
            high_ids=set(high),
            low_ids=set(low),
            top_k=top_k,
        )

    return extremes_by_noise


def write_entropy_extremes_manifest(
    output_dir: Path,
    extremes_by_noise: dict[str, EntropyExtremes],
) -> None:
    """Write the selected entropy extremes to a JSON file."""
    if not extremes_by_noise:
        return

    payload = {}
    for noise_level, extremes in extremes_by_noise.items():
        payload[noise_level] = {
            "top_k": extremes.top_k,
            "high": sorted(extremes.high_ids),
            "low": sorted(extremes.low_ids),
        }

    path = output_dir / "entropy_extremes.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Entropy extremes saved to: %s", path)


@dataclass
class ReconstructionManager:
    """Save reconstruction artifacts for selected patches."""

    output_dir: Path
    entropy_extremes: dict[str, EntropyExtremes] = field(default_factory=dict)
    save_first_n: int = 0
    first_seed: int | None = None
    save_png_noise: str = "inf"

    _saved_first_counts: dict[tuple[str, str], int] = field(
        default_factory=dict
    )
    _saved_png_counts: dict[str, int] = field(default_factory=dict)
    _saved_references: set[tuple[str, int]] = field(default_factory=set)

    def maybe_save_first(
        self,
        seed: int,
        noise_level: str,
        method: str,
        patch_id: int,
        result: np.ndarray,
    ) -> None:
        if self.save_first_n <= 0:
            return
        if self.first_seed is not None and seed != self.first_seed:
            return

        npy_dir = (
            self.output_dir / "reconstruction_arrays" / noise_level / method
        )
        npy_dir.mkdir(parents=True, exist_ok=True)
        npy_key = (noise_level, method)
        if npy_key not in self._saved_first_counts:
            self._saved_first_counts[npy_key] = (
                sum(1 for _ in npy_dir.glob("*.npy")) if npy_dir.exists() else 0
            )
        if self._saved_first_counts[npy_key] >= self.save_first_n:
            return

        np.save(npy_dir / f"{patch_id:07d}.npy", result)
        self._saved_first_counts[npy_key] += 1

        if noise_level == self.save_png_noise:
            img_dir = self.output_dir / "reconstruction_images" / method
            if method not in self._saved_png_counts:
                self._saved_png_counts[method] = (
                    sum(1 for _ in img_dir.glob("*.png"))
                    if img_dir.exists()
                    else 0
                )
            if self._saved_png_counts[method] < self.save_first_n:
                save_array_as_png(
                    result,
                    img_dir / f"{patch_id:07d}.png",
                )
                self._saved_png_counts[method] += 1

    def maybe_save_entropy_extreme(
        self,
        noise_level: str,
        method: str,
        patch_id: int,
        result: np.ndarray,
        clean: np.ndarray,
        degraded: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        extremes = self.entropy_extremes.get(noise_level)
        if extremes is None or patch_id not in extremes.all_ids:
            return

        npy_dir = (
            self.output_dir / "reconstruction_arrays" / noise_level / method
        )
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / f"{patch_id:07d}.npy", result)

        if noise_level == self.save_png_noise:
            img_dir = self.output_dir / "reconstruction_images" / method
            img_dir.mkdir(parents=True, exist_ok=True)
            save_array_as_png(result, img_dir / f"{patch_id:07d}.png")

        self._save_reference(noise_level, patch_id, clean, degraded, mask)

    def _save_reference(
        self,
        noise_level: str,
        patch_id: int,
        clean: np.ndarray,
        degraded: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        ref_key = (noise_level, patch_id)
        if ref_key in self._saved_references:
            return

        ref_npy_dir = (
            self.output_dir
            / "reconstruction_arrays"
            / noise_level
            / "_reference"
        )
        ref_npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(ref_npy_dir / f"{patch_id:07d}_clean.npy", clean)
        np.save(ref_npy_dir / f"{patch_id:07d}_degraded.npy", degraded)
        np.save(ref_npy_dir / f"{patch_id:07d}_mask.npy", mask)

        if noise_level == self.save_png_noise:
            ref_dir = self.output_dir / "reconstruction_images" / "_reference"
            ref_dir.mkdir(parents=True, exist_ok=True)
            clean_png = ref_dir / f"{patch_id:07d}_clean.png"
            if not clean_png.exists():
                save_array_as_png(clean, clean_png)
                save_array_as_png(
                    degraded, ref_dir / f"{patch_id:07d}_degraded.png"
                )
                save_array_as_png(mask, ref_dir / f"{patch_id:07d}_mask.png")

        self._saved_references.add(ref_key)


def _select_patch_ids(
    manifest: pd.DataFrame,
    satellite: str,
    seed: int,
    max_patches: int | None,
    selected_ids: list[int] | None,
) -> list[int]:
    filtered = manifest[manifest["satellite"] == satellite]
    filtered = filtered.sort_values("patch_id").reset_index(drop=True)

    if selected_ids is not None:
        id_set = set(selected_ids)
        filtered = filtered[filtered["patch_id"].isin(id_set)]
    elif max_patches is not None and len(filtered) > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(filtered), size=max_patches, replace=False)
        idx.sort()
        filtered = filtered.iloc[idx]

    return filtered["patch_id"].astype(int).tolist()


def _manifest_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns)
