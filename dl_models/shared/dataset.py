"""PyTorch Dataset reading the manifest CSV directly.

No dependency on pdi_pipeline.dataset.PatchDataset.
Reads manifest CSV produced by scripts/preprocess_dataset.py and
loads NPY files on demand.

Manifest CSV columns (from preprocess_dataset.py):
    patch_id, satellite, split, clean_path, mask_path,
    degraded_inf_path, degraded_40_path, degraded_30_path,
    degraded_20_path, gap_fraction, ...

Returns (input_tensor, clean_tensor, mask_tensor) triples where
input_tensor = cat([degraded * (1 - mask), mask], dim=0) -> (C+1, H, W).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_NOISE_COL: dict[str, str] = {
    "inf": "degraded_inf_path",
    "40": "degraded_40_path",
    "30": "degraded_30_path",
    "20": "degraded_20_path",
}


def _normalize(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize array to [0, 1] using clean image range."""
    span = max(vmax - vmin, 1e-8)
    out = (arr - vmin) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class InpaintingDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset reading manifest CSV directly for gap-filling training.

    Reads the manifest CSV produced by scripts/preprocess_dataset.py
    and converts samples to tensors in the (C+1, H, W) input convention
    used by all DL models.

    Args:
        manifest_path: Path to the manifest CSV.
        split: Dataset split ("train", "val", "test").
        satellite: Satellite filter, or None for all.
        noise_level: Noise variant to load ("inf", "40", "30", "20").
        max_patches: Optional patch count limit.
        seed: Random seed for reproducible truncation.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        satellite: str | None = None,
        noise_level: str = "inf",
        max_patches: int | None = None,
        seed: int = 42,
    ) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            msg = f"Manifest not found: {manifest_path}"
            raise FileNotFoundError(msg)

        if noise_level not in _NOISE_COL:
            msg = f"Invalid noise_level: {noise_level!r}. Valid: {list(_NOISE_COL)}"
            raise ValueError(msg)

        self._base_dir = manifest_path.parent
        self._noise_col = _NOISE_COL[noise_level]

        df = pd.read_csv(manifest_path, dtype=str)

        df = df[df["split"] == split]
        if satellite is not None:
            df = df[df["satellite"] == satellite]

        # Deterministic ordering by patch_id
        df = df.sort_values("patch_id", key=lambda s: s.astype(int))

        if max_patches is not None and len(df) > max_patches:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=max_patches, replace=False)
            idx.sort()
            df = df.iloc[idx]

        self._records = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self._records.iloc[idx]

        clean_path = self._base_dir / row["clean_path"]
        mask_path = self._base_dir / row["mask_path"]
        degraded_path = self._base_dir / row[self._noise_col]

        clean = np.load(clean_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        degraded = np.load(degraded_path).astype(np.float32)

        # Replace NaN before normalizing
        degraded = np.nan_to_num(degraded, nan=0.0)

        # Normalize to [0, 1] using clean image range
        vmin = float(clean.min())
        vmax = float(clean.max())
        if vmax - vmin < 1e-8:
            vmax = vmin + 1.0
        clean = _normalize(clean, vmin, vmax)
        degraded = _normalize(degraded, vmin, vmax)

        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask_2d = (mask > 0.5).astype(np.float32)

        # Zero out gap pixels in input
        masked_input = degraded * (1.0 - mask_2d[:, :, np.newaxis])

        # (H, W, C) -> (C, H, W)
        input_t = torch.from_numpy(masked_input).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask_2d).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).permute(2, 0, 1).float()

        # Concatenate: (C+1, H, W)
        x = torch.cat([input_t, mask_t], dim=0)
        return x, clean_t, mask_t.squeeze(0)

    @property
    def patch_ids(self) -> list[int]:
        """All patch IDs in this dataset, in order."""
        return [int(v) for v in self._records["patch_id"]]


def make_loaders(
    manifest_path: str | Path,
    batch_size: int = 32,
    satellite: str | None = None,
    noise_level: str = "inf",
    max_patches: int | None = None,
    seed: int = 42,
    num_workers: int | None = None,
) -> tuple[
    torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
]:
    """Create train and validation DataLoaders with optimized settings.

    Args:
        manifest_path: Path to the manifest CSV.
        batch_size: Samples per batch.
        satellite: Satellite filter, or None for all.
        noise_level: Noise variant to load.
        max_patches: Optional limit per split.
        seed: Random seed for truncation.
        num_workers: Worker count. Defaults to min(cpu_count, 8).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)

    pin = torch.cuda.is_available()

    train_ds = InpaintingDataset(
        manifest_path,
        split="train",
        satellite=satellite,
        noise_level=noise_level,
        max_patches=max_patches,
        seed=seed,
    )
    val_ds = InpaintingDataset(
        manifest_path,
        split="val",
        satellite=satellite,
        noise_level=noise_level,
        max_patches=max_patches,
        seed=seed,
    )

    train_loader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader: torch.utils.data.DataLoader[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader
