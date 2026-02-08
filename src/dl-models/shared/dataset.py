"""PyTorch Dataset wrapping PatchDataset for DL model training.

Returns (input_tensor, clean_tensor, mask_tensor) triples where
input_tensor = cat([degraded * (1 - mask), mask], dim=0) -> (C+1, H, W).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from pdi_pipeline.dataset import PatchDataset


class InpaintingDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset adapter for the gap-filling training loop.

    Wraps PatchDataset and converts samples to tensors in the
    (C+1, H, W) input convention used by all DL models.

    Args:
        manifest_path: Path to the manifest CSV.
        split: Dataset split ("train", "val", "test").
        satellite: Satellite filter, or None for all.
        noise_level: Noise variant to load.
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
        self._ds = PatchDataset(
            manifest_path,
            split=split,
            satellite=satellite,
            noise_level=noise_level,
            max_patches=max_patches,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self._ds[idx]

        degraded = sample.degraded  # (H, W, C)
        clean = sample.clean  # (H, W, C)
        mask = sample.mask  # (H, W)

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
