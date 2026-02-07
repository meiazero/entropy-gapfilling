"""Unit tests for select_local_points helper."""

from __future__ import annotations

import numpy as np

from pdi_pipeline.methods._scattered_data import select_local_points


def _make_grid(
    h: int, w: int, gap_center: tuple[int, int], gap_radius: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (valid_y, valid_x, gap_y, gap_x) for a centered gap."""
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = gap_center
    y_lo = max(0, cy - gap_radius)
    y_hi = min(h, cy + gap_radius + 1)
    x_lo = max(0, cx - gap_radius)
    x_hi = min(w, cx + gap_radius + 1)
    mask[y_lo:y_hi, x_lo:x_hi] = True
    valid_y, valid_x = np.where(~mask)
    gap_y, gap_x = np.where(mask)
    return valid_y, valid_x, gap_y, gap_x


class TestSelectLocalPoints:
    def test_local_subset_smaller_than_all(self) -> None:
        valid_y, valid_x, gap_y, gap_x = _make_grid(64, 64, (32, 32), 4)
        local_y, local_x = select_local_points(
            valid_y,
            valid_x,
            gap_y,
            gap_x,
            64,
            64,
            kernel_size=10,
            max_points=5000,
        )
        assert len(local_y) < len(valid_y)
        assert len(local_y) == len(local_x)

    def test_fallback_when_too_few_local(self) -> None:
        valid_y = np.array([0, 0, 63, 63])
        valid_x = np.array([0, 63, 0, 63])
        gap_y = np.array([32])
        gap_x = np.array([32])
        local_y, _local_x = select_local_points(
            valid_y,
            valid_x,
            gap_y,
            gap_x,
            64,
            64,
            kernel_size=2,
            max_points=5000,
        )
        assert len(local_y) == 4

    def test_downsampling_when_exceeds_max(self) -> None:
        valid_y, valid_x, gap_y, gap_x = _make_grid(64, 64, (32, 32), 2)
        local_y, _local_x = select_local_points(
            valid_y,
            valid_x,
            gap_y,
            gap_x,
            64,
            64,
            kernel_size=None,
            max_points=50,
        )
        assert len(local_y) <= 50

    def test_none_kernel_uses_full_image(self) -> None:
        valid_y, valid_x, gap_y, gap_x = _make_grid(64, 64, (32, 32), 2)
        local_y, _ = select_local_points(
            valid_y,
            valid_x,
            gap_y,
            gap_x,
            64,
            64,
            kernel_size=None,
            max_points=50000,
        )
        assert len(local_y) == len(valid_y)
