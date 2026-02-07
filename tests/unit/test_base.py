"""Unit tests for BaseMethod static helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pdi_pipeline.methods.base import BaseMethod


# We cannot instantiate BaseMethod directly; create a trivial concrete subclass.
class _Stub(BaseMethod):
    name = "stub"

    def apply(self, degraded, mask, *, meta=None):
        return degraded


# ── _normalize_mask ──────────────────────────────────────────────────────────


class TestNormalizeMask:
    def test_2d_float_ones_zeros(self) -> None:
        mask = np.array([[0, 1], [1, 0]], dtype=np.float32)
        result = BaseMethod._normalize_mask(mask)
        assert result.dtype == bool
        assert result.shape == (2, 2)
        assert result[0, 1] is np.True_
        assert result[0, 0] is np.False_

    def test_2d_bool_passthrough(self) -> None:
        mask = np.array([[True, False], [False, True]])
        result = BaseMethod._normalize_mask(mask)
        np.testing.assert_array_equal(result, mask)

    def test_3d_any_axis2_reduction(self) -> None:
        # (H, W, C) = (2, 2, 3); channel 0 has a gap at (0, 1)
        mask_3d = np.zeros((2, 2, 3), dtype=np.float32)
        mask_3d[0, 1, 0] = 1.0
        result = BaseMethod._normalize_mask(mask_3d)
        assert result.shape == (2, 2)
        assert result[0, 1] is np.True_
        assert result[0, 0] is np.False_

    def test_4d_raises(self) -> None:
        with pytest.raises(ValueError, match="2D or 3D"):
            BaseMethod._normalize_mask(np.zeros((2, 2, 3, 4)))

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="2D or 3D"):
            BaseMethod._normalize_mask(np.zeros((10,)))


# ── _finalize ────────────────────────────────────────────────────────────────


class TestFinalize:
    def test_dtype_is_float32(self) -> None:
        arr = np.array([[0.5, 0.7]], dtype=np.float64)
        result = BaseMethod._finalize(arr)
        assert result.dtype == np.float32

    def test_default_clip_01(self) -> None:
        arr = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)
        result = BaseMethod._finalize(arr)
        np.testing.assert_array_equal(result, [[0.0, 0.5, 1.0]])

    def test_custom_clip_range(self) -> None:
        arr = np.array([[-5.0, 50.0, 200.0]], dtype=np.float32)
        result = BaseMethod._finalize(arr, clip_range=(0.0, 100.0))
        np.testing.assert_array_equal(result, [[0.0, 50.0, 100.0]])

    def test_no_clip(self) -> None:
        arr = np.array([[-5.0, 200.0]], dtype=np.float32)
        result = BaseMethod._finalize(arr, clip_range=None)
        np.testing.assert_array_equal(result, [[-5.0, 200.0]])

    def test_nan_replaced(self) -> None:
        arr = np.array([[np.nan, 0.5]], dtype=np.float32)
        result = BaseMethod._finalize(arr)
        assert result[0, 0] == 0.0
        assert result[0, 1] == pytest.approx(0.5)

    def test_inf_replaced(self) -> None:
        arr = np.array([[np.inf, -np.inf]], dtype=np.float32)
        result = BaseMethod._finalize(arr)
        assert result[0, 0] == 1.0  # posinf -> clip_max
        assert result[0, 1] == 0.0  # neginf -> clip_min

    def test_empty_array(self) -> None:
        arr = np.array([], dtype=np.float32).reshape(0, 0)
        result = BaseMethod._finalize(arr)
        assert result.shape == (0, 0)


# ── _apply_channelwise ──────────────────────────────────────────────────────


class TestApplyChannelwise:
    @staticmethod
    def _double_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        return ch * 2.0

    def test_single_channel(self) -> None:
        img = np.ones((4, 4), dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        result = BaseMethod._apply_channelwise(img, mask, self._double_fn)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, 2.0)

    def test_multichannel_independent(self) -> None:
        img = np.stack(
            [np.full((4, 4), v, dtype=np.float32) for v in (1, 2, 3)],
            axis=2,
        )
        mask = np.zeros((4, 4), dtype=bool)
        result = BaseMethod._apply_channelwise(img, mask, self._double_fn)
        assert result.shape == (4, 4, 3)
        np.testing.assert_allclose(result[:, :, 0], 2.0)
        np.testing.assert_allclose(result[:, :, 1], 4.0)
        np.testing.assert_allclose(result[:, :, 2], 6.0)

    def test_mask_passed_through(self) -> None:
        """Verify the mask argument reaches the channel function."""
        received_masks: list[np.ndarray] = []

        def _record(ch: np.ndarray, m: np.ndarray) -> np.ndarray:
            received_masks.append(m)
            return ch

        img = np.ones((4, 4, 2), dtype=np.float32)
        mask = np.eye(4, dtype=bool)
        BaseMethod._apply_channelwise(img, mask, _record)
        assert len(received_masks) == 2
        for m in received_masks:
            np.testing.assert_array_equal(m, mask)


# ── fit (default no-op) ─────────────────────────────────────────────────────


class TestFit:
    def test_returns_self(self) -> None:
        stub = _Stub()
        assert stub.fit() is stub
