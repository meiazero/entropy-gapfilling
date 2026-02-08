"""Contract tests for all 4 deep learning inpainting models.

Validates output shape, dtype, range [0, 1], and valid pixel
preservation for each model without requiring trained checkpoints.

These tests are isolated from the main pipeline test suite.
The DL models are not part of the experiment battery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add dl-models root to import path
_DL_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "dl-models"
if str(_DL_ROOT) not in sys.path:
    sys.path.insert(0, str(_DL_ROOT))

from ae.model import AEInpainting
from gan.model import GANInpainting
from transformer.model import TransformerInpainting
from vae.model import VAEInpainting


@pytest.fixture()
def synthetic_patch() -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 64x64x4 patch with a gap mask.

    Returns:
        (degraded, mask) where degraded is (64, 64, 4) float32 in [0, 1]
        and mask is (64, 64) float32 with 1=gap.
    """
    rng = np.random.default_rng(42)
    clean = rng.random((64, 64, 4)).astype(np.float32)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0  # 20x20 square gap
    degraded = clean.copy()
    degraded[mask > 0.5] = 0.0
    return degraded, mask


ALL_MODELS = [
    AEInpainting,
    VAEInpainting,
    GANInpainting,
    TransformerInpainting,
]

MODEL_NAMES = [
    "ae_inpainting",
    "vae_inpainting",
    "gan_inpainting",
    "transformer_inpainting",
]


@pytest.mark.unit
class TestDLModelContracts:
    """Contract tests applying to all DL inpainting models."""

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODELS,
        ids=MODEL_NAMES,
    )
    def test_output_shape(
        self,
        model_cls: type,
        synthetic_patch: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Output must match input spatial shape."""
        degraded, mask = synthetic_patch
        model = model_cls(device="cpu")
        result = model.apply(degraded, mask)
        assert result.shape == degraded.shape

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODELS,
        ids=MODEL_NAMES,
    )
    def test_output_dtype(
        self,
        model_cls: type,
        synthetic_patch: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Output must be float32."""
        degraded, mask = synthetic_patch
        model = model_cls(device="cpu")
        result = model.apply(degraded, mask)
        assert result.dtype == np.float32

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODELS,
        ids=MODEL_NAMES,
    )
    def test_output_range(
        self,
        model_cls: type,
        synthetic_patch: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Output must be in [0, 1] range."""
        degraded, mask = synthetic_patch
        model = model_cls(device="cpu")
        result = model.apply(degraded, mask)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODELS,
        ids=MODEL_NAMES,
    )
    def test_valid_pixels_preserved(
        self,
        model_cls: type,
        synthetic_patch: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Valid (non-gap) pixels must be preserved exactly."""
        degraded, mask = synthetic_patch
        model = model_cls(device="cpu")
        result = model.apply(degraded, mask)
        valid = mask < 0.5
        np.testing.assert_allclose(result[valid], degraded[valid], atol=1e-5)

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODELS,
        ids=MODEL_NAMES,
    )
    def test_no_nan_in_output(
        self,
        model_cls: type,
        synthetic_patch: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Output must contain no NaN values."""
        degraded, mask = synthetic_patch
        model = model_cls(device="cpu")
        result = model.apply(degraded, mask)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize(
        "model_cls,expected_name",
        list(zip(ALL_MODELS, MODEL_NAMES)),
        ids=MODEL_NAMES,
    )
    def test_model_name(self, model_cls: type, expected_name: str) -> None:
        """Each model must have the correct name attribute."""
        assert model_cls.name == expected_name


@pytest.mark.unit
class TestDLModel2D:
    """Test that DL models handle 2D (single-channel) input."""

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODELS,
        ids=MODEL_NAMES,
    )
    def test_2d_input(self, model_cls: type) -> None:
        """Models should handle (H, W) input gracefully."""
        rng = np.random.default_rng(99)
        degraded_2d = rng.random((64, 64)).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:20, 10:20] = 1.0

        # Override channels for 2D input: in_channels=2 (1 img + 1 mask), out=1
        model = model_cls(device="cpu", in_channels=2, out_channels=1)
        result = model.apply(degraded_2d, mask)
        assert result.shape == degraded_2d.shape
        assert result.dtype == np.float32
