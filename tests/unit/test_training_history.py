"""Unit tests for TrainingHistory and compute_validation_metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dl_models.shared.metrics import compute_validation_metrics
from dl_models.shared.trainer import TrainingHistory


@pytest.mark.unit
class TestTrainingHistory:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        history = TrainingHistory(
            "ae",
            tmp_path,
            metadata={"lr": 0.001, "epochs": 50},
        )
        history.record({"epoch": 1, "train_loss": 0.5, "val_loss": 0.6})
        history.record({"epoch": 2, "train_loss": 0.4, "val_loss": 0.5})

        loaded = TrainingHistory.load(history.path)

        assert loaded.model_name == "ae"
        assert loaded.metadata == {"lr": 0.001, "epochs": 50}
        assert len(loaded.epochs) == 2
        assert loaded.epochs[0]["epoch"] == 1
        assert loaded.epochs[1]["val_loss"] == 0.5

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        history = TrainingHistory("vae", nested)
        history.record({"epoch": 1})

        assert history.path.exists()

    def test_path_property(self, tmp_path: Path) -> None:
        history = TrainingHistory("gan", tmp_path)
        assert history.path == tmp_path / "gan_history.json"

    def test_json_valid(self, tmp_path: Path) -> None:
        history = TrainingHistory("vit", tmp_path)
        history.record({"epoch": 1, "lr": 1e-4})

        data = json.loads(history.path.read_text())
        assert data["model_name"] == "vit"
        assert isinstance(data["epochs"], list)
        assert data["epochs"][0]["lr"] == 1e-4

    def test_incremental_save(self, tmp_path: Path) -> None:
        history = TrainingHistory("ae", tmp_path)
        history.record({"epoch": 1})

        data1 = json.loads(history.path.read_text())
        assert len(data1["epochs"]) == 1

        history.record({"epoch": 2})
        data2 = json.loads(history.path.read_text())
        assert len(data2["epochs"]) == 2


@pytest.mark.unit
class TestComputeValidationMetrics:
    def _make_batch(
        self,
        b: int = 4,
        c: int = 1,
        h: int = 32,
        w: int = 32,
        noise: float = 0.01,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Create synthetic pred/target/mask batches."""
        target = torch.rand(b, c, h, w)
        pred = target + noise * torch.randn_like(target)
        pred = pred.clamp(0, 1)
        mask = torch.ones(b, h, w)
        return [pred], [target], [mask]

    def test_returns_expected_keys(self) -> None:
        preds, targets, masks = self._make_batch()
        result = compute_validation_metrics(preds, targets, masks)

        expected_keys = {
            "val_psnr",
            "val_ssim",
            "val_rmse",
            "val_pixel_acc_002",
            "val_f1_002",
            "val_pixel_acc_005",
            "val_f1_005",
            "val_pixel_acc_01",
            "val_f1_01",
        }
        assert set(result.keys()) == expected_keys

    def test_reasonable_values(self) -> None:
        preds, targets, masks = self._make_batch(noise=0.01)
        result = compute_validation_metrics(preds, targets, masks)

        assert result["val_psnr"] > 20.0
        assert 0.0 <= result["val_ssim"] <= 1.0
        assert result["val_rmse"] > 0.0
        assert result["val_rmse"] < 1.0

    def test_identical_tensors_perfect_accuracy(self) -> None:
        b, c, h, w = 2, 1, 16, 16
        target = torch.rand(b, c, h, w)
        pred = target.clone()
        mask = torch.ones(b, h, w)

        result = compute_validation_metrics([pred], [target], [mask])

        assert result["val_pixel_acc_002"] == 1.0
        assert result["val_pixel_acc_005"] == 1.0
        assert result["val_pixel_acc_01"] == 1.0
        assert result["val_rmse"] < 1e-6

    def test_f1_derivation_from_pixel_accuracy(self) -> None:
        preds, targets, masks = self._make_batch(noise=0.03)
        result = compute_validation_metrics(preds, targets, masks)

        for tau_key in ["002", "005", "01"]:
            pa = result[f"val_pixel_acc_{tau_key}"]
            f1 = result[f"val_f1_{tau_key}"]
            expected_f1 = 2 * pa / (1 + pa) if (1 + pa) > 0 else 0.0
            assert abs(f1 - expected_f1) < 1e-6

    def test_multichannel_input(self) -> None:
        preds, targets, masks = self._make_batch(c=3, noise=0.02)
        result = compute_validation_metrics(preds, targets, masks)

        assert result["val_psnr"] > 0.0
        assert result["val_ssim"] > 0.0

    def test_ssim_subsample_cap(self) -> None:
        """With many samples, SSIM should still compute (subsampled)."""
        b, c, h, w = 16, 1, 16, 16
        target = torch.rand(b, c, h, w)
        pred = target + 0.01 * torch.randn_like(target)
        pred = pred.clamp(0, 1)
        mask = torch.ones(b, h, w)

        batches = 5
        result = compute_validation_metrics(
            [pred] * batches,
            [target] * batches,
            [mask] * batches,
            max_ssim_samples=64,
        )
        assert result["val_ssim"] > 0.0

    def test_empty_input(self) -> None:
        result = compute_validation_metrics([], [], [])
        assert result["val_psnr"] == 0.0
        assert result["val_ssim"] == 0.0
        assert result["val_rmse"] == 0.0
