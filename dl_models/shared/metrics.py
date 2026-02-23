"""Quality metrics for gap-filling evaluation.

Standalone implementations using numpy and skimage only.
No imports from pdi_pipeline.metrics.

All metrics are computed strictly on gap pixels (mask=1).
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

_MSE_FLOOR = 1e-12
_PIXEL_ACC_THRESHOLDS = {
    "002": 0.02,
    "005": 0.05,
    "01": 0.10,
}


def _gap_diff(
    clean: np.ndarray,
    pred: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel difference on gap pixels only.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        pred: Predicted image, same shape.
        gap: Boolean mask, (H, W), True=gap.

    Returns:
        1-D array of ``clean - pred`` values at gap locations.
    """
    c = clean.astype(np.float64)
    p = pred.astype(np.float64)
    if c.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], c.shape)
        return c[gap_3d] - p[gap_3d]
    return c[gap] - p[gap]


def psnr(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """PSNR on gap pixels only.

    PSNR = 10 * log10(1.0 / MSE) where MSE is over gap pixels.
    Data range assumed to be [0, 1].

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        pred: Predicted image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        PSNR in dB. Returns inf if MSE is effectively zero.
    """
    gap = mask > 0.5
    if not np.any(gap):
        return float("inf")

    diff = _gap_diff(clean, pred, gap)
    mse = float(np.mean(diff**2))
    if mse < _MSE_FLOOR:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """SSIM map masked to gap pixels only.

    Computes the full SSIM map via skimage, then averages only the
    values at gap pixel locations.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        pred: Predicted image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        Mean SSIM value restricted to gap region.
    """
    gap = mask > 0.5
    if not np.any(gap):
        return 1.0

    is_multichannel = clean.ndim == 3
    _, ssim_map = structural_similarity(
        clean,
        pred,
        data_range=1.0,
        full=True,
        channel_axis=2 if is_multichannel else None,
    )

    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)

    return float(np.mean(ssim_map[gap]))


def rmse(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """RMSE on gap pixels only.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        pred: Predicted image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        RMSE value over gap pixels.
    """
    gap = mask > 0.5
    if not np.any(gap):
        return 0.0

    diff = _gap_diff(clean, pred, gap)
    return float(np.sqrt(np.mean(diff**2)))


def compute_all(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute psnr, ssim, rmse for a single patch.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        pred: Predicted image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        Dict with keys 'psnr', 'ssim', 'rmse'.
    """
    return {
        "psnr": psnr(clean, pred, mask),
        "ssim": ssim(clean, pred, mask),
        "rmse": rmse(clean, pred, mask),
    }


def _to_numpy_batches(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
    mask_batch: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_np = pred_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
    mask_np = mask_batch.detach().cpu().numpy()
    return pred_np, target_np, mask_np


def _prepare_sample(
    pred: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape[2] == 1:
        return pred[:, :, 0], target[:, :, 0]
    return pred, target


def _update_pixel_acc(
    pixel_acc: dict[str, list[float]],
    diff: np.ndarray,
    gap: np.ndarray,
) -> None:
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)
    for key, tau in _PIXEL_ACC_THRESHOLDS.items():
        pixel_acc[key].append(float(np.mean(diff[gap] <= tau)))


def _accumulate_sample_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    psnr_vals: list[float],
    ssim_vals: list[float],
    rmse_vals: list[float],
    pixel_acc: dict[str, list[float]],
    ssim_count: int,
    max_ssim_samples: int,
) -> int:
    pred, target = _prepare_sample(pred, target)
    psnr_vals.append(psnr(target, pred, mask))
    rmse_vals.append(rmse(target, pred, mask))

    gap = mask > 0.5
    if np.any(gap):
        diff = np.abs(target - pred)
        _update_pixel_acc(pixel_acc, diff, gap)
    else:
        for key in pixel_acc:
            pixel_acc[key].append(1.0)

    if ssim_count < max_ssim_samples:
        ssim_vals.append(ssim(target, pred, mask))
        return ssim_count + 1
    return ssim_count


def compute_validation_metrics(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    max_ssim_samples: int = 64,
) -> dict[str, float]:
    """Bridge torch tensors to numpy and compute quality metrics.

    Converts (B, C, H, W) tensors to numpy (H, W, C) and calls
    psnr/ssim/rmse per sample.

    Args:
        preds: List of prediction tensors, each (B, C, H, W).
        targets: List of clean reference tensors, each (B, C, H, W).
        masks: List of mask tensors, each (B, H, W) with 1=gap.
        max_ssim_samples: Cap for SSIM computation (performance).

    Returns:
        Dict with val_psnr, val_ssim, val_rmse, val_pixel_acc_*,
        and val_f1_* for multiple error thresholds.
    """
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    rmse_vals: list[float] = []
    pixel_acc: dict[str, list[float]] = {
        key: [] for key in _PIXEL_ACC_THRESHOLDS
    }

    ssim_count = 0

    for pred_batch, target_batch, mask_batch in zip(
        preds, targets, masks, strict=True
    ):
        pred_np, target_np, mask_np = _to_numpy_batches(
            pred_batch, target_batch, mask_batch
        )
        b = pred_np.shape[0]
        for i in range(b):
            ssim_count = _accumulate_sample_metrics(
                pred_np[i],
                target_np[i],
                mask_np[i],
                psnr_vals,
                ssim_vals,
                rmse_vals,
                pixel_acc,
                ssim_count,
                max_ssim_samples,
            )

    result: dict[str, float] = {}

    if psnr_vals:
        finite_psnr = [v for v in psnr_vals if np.isfinite(v)]
        result["val_psnr"] = float(np.mean(finite_psnr)) if finite_psnr else 0.0
    else:
        result["val_psnr"] = 0.0

    result["val_ssim"] = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    result["val_rmse"] = float(np.mean(rmse_vals)) if rmse_vals else 0.0

    for key, values in pixel_acc.items():
        acc = float(np.mean(values)) if values else 0.0
        result[f"val_pixel_acc_{key}"] = acc
        result[f"val_f1_{key}"] = 2 * acc / (1 + acc) if (1 + acc) > 0 else 0.0

    return result
