"""Quality metrics for gap-filling evaluation.

Standalone implementations using numpy and skimage only.
No imports from pdi_pipeline.metrics.

All metrics are computed strictly on gap pixels (mask=1).
Metric set matches pdi_pipeline.metrics.compute_all() for cross-method
comparability: PSNR, SSIM, RMSE, SAM, ERGAS, per-band RMSE, pixel
accuracy and F1 at three error thresholds.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

_MSE_FLOOR = 1e-12

_PIXEL_ACC_THRESHOLDS: dict[str, float] = {
    "002": 0.02,
    "005": 0.05,
    "01": 0.10,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gap_mask(mask: np.ndarray) -> np.ndarray:
    """Return a boolean (H, W) gap mask."""
    m = mask if mask.ndim == 2 else mask[:, :, 0]
    return m > 0.5


def _gap_diff_flat(
    clean: np.ndarray,
    pred: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    """1-D array of ``clean - pred`` values at gap locations."""
    c = clean.astype(np.float64)
    p = pred.astype(np.float64)
    if c.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], c.shape)
        return c[gap_3d] - p[gap_3d]
    return c[gap] - p[gap]


def _gap_abs_diff_flat(
    clean: np.ndarray,
    pred: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    return np.abs(_gap_diff_flat(clean, pred, gap))


# ---------------------------------------------------------------------------
# Public scalar metrics (mask-on-gap)
# ---------------------------------------------------------------------------


def psnr(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """PSNR on gap pixels only. Data range assumed [0, 1]."""
    gap = _gap_mask(mask)
    if not np.any(gap):
        return float("inf")
    diff = _gap_diff_flat(clean, pred, gap)
    mse = float(np.mean(diff**2))
    if mse < _MSE_FLOOR:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean SSIM restricted to gap pixels."""
    gap = _gap_mask(mask)
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
    """RMSE on gap pixels only."""
    gap = _gap_mask(mask)
    if not np.any(gap):
        return 0.0
    diff = _gap_diff_flat(clean, pred, gap)
    return float(np.sqrt(np.mean(diff**2)))


def sam(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean Spectral Angle Mapper (degrees) on gap pixels.

    Requires multichannel input (H, W, C) with C >= 2.
    Returns 0.0 for 2-D inputs.
    """
    gap = _gap_mask(mask)
    if not np.any(gap) or clean.ndim != 3 or clean.shape[2] < 2:
        return 0.0
    c = clean.astype(np.float64)
    p = pred.astype(np.float64)
    gap_3d = np.broadcast_to(gap[:, :, np.newaxis], c.shape)
    c_vecs = c[gap_3d].reshape(-1, c.shape[2])
    p_vecs = p[gap_3d].reshape(-1, c.shape[2])
    dot = np.sum(c_vecs * p_vecs, axis=1)
    norm_c = np.linalg.norm(c_vecs, axis=1)
    norm_p = np.linalg.norm(p_vecs, axis=1)
    denom = norm_c * norm_p
    valid = denom > 1e-10
    if not np.any(valid):
        return 0.0
    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(cos_angle))))


def ergas(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> float:
    """ERGAS on gap pixels (h/l = 1 for gap-filling).

    Requires multichannel input (H, W, C) with C >= 2.
    Returns 0.0 for 2-D inputs.
    """
    gap = _gap_mask(mask)
    if not np.any(gap) or clean.ndim != 3 or clean.shape[2] < 2:
        return 0.0
    c = clean.astype(np.float64)
    p = pred.astype(np.float64)
    gap_3d = np.broadcast_to(gap[:, :, np.newaxis], c.shape)
    c_gap = c[gap_3d].reshape(-1, c.shape[2])
    p_gap = p[gap_3d].reshape(-1, c.shape[2])
    rmse_per_band = np.sqrt(np.mean((c_gap - p_gap) ** 2, axis=0))
    mean_per_band = np.mean(c_gap, axis=0)
    valid = np.abs(mean_per_band) > 1e-10
    if not np.any(valid):
        return 0.0
    ratio_sq = (rmse_per_band[valid] / mean_per_band[valid]) ** 2
    return float(100.0 * np.sqrt(np.mean(ratio_sq)))


def pixel_accuracy(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    threshold: float,
) -> float:
    """Fraction of gap pixels with |error| <= threshold."""
    gap = _gap_mask(mask)
    if not np.any(gap):
        return 1.0
    diff = _gap_abs_diff_flat(clean, pred, gap)
    return float(np.mean(diff <= threshold))


def rmse_per_band(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> list[float]:
    """Per-band RMSE on gap pixels. Returns [] for 2-D inputs."""
    gap = _gap_mask(mask)
    if not np.any(gap) or clean.ndim != 3:
        return []
    n_bands = clean.shape[2]
    gap_3d = np.broadcast_to(gap[:, :, np.newaxis], clean.shape)
    diff_flat = (clean.astype(np.float64) - pred.astype(np.float64))[
        gap_3d
    ].reshape(-1, n_bands)
    return [
        float(np.sqrt(np.mean(diff_flat[:, b] ** 2))) for b in range(n_bands)
    ]


# ---------------------------------------------------------------------------
# Aggregate: all per-patch metrics in one call
# ---------------------------------------------------------------------------


def compute_all(
    clean: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute the full metric set for a single patch.

    Returns a dict with keys matching pdi_pipeline.metrics.compute_all():
      psnr, ssim, rmse,
      sam, ergas (multichannel only),
      rmse_b0..b{C-1} (multichannel only),
      pixel_acc_002/005/01, f1_002/005/01.

    Args:
        clean: Reference image, (H, W) or (H, W, C), float in [0, 1].
        pred: Predicted image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        Dict[str, float] with all metric values.
    """
    gap = _gap_mask(mask)

    results: dict[str, float] = {
        "psnr": psnr(clean, pred, mask),
        "ssim": ssim(clean, pred, mask),
        "rmse": rmse(clean, pred, mask),
    }

    for suffix, threshold in _PIXEL_ACC_THRESHOLDS.items():
        acc = pixel_accuracy(clean, pred, mask, threshold)
        results[f"pixel_acc_{suffix}"] = acc
        results[f"f1_{suffix}"] = (
            float(2.0 * acc / (1.0 + acc)) if (1.0 + acc) > 0 else 0.0
        )

    if clean.ndim == 3 and clean.shape[2] >= 2:
        results["sam"] = sam(clean, pred, mask)
        results["ergas"] = ergas(clean, pred, mask)
        for b, rmse_b in enumerate(rmse_per_band(clean, pred, mask)):
            results[f"rmse_b{b}"] = rmse_b

    _ = gap  # gap used indirectly via public functions above
    return results


# ---------------------------------------------------------------------------
# Batch validation (used during training - torch tensors)
# ---------------------------------------------------------------------------


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


def compute_validation_metrics(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    max_ssim_samples: int = 0,
) -> dict[str, float]:
    """Bridge torch tensors to numpy and compute quality metrics.

    Converts (B, C, H, W) tensors to numpy (H, W, C) and calls
    compute_all() per sample.

    Args:
        preds: List of prediction tensors, each (B, C, H, W).
        targets: List of clean reference tensors, each (B, C, H, W).
        masks: List of mask tensors, each (B, H, W) with 1=gap.
        max_ssim_samples: Kept for backward compatibility but ignored;
            SSIM is now computed for every sample to be consistent with
            the final evaluation. Set to 0 (default) to always compute
            full SSIM.

    Returns:
        Dict with val_{metric} keys for all metrics in compute_all().
    """
    accum: dict[str, list[float]] = {}

    for pred_batch, target_batch, mask_batch in zip(
        preds, targets, masks, strict=True
    ):
        pred_np, target_np, mask_np = _to_numpy_batches(
            pred_batch, target_batch, mask_batch
        )
        b = pred_np.shape[0]
        for i in range(b):
            p, t = _prepare_sample(pred_np[i], target_np[i])
            scores = compute_all(t, p, mask_np[i])
            for key, val in scores.items():
                accum.setdefault(key, []).append(val)

    result: dict[str, float] = {}
    for key, values in accum.items():
        finite = [v for v in values if np.isfinite(v)]
        result[f"val_{key}"] = (
            float(np.mean(finite)) if finite else float("nan")
        )

    return result
