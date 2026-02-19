"""Quality metrics for gap-filling evaluation.

Standalone implementations using numpy and skimage only.
No imports from pdi_pipeline.metrics.

All metrics are computed strictly on gap pixels (mask=1).
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


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

    c = clean.astype(np.float64)
    p = pred.astype(np.float64)

    if c.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], c.shape)
        diff = c[gap_3d] - p[gap_3d]
    else:
        diff = c[gap] - p[gap]

    mse = float(np.mean(diff**2))
    if mse < 1e-12:
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

    c = clean.astype(np.float64)
    p = pred.astype(np.float64)

    if c.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], c.shape)
        diff = c[gap_3d] - p[gap_3d]
    else:
        diff = c[gap] - p[gap]

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


def compute_validation_metrics(
    preds: list[torch.Tensor],
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    max_ssim_samples: int = 64,
) -> dict[str, float]:
    """Bridge torch tensors to numpy and compute quality metrics.

    Converts (B, C, H, W) tensors to numpy (H, W, C) and calls
    psnr/ssim/rmse per sample. Also computes pixel accuracy and F1
    at multiple thresholds.

    Args:
        preds: List of prediction tensors, each (B, C, H, W).
        targets: List of clean reference tensors, each (B, C, H, W).
        masks: List of mask tensors, each (B, H, W) with 1=gap.
        max_ssim_samples: Cap for SSIM computation (performance).

    Returns:
        Dict with val_psnr, val_ssim, val_rmse, and pixel accuracy /
        F1 at thresholds 0.02, 0.05, 0.10.
    """
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    rmse_vals: list[float] = []

    thresholds = [0.02, 0.05, 0.10]
    acc_counts = dict.fromkeys(thresholds, 0.0)
    acc_totals = dict.fromkeys(thresholds, 0.0)

    ssim_count = 0

    for pred_batch, target_batch, mask_batch in zip(
        preds, targets, masks, strict=False
    ):
        pred_np = pred_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
        target_np = target_batch.detach().cpu().permute(0, 2, 3, 1).numpy()
        mask_np = mask_batch.detach().cpu().numpy()
        b = pred_np.shape[0]
        for i in range(b):
            p = pred_np[i]
            t = target_np[i]
            m = mask_np[i]

            if p.shape[2] == 1:
                p = p[:, :, 0]
                t = t[:, :, 0]

            psnr_vals.append(psnr(t, p, m))
            rmse_vals.append(rmse(t, p, m))

            if ssim_count < max_ssim_samples:
                ssim_vals.append(ssim(t, p, m))
                ssim_count += 1

            gap = m > 0.5
            if np.any(gap):
                if p.ndim == 3:
                    gap_3d = np.broadcast_to(gap[:, :, np.newaxis], p.shape)
                    diff = np.abs(p[gap_3d] - t[gap_3d])
                else:
                    diff = np.abs(p[gap] - t[gap])
                n_gap = float(diff.size)
                for tau in thresholds:
                    correct = float(np.sum(diff < tau))
                    acc_counts[tau] += correct
                    acc_totals[tau] += n_gap

    result: dict[str, float] = {}

    if psnr_vals:
        finite_psnr = [v for v in psnr_vals if np.isfinite(v)]
        result["val_psnr"] = float(np.mean(finite_psnr)) if finite_psnr else 0.0
    else:
        result["val_psnr"] = 0.0

    result["val_ssim"] = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    result["val_rmse"] = float(np.mean(rmse_vals)) if rmse_vals else 0.0

    for tau in thresholds:
        tau_key = str(tau).replace(".", "")
        pa = acc_counts[tau] / acc_totals[tau] if acc_totals[tau] > 0 else 0.0
        f1 = 2 * pa / (1 + pa) if (1 + pa) > 0 else 0.0
        result[f"val_pixel_acc_{tau_key}"] = pa
        result[f"val_f1_{tau_key}"] = f1

    return result
