"""Image quality metrics for gap-filling evaluation.

All metrics are computed strictly on gap pixels (mask=1) to avoid
diluting the signal with untouched pixels. Local variants produce
per-pixel maps via sliding windows for spatial analysis (LISA).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity

from pdi_pipeline.exceptions import DimensionError

logger = logging.getLogger(__name__)

_PSNR_CAP = 100.0
_MSE_FLOOR = 1e-12


def _as_gap_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a float mask to a boolean gap mask (True = gap pixel)."""
    return mask > 0.5


def _validate_inputs(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize inputs for metric computation.

    Args:
        clean: Reference image, (H, W) or (H, W, C), float32 in [0, 1].
        reconstructed: Reconstructed image, same shape as clean.
        mask: Binary mask, (H, W), where 1=gap pixels to evaluate.

    Returns:
        Tuple of (clean, reconstructed, mask_2d) with validated shapes.
    """
    clean = np.asarray(clean, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if clean.shape != reconstructed.shape:
        msg = (
            f"Shape mismatch: clean {clean.shape} "
            f"vs reconstructed {reconstructed.shape}"
        )
        raise DimensionError(msg)

    mask_2d = mask
    if mask_2d.ndim == 3:
        mask_2d = mask_2d[:, :, 0]

    if mask_2d.ndim != 2:
        msg = f"Mask must be 2D or 3D, got ndim={mask.ndim}"
        raise DimensionError(msg)

    h, w = mask_2d.shape[:2]
    if clean.ndim == 2 and clean.shape != (h, w):
        msg = (
            f"Spatial dims mismatch: clean {clean.shape} "
            f"vs mask {mask_2d.shape}"
        )
        raise DimensionError(msg)
    if clean.ndim == 3 and clean.shape[:2] != (h, w):
        msg = (
            f"Spatial dims mismatch: clean {clean.shape[:2]} "
            f"vs mask {mask_2d.shape}"
        )
        raise DimensionError(msg)

    return clean, reconstructed, _as_gap_mask(mask_2d)


# ---------------------------------------------------------------------------
# Private core implementations (accept already-validated arrays)
# ---------------------------------------------------------------------------


def _psnr_core(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    gap: np.ndarray,
) -> float:
    if not np.any(gap):
        return float("inf")
    if clean.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], clean.shape)
        diff = clean[gap_3d] - reconstructed[gap_3d]
    else:
        diff = clean[gap] - reconstructed[gap]
    mse = float(np.mean(diff**2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_core(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    gap: np.ndarray,
) -> float:
    if not np.any(gap):
        return 1.0
    is_multichannel = clean.ndim == 3
    _, ssim_map = structural_similarity(
        clean,
        reconstructed,
        data_range=1.0,
        full=True,
        channel_axis=2 if is_multichannel else None,
    )
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)
    return float(np.mean(ssim_map[gap]))


def _rmse_core(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    gap: np.ndarray,
) -> float:
    if not np.any(gap):
        return 0.0
    if clean.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], clean.shape)
        diff = clean[gap_3d] - reconstructed[gap_3d]
    else:
        diff = clean[gap] - reconstructed[gap]
    return float(np.sqrt(np.mean(diff**2)))


def _sam_core(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    gap: np.ndarray,
) -> float:
    if not np.any(gap):
        return 0.0
    clean_vecs = clean[gap]
    recon_vecs = reconstructed[gap]
    dot = np.sum(clean_vecs * recon_vecs, axis=1)
    norm_clean = np.linalg.norm(clean_vecs, axis=1)
    norm_recon = np.linalg.norm(recon_vecs, axis=1)
    denom = norm_clean * norm_recon
    valid = denom > 1e-10
    if not np.any(valid):
        return 0.0
    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(cos_angle))))


def _ergas_core(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    gap: np.ndarray,
) -> float:
    if not np.any(gap):
        return 0.0
    clean_gap = clean[gap]
    recon_gap = reconstructed[gap]
    rmse_per_band = np.sqrt(np.mean((clean_gap - recon_gap) ** 2, axis=0))
    mean_per_band = np.mean(clean_gap, axis=0)
    valid = np.abs(mean_per_band) > 1e-10
    if not np.any(valid):
        return 0.0
    ratio_sq = (rmse_per_band[valid] / mean_per_band[valid]) ** 2
    return float(100.0 * np.sqrt(np.mean(ratio_sq)))


def psnr(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Peak Signal-to-Noise Ratio on gap pixels only.

    PSNR = 10 * log10(1.0 / MSE) where MSE is computed over gap pixels.
    Data range assumed to be [0, 1].

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        PSNR in dB. Returns inf if MSE is effectively zero.
    """
    clean, reconstructed, gap = _validate_inputs(clean, reconstructed, mask)
    return _psnr_core(clean, reconstructed, gap)


def ssim(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Structural Similarity Index on gap pixels only.

    Computes the full SSIM map via skimage, then averages only the
    values at gap pixel locations.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        Mean SSIM value restricted to gap region.
    """
    clean, reconstructed, gap = _validate_inputs(clean, reconstructed, mask)
    return _ssim_core(clean, reconstructed, gap)


def rmse(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Root Mean Squared Error on gap pixels only.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        RMSE value over gap pixels.
    """
    clean, reconstructed, gap = _validate_inputs(clean, reconstructed, mask)
    return _rmse_core(clean, reconstructed, gap)


def sam(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Spectral Angle Mapper on gap pixels only.

    Computes the angle between spectral vectors at each gap pixel,
    then returns the mean angle in degrees. Requires (H, W, C) input
    with C >= 2.

    Args:
        clean: Reference image, (H, W, C).
        reconstructed: Reconstructed image, (H, W, C).
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        Mean spectral angle in degrees over gap pixels.
    """
    clean, reconstructed, gap = _validate_inputs(clean, reconstructed, mask)

    if clean.ndim != 3 or clean.shape[2] < 2:
        msg = (
            "SAM requires multichannel input (H, W, C) with C >= 2, "
            f"got shape {clean.shape}"
        )
        raise DimensionError(msg)

    return _sam_core(clean, reconstructed, gap)


def ergas(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese on gap pixels.

    Standard remote sensing metric for spectral fidelity assessment.
    For gap-filling (no resolution change), the spatial ratio h/l = 1:

        ERGAS = 100 * sqrt(1/B * sum_b( (RMSE_b / mean_b)^2 ))

    Requires multichannel input (H, W, C) with C >= 2.

    Args:
        clean: Reference image, (H, W, C), float32 in [0, 1].
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        ERGAS value (dimensionless). Lower is better.
    """
    clean, reconstructed, gap = _validate_inputs(clean, reconstructed, mask)

    if clean.ndim != 3 or clean.shape[2] < 2:
        msg = (
            "ERGAS requires multichannel input (H, W, C) with C >= 2, "
            f"got shape {clean.shape}"
        )
        raise DimensionError(msg)

    return _ergas_core(clean, reconstructed, gap)


def local_psnr(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    window: int = 15,
) -> np.ndarray:
    """Per-pixel PSNR map via sliding window.

    Computes PSNR within a local window centered on each pixel, using
    only gap pixels within that window. Used for LISA spatial analysis.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.
        window: Side length of the square window (must be odd).

    Returns:
        Float32 array of shape (H, W) with local PSNR values.
        Pixels with no gap neighbors in their window get NaN.
    """
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)
    gap = mask_2d

    if clean.ndim == 3:
        # Average squared error across channels
        sq_err = np.mean((clean - reconstructed) ** 2, axis=2)
    else:
        sq_err = (clean - reconstructed) ** 2

    h, w = sq_err.shape
    result = np.full((h, w), np.nan, dtype=np.float32)

    # uniform_filter with mode="constant", cval=0 implicitly zero-pads at
    # borders, which is equivalent to treating non-gap pixels as 0 contribution.
    # Manual np.pad is therefore redundant and removed.
    gap_f = gap.astype(np.float64)
    masked_err = sq_err.astype(np.float64) * gap_f

    # Vectorized sliding-window sums via scipy uniform_filter -- O(HW)
    area = float(window * window)
    gap_count = (
        uniform_filter(gap_f, size=window, mode="constant", cval=0.0) * area
    )
    err_sum = (
        uniform_filter(masked_err, size=window, mode="constant", cval=0.0)
        * area
    )

    # Compute MSE and PSNR only where there are gap pixels in the window
    has_gap = gap_count > 0.5  # at least 1 gap pixel
    mse = np.zeros_like(err_sum, dtype=np.float64)
    np.divide(err_sum, gap_count, out=mse, where=has_gap)

    valid_mse = mse > _MSE_FLOOR
    result[has_gap & valid_mse] = (
        10.0 * np.log10(1.0 / mse[has_gap & valid_mse])
    ).astype(np.float32)
    result[has_gap & ~valid_mse] = _PSNR_CAP

    return result


def local_ssim(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
    window: int = 15,
) -> np.ndarray:
    """Per-pixel SSIM map restricted to gap region.

    Computes the full SSIM map and masks it to gap pixels only.
    The SSIM computation itself already uses local windows internally.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.
        window: Passed as win_size to skimage SSIM (must be odd).

    Returns:
        Float32 array of shape (H, W) with SSIM values at gap pixels
        and NaN elsewhere.
    """
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)
    gap = mask_2d
    is_multichannel = clean.ndim == 3

    win_size = min(window, min(clean.shape[0], clean.shape[1]))
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    _, ssim_map = structural_similarity(
        clean,
        reconstructed,
        data_range=1.0,
        full=True,
        win_size=win_size,
        channel_axis=2 if is_multichannel else None,
    )

    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)

    result = np.full_like(ssim_map, np.nan, dtype=np.float32)
    result[gap] = ssim_map[gap]
    return result


def compute_all(
    clean: np.ndarray,
    reconstructed: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute all four metrics at once.

    Args:
        clean: Reference image, (H, W) or (H, W, C).
        reconstructed: Reconstructed image, same shape.
        mask: Binary mask, (H, W), 1=gap.

    Returns:
        Dictionary with keys 'psnr', 'ssim', 'rmse', and optionally
        'sam' (only if input is multichannel with C >= 2).
    """
    # Validate once; pass pre-validated arrays to core functions.
    clean, reconstructed, gap = _validate_inputs(clean, reconstructed, mask)

    results: dict[str, float] = {
        "psnr": _psnr_core(clean, reconstructed, gap),
        "ssim": _ssim_core(clean, reconstructed, gap),
        "rmse": _rmse_core(clean, reconstructed, gap),
    }

    if clean.ndim == 3 and clean.shape[2] >= 2:
        results["sam"] = _sam_core(clean, reconstructed, gap)
        results["ergas"] = _ergas_core(clean, reconstructed, gap)

    return results
