"""Image quality metrics for gap-filling evaluation.

All metrics are computed strictly on gap pixels (mask=1) to avoid
diluting the signal with untouched pixels. Local variants produce
per-pixel maps via sliding windows for spatial analysis (LISA).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


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
        raise ValueError(msg)

    mask_2d = mask
    if mask_2d.ndim == 3:
        mask_2d = mask_2d[:, :, 0]

    if mask_2d.ndim != 2:
        msg = f"Mask must be 2D or 3D, got ndim={mask.ndim}"
        raise ValueError(msg)

    h, w = mask_2d.shape[:2]
    if clean.ndim == 2 and clean.shape != (h, w):
        msg = (
            f"Spatial dims mismatch: clean {clean.shape} "
            f"vs mask {mask_2d.shape}"
        )
        raise ValueError(msg)
    if clean.ndim == 3 and clean.shape[:2] != (h, w):
        msg = (
            f"Spatial dims mismatch: clean {clean.shape[:2]} "
            f"vs mask {mask_2d.shape}"
        )
        raise ValueError(msg)

    return clean, reconstructed, _as_gap_mask(mask_2d)


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
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)
    gap = mask_2d

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
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)
    gap = mask_2d

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

    # ssim_map is (H, W) regardless of channel_axis
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)

    return float(np.mean(ssim_map[gap]))


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
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)
    gap = mask_2d

    if not np.any(gap):
        return 0.0

    if clean.ndim == 3:
        gap_3d = np.broadcast_to(gap[:, :, np.newaxis], clean.shape)
        diff = clean[gap_3d] - reconstructed[gap_3d]
    else:
        diff = clean[gap] - reconstructed[gap]

    return float(np.sqrt(np.mean(diff**2)))


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
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)

    if clean.ndim != 3 or clean.shape[2] < 2:
        msg = (
            "SAM requires multichannel input (H, W, C) with C >= 2, "
            f"got shape {clean.shape}"
        )
        raise ValueError(msg)

    gap = mask_2d
    if not np.any(gap):
        return 0.0

    # Extract spectral vectors at gap pixels: (N, C)
    clean_vecs = clean[gap]
    recon_vecs = reconstructed[gap]

    # Dot product and norms along spectral axis
    dot = np.sum(clean_vecs * recon_vecs, axis=1)
    norm_clean = np.linalg.norm(clean_vecs, axis=1)
    norm_recon = np.linalg.norm(recon_vecs, axis=1)

    denom = norm_clean * norm_recon
    # Avoid division by zero for zero vectors
    valid = denom > 1e-10
    if not np.any(valid):
        return 0.0

    cos_angle = np.clip(dot[valid] / denom[valid], -1.0, 1.0)
    angles_rad = np.arccos(cos_angle)
    angles_deg = np.degrees(angles_rad)

    return float(np.mean(angles_deg))


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
    clean, reconstructed, mask_2d = _validate_inputs(clean, reconstructed, mask)

    if clean.ndim != 3 or clean.shape[2] < 2:
        msg = (
            "ERGAS requires multichannel input (H, W, C) with C >= 2, "
            f"got shape {clean.shape}"
        )
        raise ValueError(msg)

    gap = mask_2d
    if not np.any(gap):
        return 0.0

    n_bands = clean.shape[2]
    band_sum = 0.0

    for b in range(n_bands):
        clean_b = clean[:, :, b]
        recon_b = reconstructed[:, :, b]

        diff = clean_b[gap] - recon_b[gap]
        rmse_b = float(np.sqrt(np.mean(diff**2)))
        mean_b = float(np.mean(clean_b[gap]))

        if abs(mean_b) < 1e-10:
            continue

        band_sum += (rmse_b / mean_b) ** 2

    return float(100.0 * np.sqrt(band_sum / n_bands))


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
    half = window // 2
    result = np.full((h, w), np.nan, dtype=np.float32)

    # Pad arrays for border handling
    sq_err_pad = np.pad(sq_err, half, mode="reflect")
    gap_pad = np.pad(gap, half, mode="constant", constant_values=False)

    for i in range(h):
        for j in range(w):
            local_gap = gap_pad[i : i + window, j : j + window]
            if not np.any(local_gap):
                continue
            local_err = sq_err_pad[i : i + window, j : j + window]
            mse = float(np.mean(local_err[local_gap]))
            if mse < 1e-12:
                result[i, j] = 100.0  # cap at 100 dB
            else:
                result[i, j] = 10.0 * np.log10(1.0 / mse)

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
    results = {
        "psnr": psnr(clean, reconstructed, mask),
        "ssim": ssim(clean, reconstructed, mask),
        "rmse": rmse(clean, reconstructed, mask),
    }

    clean_arr = np.asarray(clean)
    if clean_arr.ndim == 3 and clean_arr.shape[2] >= 2:
        results["sam"] = sam(clean, reconstructed, mask)
        results["ergas"] = ergas(clean, reconstructed, mask)

    return results
