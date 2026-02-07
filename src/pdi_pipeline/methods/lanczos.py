r"""Lanczos spectral gap-filling via iterative band-limited projection.

The Lanczos kernel approximates the ideal low-pass (sinc) filter with a
compactly supported window.  For gap-filling at integer grid positions the
classical separable kernel evaluates to zero at non-zero integer lags
(a property of the Nyquist sampling theorem), so a direct weighted-average
formulation is degenerate.

Instead we implement the Papoulis--Gerchberg iterative algorithm with a
Lanczos-windowed frequency response, which recovers the minimum-bandwidth
signal consistent with the observed samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from pdi_pipeline.methods.base import BaseMethod


class LanczosInterpolator(BaseMethod):
    r"""Lanczos spectral gap-filling (Papoulis--Gerchberg projection).

    Mathematical Formulation
    ------------------------
    The 1-D Lanczos-$a$ kernel is

    $$L_a(t) = \operatorname{sinc}(t)\,\operatorname{sinc}\!\left(\frac{t}{a}\right),
      \quad |t| < a,$$

    whose Fourier transform is the convolution of two rectangular pulses --
    a trapezoidal frequency response $\hat{L}_a(\nu)$:

    $$\hat{L}_a(\nu) = \begin{cases}
        1 & |\nu| \le \dfrac{a-1}{2a} \\[6pt]
        \dfrac{a + 1 - 2a|\nu|}{2} & \dfrac{a-1}{2a} < |\nu| \le \dfrac{a+1}{2a} \\[6pt]
        0 & |\nu| > \dfrac{a+1}{2a}
    \end{cases}$$

    The 2-D filter is separable:
    $\hat{H}_a(\nu_x, \nu_y) = \hat{L}_a(\nu_x)\,\hat{L}_a(\nu_y)$.

    **Papoulis--Gerchberg iteration.**  Let $M$ denote the binary mask of
    known pixels ($M_{ij} = 1$ if observed), $f$ the observed image, and
    $H_a$ the Lanczos low-pass operator.  Starting from an initial estimate
    $u^{(0)}$ (nearest-neighbour fill), the iteration is

    $$u^{(k+1)} = M \odot f \;+\; (1 - M) \odot (H_a * u^{(k)})$$

    i.e.\ known pixels are restored exactly while gap pixels receive the
    band-limited projection of the current estimate.  Under mild conditions
    the sequence converges to the minimum-bandwidth signal that agrees with
    the observations on $M$.

    Properties
    ----------
    * **Complexity per iteration:** $O(HW \log(HW))$ via FFT.
    * **Smoothness:** The result is band-limited (smooth), controlled by $a$.
    * **Convergence:** Monotone decrease in gap-pixel RMS change; typically
      converges in 20--50 iterations for 64x64 patches.

    Citation
    --------
    Papoulis, A. (1975). "A new algorithm in spectral analysis and
    band-limited extrapolation." *IEEE Trans. Circuits and Systems*,
    22(9), 735--742.

    Gerchberg, R. W. (1974). "Super-resolution through error energy
    reduction." *Optica Acta*, 21(9), 709--720.
    """

    name = "lanczos"

    def __init__(
        self,
        a: int = 3,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
    ) -> None:
        """Initialize Lanczos spectral interpolator.

        Args:
            a: Lanczos window parameter (controls passband width).
               a=2 gives a narrower passband (smoother result).
               a=3 (default) balances detail preservation and smoothness.
            max_iterations: Maximum Papoulis-Gerchberg iterations.
            tolerance: RMS convergence threshold on gap pixels.
        """
        if a < 1:
            raise ValueError("Lanczos parameter 'a' must be >= 1")
        self.a = a
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def _build_frequency_response(self, height: int, width: int) -> np.ndarray:
        """Build separable 2-D Lanczos frequency response."""
        fy = np.fft.fftfreq(height)
        fx = np.fft.fftfreq(width)

        a = self.a
        low = (a - 1) / (2 * a)
        high = (a + 1) / (2 * a)

        def _lanczos_1d(freq: np.ndarray) -> np.ndarray:
            f_abs = np.abs(freq)
            response = np.zeros_like(f_abs)
            response[f_abs <= low] = 1.0
            transition = (f_abs > low) & (f_abs <= high)
            if np.any(transition):
                response[transition] = (high - f_abs[transition]) / (high - low)
            return response

        return np.outer(_lanczos_1d(fy), _lanczos_1d(fx))

    def _fill_channel(
        self,
        channel: np.ndarray,
        mask_2d: np.ndarray,
        freq_response: np.ndarray,
        nn_indices: np.ndarray,
    ) -> np.ndarray:
        """Apply Papoulis-Gerchberg iteration to one channel."""
        valid_mask = ~mask_2d

        result = channel.copy().astype(np.float64)

        # Initialize gaps with nearest-neighbour values
        gap_y, gap_x = np.where(mask_2d)
        nn_y = nn_indices[0, gap_y, gap_x]
        nn_x = nn_indices[1, gap_y, gap_x]
        result[gap_y, gap_x] = channel[nn_y, nn_x]

        for _ in range(self.max_iterations):
            # Band-limit via FFT
            spectrum = np.fft.fft2(result)
            spectrum *= freq_response
            filtered = np.real(np.fft.ifft2(spectrum))

            # Restore known pixels, keep filtered values for gaps
            old_gaps = result[mask_2d].copy()
            result[mask_2d] = filtered[mask_2d]
            result[valid_mask] = channel[valid_mask]

            # Check convergence (RMS change on gap pixels)
            new_gaps = result[mask_2d]
            if old_gaps.size == 0:
                break
            rms_change = float(np.sqrt(np.mean((new_gaps - old_gaps) ** 2)))
            if rms_change < self.tolerance:
                break

        return result.astype(np.float32)

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        mask_2d = self._normalize_mask(mask)
        height, width = degraded.shape[:2]

        valid_mask = ~mask_2d
        if not np.any(valid_mask) or not np.any(mask_2d):
            return self._finalize(degraded.copy())

        freq_response = self._build_frequency_response(height, width)

        _, nn_indices = distance_transform_edt(
            ~valid_mask,
            return_distances=True,
            return_indices=True,
        )

        def _channel_fn(ch: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            return self._fill_channel(ch, mask_2d, freq_response, nn_indices)

        result = self._apply_channelwise(degraded, mask_2d, _channel_fn)
        return self._finalize(result)
