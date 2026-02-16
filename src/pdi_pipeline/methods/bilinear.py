r"""Bilinear interpolation for image gap-filling.

Uses piecewise linear interpolation on a Delaunay triangulation of known
pixels.  For each gap pixel the enclosing triangle is identified and the
value is recovered via barycentric (affine) weighting of its three
vertices -- the natural 2-D generalisation of linear interpolation to
irregularly spaced data.
"""

from __future__ import annotations

import logging

import numpy as np

from pdi_pipeline.methods._griddata import griddata_fill
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)


class BilinearInterpolator(BaseMethod):
    r"""Piecewise-linear interpolation on Delaunay triangulation.

    Mathematical Formulation
    ------------------------
    Given a set of known pixels $\{(x_i, y_i, f_i)\}_{i=1}^N$ on a regular
    grid, we construct the Delaunay triangulation $\mathcal{T}$ of the sites
    $\{(x_i, y_i)\}$.

    For a gap pixel at position $\mathbf{p} = (x_0, y_0)$ lying inside
    triangle $T_k = \{(x_a, y_a),\,(x_b, y_b),\,(x_c, y_c)\}$, the
    interpolated value is the unique affine (barycentric) combination:

    $$\hat f(\mathbf{p})
      = \lambda_a\, f_a + \lambda_b\, f_b + \lambda_c\, f_c$$

    where the barycentric coordinates $(\lambda_a, \lambda_b, \lambda_c)$
    satisfy

    $$\begin{pmatrix} x_a - x_c & x_b - x_c \\
                       y_a - y_c & y_b - y_c \end{pmatrix}
      \begin{pmatrix} \lambda_a \\ \lambda_b \end{pmatrix}
      = \begin{pmatrix} x_0 - x_c \\ y_0 - y_c \end{pmatrix},
      \qquad \lambda_c = 1 - \lambda_a - \lambda_b.$$

    The surface is $C^0$ (continuous) across triangle edges and reproduces
    linear functions exactly.  For gap pixels outside the convex hull of
    the known sites (e.g.\ at image borders), a nearest-neighbour fallback
    is applied.

    Properties
    ----------
    * **Complexity:** $O(N \log N)$ for the Delaunay triangulation plus
      $O(M \log N)$ point-location queries ($M$ = number of gap pixels).
    * **Smoothness:** $C^0$ -- value-continuous but gradient-discontinuous
      across triangle edges.
    * **Locality:** Each gap pixel depends on exactly three known pixels.

    Citation
    --------
    Amidror, I. (2002). "Scattered data interpolation methods for
    electronic imaging systems: a survey." *Journal of Electronic
    Imaging*, 11(2), 157--176.

    See also: ``scipy.interpolate.griddata`` with *method='linear'*.
    """

    name = "bilinear"

    def __init__(self) -> None:
        """Initialize the bilinear interpolator."""

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Apply bilinear (Delaunay linear) interpolation to fill gaps.

        Args:
            degraded: Array with missing data, shape ``(H, W)`` or
                ``(H, W, C)``, dtype ``float32``, values in ``[0, 1]``.
            mask: Binary mask where ``True``/``1`` marks gap pixels to fill.
                Shape ``(H, W)`` or broadcastable ``(H, W, C)``.
            meta: Optional metadata (CRS, transform, band names, etc.).

        Returns:
            Reconstructed ``float32`` array with same shape as *degraded*,
            values clipped to ``[0, 1]``, no ``NaN``/``Inf``.
        """
        degraded, mask_2d = self._validate_inputs(degraded, mask)
        early = self._early_exit_if_no_gaps(degraded, mask_2d)
        if early is not None:
            return early

        logger.debug("Running bilinear (Delaunay linear) gap-filling.")
        result = griddata_fill(
            degraded, mask_2d, "linear", self._apply_channelwise
        )
        if result is None:
            logger.debug(
                "griddata_fill returned None (no valid pixels); "
                "falling back to input copy."
            )
            return self._finalize(degraded.copy())
        return self._finalize(result)
