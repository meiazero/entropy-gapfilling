r"""Bicubic interpolation for image gap-filling.

Uses the Clough--Tocher $C^1$ piecewise-cubic scheme on a Delaunay
triangulation of the known pixels.  Each Delaunay triangle is subdivided
into three micro-triangles and a cubic polynomial is fitted per
micro-triangle so that the global surface is continuously differentiable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pdi_pipeline.methods._griddata import griddata_fill
from pdi_pipeline.methods.base import BaseMethod


class BicubicInterpolator(BaseMethod):
    r"""Clough--Tocher $C^1$ cubic interpolation for gap-filling.

    Mathematical Formulation
    ------------------------
    Given known pixels $\{(x_i, y_i, f_i)\}_{i=1}^N$ the Delaunay
    triangulation $\mathcal{T}$ is constructed, then each triangle
    $T_k$ is split into three sub-triangles by connecting each vertex to
    the centroid.

    On each sub-triangle a cubic polynomial

    $$p(x,y) = \sum_{|\alpha| \le 3} c_\alpha\, x^{\alpha_1} y^{\alpha_2}$$

    is fitted (10 coefficients per sub-triangle, 30 per macro-triangle).
    The coefficients are determined by imposing:

    1. **Interpolation** at the three vertices:
       $p(x_i, y_i) = f_i$.
    2. **Gradient matching** at vertices: $\nabla p$ agrees with the
       gradient estimated from surrounding data.
    3. **$C^1$ continuity** across all edges (both macro-edges shared
       between triangles and micro-edges within a macro-triangle):

       $$p^{(k)}(x,y) = p^{(l)}(x,y), \quad
         \nabla p^{(k)}(x,y) = \nabla p^{(l)}(x,y)
         \quad \forall\, (x,y) \in e_{kl}.$$

    This yields a globally $C^1$ surface that reproduces cubic polynomials
    exactly.  For gap pixels outside the convex hull of the known sites
    a nearest-neighbour fallback is used.

    Properties
    ----------
    * **Complexity:** $O(N \log N)$ triangulation + $O(M \log N)$ queries.
    * **Smoothness:** $C^1$ -- both value and gradient are continuous.
    * **Locality:** Each gap pixel depends on $\sim 10$ known pixels
      (the vertices and gradient stencils of the enclosing triangle).

    Citation
    --------
    Clough, R. W. and Tocher, J. L. (1965). "Finite element stiffness
    matrices for analysis of plates in bending." *Proceedings of the
    Conference on Matrix Methods in Structural Mechanics*, 515--545.

    Alfeld, P. (1984). "A trivariate Clough--Tocher scheme for tetrahedral
    data." *Computer Aided Geometric Design*, 1(2), 169--181.

    See also: ``scipy.interpolate.griddata`` with *method='cubic'*.
    """

    name = "bicubic"

    def apply(
        self,
        degraded: np.ndarray,
        mask: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
    ) -> np.ndarray:
        mask_2d = self._normalize_mask(mask)
        result = griddata_fill(
            degraded, mask_2d, "cubic", self._apply_channelwise
        )
        if result is None:
            return self._finalize(degraded.copy())
        return self._finalize(result)
