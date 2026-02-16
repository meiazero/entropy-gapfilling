"""Centralized method registry and factory for interpolation methods.

Provides :func:`get_interpolator` to instantiate methods by name, and
:func:`list_methods` / :func:`list_categories` for introspection.
"""

from __future__ import annotations

import logging
from typing import Any

from pdi_pipeline.exceptions import ConfigError
from pdi_pipeline.methods.base import BaseMethod

logger = logging.getLogger(__name__)

# Lazy-loaded registry: populated on first access.
_REGISTRY: dict[str, type[BaseMethod]] | None = None

# Canonical name -> (module_path, class_name) for lazy import.
_METHOD_MAP: dict[str, tuple[str, str]] = {
    # --- spatial ---
    "nearest": (
        "pdi_pipeline.methods.nearest",
        "NearestInterpolator",
    ),
    "bilinear": (
        "pdi_pipeline.methods.bilinear",
        "BilinearInterpolator",
    ),
    "bicubic": (
        "pdi_pipeline.methods.bicubic",
        "BicubicInterpolator",
    ),
    "lanczos": (
        "pdi_pipeline.methods.lanczos",
        "LanczosInterpolator",
    ),
    # --- kernel ---
    "idw": (
        "pdi_pipeline.methods.idw",
        "IDWInterpolator",
    ),
    "rbf": (
        "pdi_pipeline.methods.rbf",
        "RBFInterpolator",
    ),
    "spline": (
        "pdi_pipeline.methods.spline",
        "SplineInterpolator",
    ),
    # --- geostatistical ---
    "kriging": (
        "pdi_pipeline.methods.kriging",
        "KrigingInterpolator",
    ),
    # --- transform ---
    "dct": (
        "pdi_pipeline.methods.transforms",
        "DCTInpainting",
    ),
    "wavelet": (
        "pdi_pipeline.methods.transforms",
        "WaveletInpainting",
    ),
    "tv": (
        "pdi_pipeline.methods.transforms",
        "TVInpainting",
    ),
    # --- compressive sensing ---
    "cs_dct": (
        "pdi_pipeline.methods.compressive",
        "L1DCTInpainting",
    ),
    "cs_wavelet": (
        "pdi_pipeline.methods.compressive",
        "L1WaveletInpainting",
    ),
    # --- patch-based ---
    "non_local": (
        "pdi_pipeline.methods.patch_based",
        "NonLocalMeansInterpolator",
    ),
    "exemplar": (
        "pdi_pipeline.methods.patch_based",
        "ExemplarBasedInterpolator",
    ),
}

# Convenience aliases so config files can use either name.
_ALIASES: dict[str, str] = {
    "l1_dct": "cs_dct",
    "l1_wavelet": "cs_wavelet",
    "non_local_means": "non_local",
    "exemplar_based": "exemplar",
    "inverse_distance": "idw",
    "total_variation": "tv",
}

# Category grouping for display / reporting.
_CATEGORIES: dict[str, list[str]] = {
    "spatial": ["nearest", "bilinear", "bicubic", "lanczos"],
    "kernel": ["idw", "rbf", "spline"],
    "geostatistical": ["kriging"],
    "transform": ["dct", "wavelet", "tv"],
    "compressive": ["cs_dct", "cs_wavelet"],
    "patch_based": ["non_local", "exemplar"],
}


def _resolve_name(name: str) -> str:
    """Resolve an alias to its canonical registry name.

    Args:
        name: Method name or alias.

    Returns:
        Canonical method name.

    Raises:
        ConfigError: If the name is unknown.
    """
    canonical = _ALIASES.get(name, name)
    if canonical not in _METHOD_MAP:
        valid = sorted({*_METHOD_MAP, *_ALIASES})
        msg = f"Unknown method {name!r}. Valid names: {valid}"
        raise ConfigError(msg)
    return canonical


def _import_class(name: str) -> type[BaseMethod]:
    """Lazily import and return the class for *name*."""
    import importlib

    canonical = _resolve_name(name)
    module_path, class_name = _METHOD_MAP[canonical]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls  # type: ignore[no-any-return]


def _build_registry() -> dict[str, type[BaseMethod]]:
    """Build the full registry by importing all method classes."""
    registry: dict[str, type[BaseMethod]] = {}
    for name in _METHOD_MAP:
        registry[name] = _import_class(name)
    return registry


def get_interpolator(name: str, **kwargs: Any) -> BaseMethod:
    """Instantiate an interpolation method by name.

    This is the main factory function.  It lazily imports method
    classes to avoid loading heavy dependencies (e.g. ``pykrige``,
    ``cvxpy``) until actually needed.

    Args:
        name: Method name or alias (e.g. ``"bicubic"``,
            ``"l1_dct"``).
        **kwargs: Keyword arguments forwarded to the method
            constructor.

    Returns:
        An initialized :class:`BaseMethod` instance.

    Raises:
        ConfigError: If the method name is unknown.
    """
    canonical = _resolve_name(name)
    cls = _import_class(canonical)
    logger.debug(
        "Instantiating method %s (%s) with params=%s",
        canonical,
        cls.__name__,
        kwargs,
    )
    return cls(**kwargs)


def list_methods() -> list[str]:
    """Return all canonical method names (sorted)."""
    return sorted(_METHOD_MAP)


def list_aliases() -> dict[str, str]:
    """Return all alias -> canonical name mappings."""
    return dict(_ALIASES)


def list_categories() -> dict[str, list[str]]:
    """Return method groupings by category."""
    return {k: list(v) for k, v in _CATEGORIES.items()}


def get_all_methods() -> dict[str, type[BaseMethod]]:
    """Return the full registry (imports all classes)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return dict(_REGISTRY)
