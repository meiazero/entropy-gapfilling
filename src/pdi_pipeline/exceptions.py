"""Custom exception hierarchy for the PDI pipeline.

All domain-specific exceptions inherit from :class:`PDIError`, which
itself inherits from :class:`Exception`. This enables callers to catch
broad (``except PDIError``) or narrow (``except ConvergenceError``)
depending on context.
"""

from __future__ import annotations


class PDIError(Exception):
    """Base exception for all PDI pipeline errors."""


class DimensionError(PDIError, ValueError):
    """Raised when array shapes or dimensionality are invalid.

    Inherits from both :class:`PDIError` and :class:`ValueError` so
    callers who catch generic ``ValueError`` still see these.
    """


class ConvergenceError(PDIError, RuntimeError):
    """Raised when an iterative method fails to converge.

    Inherits from both :class:`PDIError` and :class:`RuntimeError`.
    """


class ValidationError(PDIError, ValueError):
    """Raised when input validation fails.

    Covers dtype mismatches, out-of-range values, invalid masks, and
    other precondition violations.
    """


class ConfigError(PDIError, ValueError):
    """Raised when configuration loading or validation fails."""


class InterpolationError(PDIError, RuntimeError):
    """Raised when an interpolation method encounters an unrecoverable error.

    Used as a catch-all for method-specific failures that do not fit
    :class:`ConvergenceError` (e.g. singular matrices, insufficient data).
    """


class InsufficientDataError(InterpolationError):
    """Raised when there are not enough valid data points.

    Typical for scattered-data methods (kriging, RBF, spline) that
    require a minimum number of training points.
    """
