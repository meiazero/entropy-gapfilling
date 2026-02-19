"""Unit tests for the custom exception hierarchy."""

from __future__ import annotations

import pytest

from pdi_pipeline.exceptions import (
    ConfigError,
    ConvergenceError,
    DimensionError,
    InsufficientDataError,
    InterpolationError,
    PDIError,
    ValidationError,
)


class TestPDIError:
    def test_is_exception(self) -> None:
        assert issubclass(PDIError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(PDIError):
            exc = PDIError("base error")
            raise exc


class TestDimensionError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(DimensionError, PDIError)

    def test_is_value_error(self) -> None:
        assert issubclass(DimensionError, ValueError)

    def test_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            exc = DimensionError("bad shape")
            raise exc

    def test_caught_by_pdi_error(self) -> None:
        with pytest.raises(PDIError):
            exc = DimensionError("bad shape")
            raise exc


class TestConvergenceError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(ConvergenceError, PDIError)

    def test_is_runtime_error(self) -> None:
        assert issubclass(ConvergenceError, RuntimeError)

    def test_caught_by_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            exc = ConvergenceError("did not converge")
            raise exc


class TestValidationError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(ValidationError, PDIError)

    def test_is_value_error(self) -> None:
        assert issubclass(ValidationError, ValueError)

    def test_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            exc = ValidationError("invalid input")
            raise exc


class TestConfigError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(ConfigError, PDIError)

    def test_is_value_error(self) -> None:
        assert issubclass(ConfigError, ValueError)

    def test_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            exc = ConfigError("bad config")
            raise exc


class TestInterpolationError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(InterpolationError, PDIError)

    def test_is_runtime_error(self) -> None:
        assert issubclass(InterpolationError, RuntimeError)

    def test_caught_by_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            exc = InterpolationError("interp failed")
            raise exc


class TestInsufficientDataError:
    def test_is_interpolation_error(self) -> None:
        assert issubclass(InsufficientDataError, InterpolationError)

    def test_is_pdi_error(self) -> None:
        assert issubclass(InsufficientDataError, PDIError)

    def test_is_runtime_error(self) -> None:
        assert issubclass(InsufficientDataError, RuntimeError)

    def test_caught_by_interpolation_error(self) -> None:
        with pytest.raises(InterpolationError):
            exc = InsufficientDataError("not enough points")
            raise exc


class TestMessagePreservation:
    """All exceptions should preserve their message."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            PDIError,
            DimensionError,
            ConvergenceError,
            ValidationError,
            ConfigError,
            InterpolationError,
            InsufficientDataError,
        ],
    )
    def test_message_preserved(self, exc_cls: type[PDIError]) -> None:
        msg = "test message for " + exc_cls.__name__
        exc = exc_cls(msg)
        assert str(exc) == msg
