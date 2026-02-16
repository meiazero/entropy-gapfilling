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
            raise PDIError("base error")


class TestDimensionError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(DimensionError, PDIError)

    def test_is_value_error(self) -> None:
        assert issubclass(DimensionError, ValueError)

    def test_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise DimensionError("bad shape")

    def test_caught_by_pdi_error(self) -> None:
        with pytest.raises(PDIError):
            raise DimensionError("bad shape")


class TestConvergenceError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(ConvergenceError, PDIError)

    def test_is_runtime_error(self) -> None:
        assert issubclass(ConvergenceError, RuntimeError)

    def test_caught_by_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise ConvergenceError("did not converge")


class TestValidationError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(ValidationError, PDIError)

    def test_is_value_error(self) -> None:
        assert issubclass(ValidationError, ValueError)

    def test_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise ValidationError("invalid input")


class TestConfigError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(ConfigError, PDIError)

    def test_is_value_error(self) -> None:
        assert issubclass(ConfigError, ValueError)

    def test_caught_by_value_error(self) -> None:
        with pytest.raises(ValueError):
            raise ConfigError("bad config")


class TestInterpolationError:
    def test_is_pdi_error(self) -> None:
        assert issubclass(InterpolationError, PDIError)

    def test_is_runtime_error(self) -> None:
        assert issubclass(InterpolationError, RuntimeError)

    def test_caught_by_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            raise InterpolationError("interp failed")


class TestInsufficientDataError:
    def test_is_interpolation_error(self) -> None:
        assert issubclass(InsufficientDataError, InterpolationError)

    def test_is_pdi_error(self) -> None:
        assert issubclass(InsufficientDataError, PDIError)

    def test_is_runtime_error(self) -> None:
        assert issubclass(InsufficientDataError, RuntimeError)

    def test_caught_by_interpolation_error(self) -> None:
        with pytest.raises(InterpolationError):
            raise InsufficientDataError("not enough points")


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
