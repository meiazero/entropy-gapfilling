"""Shared logging configuration for the PDI pipeline.

Provides :func:`setup_logging` for consistent structured logging
across all scripts and library modules, and :func:`setup_file_logging`
for persisting logs to disk alongside experiment results.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    *,
    fmt: str = ("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
) -> None:
    """Configure the root logger with a consistent format.

    Safe to call multiple times; clears existing handlers first.

    Args:
        level: Logging level (e.g. ``logging.INFO``).
        fmt: Log message format string.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates.
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)


def setup_file_logging(
    log_dir: str | Path,
    name: str = "experiment",
    level: int = logging.DEBUG,
) -> Path:
    """Add a file handler to the root logger.

    Args:
        log_dir: Directory where the log file is created.
        name: Base name for the log file (without extension).
        level: Logging level for the file handler.

    Returns:
        Path to the created log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    )
    logging.getLogger().addHandler(file_handler)

    return log_path


def get_project_root() -> Path:
    """Return the project root directory.

    Walks upward from this file until finding ``pyproject.toml``.

    Returns:
        Absolute path to the project root.

    Raises:
        FileNotFoundError: If ``pyproject.toml`` is not found.
    """
    current = Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "Could not find project root (no pyproject.toml found)"
    raise FileNotFoundError(msg)
