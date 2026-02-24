"""Shared logging configuration for the PDI pipeline.

Provides :func:`setup_logging` for consistent structured logging
across all scripts and library modules, and :func:`setup_file_logging`
for persisting logs to disk alongside experiment results.
"""

from __future__ import annotations

import logging
import sys
import time
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


class StreamProgress:
    """A progress reporter that logs to standard logging instead of TTY.

    Mimics basic `tqdm` behavior to be used as a drop-in replacement
    in environments like SLURM where `\r` carriage returns cause massive logs.
    """

    def __init__(
        self,
        iterable=None,
        desc: str = "Progress",
        total: int | None = None,
        log_interval_s: float = 1.0,
        logger: logging.Logger | None = None,
    ):
        self.iterable = iterable
        self.desc = desc
        self.total = total
        self.log_interval_s = log_interval_s
        self.logger = logger or logging.getLogger(__name__)

        if self.total is None and hasattr(iterable, "__len__"):
            self.total = len(iterable)

        self.n = 0
        self.start_t = time.perf_counter()
        self.last_log_t = self.start_t

    def __iter__(self):
        if self.iterable is None:
            return self

        self.start_t = time.perf_counter()
        self.last_log_t = self.start_t
        for obj in self.iterable:
            yield obj
            self.update(1)

    def __enter__(self):
        self.start_t = time.perf_counter()
        self.last_log_t = self.start_t
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._log_progress(force=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        current_t = time.perf_counter()
        if current_t - self.last_log_t >= self.log_interval_s:
            self._log_progress()
            self.last_log_t = current_t

    def _log_progress(self, force: bool = False) -> None:
        elapsed = time.perf_counter() - self.start_t
        rate = self.n / elapsed if elapsed > 0 else 0

        msg = f"{self.desc}: {self.n}"
        if self.total is not None:
            pct = 100.0 * self.n / self.total if self.total > 0 else 0
            msg += f"/{self.total} [{pct:.1f}%]"

        msg += f" [{elapsed:.1f}s, {rate:.2f} it/s]"

        # Log at INFO level so it shows up without DEBUG enabled
        self.logger.info(msg)
