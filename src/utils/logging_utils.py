"""Experiment logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from experiment_registry import (
    append_registry_record,
    compact_utc_timestamp,
    finish_run,
    short_run_id,
    start_run,
)
from src.config.paths import EXPERIMENT_LOGS_DIR


def create_experiment_logger(
    name: str,
    log_path: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a stream/file logger for reproducible experiment scripts."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if log_path is None:
        log_path = EXPERIMENT_LOGS_DIR / f"{name}.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger

__all__ = [
    "append_registry_record",
    "compact_utc_timestamp",
    "create_experiment_logger",
    "finish_run",
    "short_run_id",
    "start_run",
]
