"""Lightweight experiment registry for reproducible research runs."""

from __future__ import annotations

import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.paths import EXPERIMENT_LOGS_DIR


REGISTRY_DIR = EXPERIMENT_LOGS_DIR
RUNS_JSONL = REGISTRY_DIR / "runs.jsonl"


def _json_default(value: Any) -> Any:
    """Serialize numpy/scalar/path values for JSON logging."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def short_run_id(run_id: str) -> str:
    """Return a short experiment id derived from a UUID run id."""

    return run_id.split("-", 1)[0]


def append_registry_record(record: dict[str, Any], path: Path = RUNS_JSONL) -> None:
    """Append a single experiment event to the JSONL registry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def start_run(
    name: str,
    dataset: str,
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Create a run-start record and return the run id."""

    run_id = str(uuid.uuid4())
    append_registry_record(
        {
            "event": "start",
            "run_id": run_id,
            "name": name,
            "dataset": dataset,
            "params": params or {},
            "tags": tags or [],
            "timestamp_utc": utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    )
    return run_id


def finish_run(
    run_id: str,
    status: str = "completed",
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, str | Path] | None = None,
    notes: str | None = None,
) -> None:
    """Create a run-finish record."""

    append_registry_record(
        {
            "event": "finish",
            "run_id": run_id,
            "status": status,
            "metrics": metrics or {},
            "artifacts": artifacts or {},
            "notes": notes,
            "timestamp_utc": utc_now(),
        }
    )
