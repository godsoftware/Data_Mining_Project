"""Save helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from matplotlib.figure import Figure
import pandas as pd


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame as CSV and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON file and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_figure(figure: Figure, path: str | Path, dpi: int = 300) -> Path:
    """Save a Matplotlib figure and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_model(model: Any, path: str | Path) -> Path:
    """Save a fitted model with joblib and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
