"""SHAP background sensitivity placeholders."""

from __future__ import annotations

import pandas as pd


def background_sample_grid(X: pd.DataFrame, sample_sizes=(50, 100, 250), random_state: int = 42):
    """Yield deterministic background samples for sensitivity experiments."""

    for size in sample_sizes:
        yield size, X.sample(min(size, len(X)), random_state=random_state)

