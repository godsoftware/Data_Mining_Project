"""Shared data validation checks."""

from __future__ import annotations

import pandas as pd


def basic_table_checks(df: pd.DataFrame, target: str) -> dict:
    """Return shape, missing, duplicate, and target-count checks."""

    if target not in df.columns:
        raise KeyError(f"Target column is missing: {target}")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {
            str(label): int(count)
            for label, count in df[target].value_counts().sort_index().items()
        },
    }


def assert_no_target_leakage_columns(
    df: pd.DataFrame,
    target: str,
    forbidden_terms=None,
    id_columns: list[str] | None = None,
) -> None:
    """Raise if obvious target-derived or identifier columns are present."""

    forbidden_terms = forbidden_terms or ["target", "default_next_month", "bad_flag"]
    id_columns = id_columns or ["ID"]
    feature_columns = [column for column in df.columns if column != target]
    bad_columns = [
        column
        for column in feature_columns
        if column in id_columns or any(term.lower() in column.lower() for term in forbidden_terms)
    ]
    if bad_columns:
        raise AssertionError(f"Potential leakage columns found: {bad_columns}")
