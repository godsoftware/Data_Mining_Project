"""Deletion/insertion faithfulness curves for tabular credit-risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def reference_row(X: pd.DataFrame) -> pd.Series:
    """Build a simple replacement row from train/test data medians and modes."""

    values = {}
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            values[column] = series.median()
        else:
            mode = series.mode(dropna=True)
            values[column] = mode.iloc[0] if not mode.empty else np.nan
    return pd.Series(values)


def deletion_insertion_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    max_features: int = 10,
) -> pd.DataFrame:
    """Progressively delete/insert ranked raw features and measure degradation."""

    selected = [feature for feature in ranked_features if feature in X.columns][:max_features]
    if not selected:
        return pd.DataFrame()

    baseline_proba = model.predict_proba(X)[:, 1]
    baseline_auc = roc_auc_score(y, baseline_proba)
    baseline_pr_auc = average_precision_score(y, baseline_proba)
    ref = reference_row(X)
    fully_masked = X.copy()
    for column in selected:
        fully_masked[column] = ref[column]

    rows = [
        {
            "curve": "baseline",
            "step": 0,
            "feature": "__NONE__",
            "roc_auc": baseline_auc,
            "pr_auc": baseline_pr_auc,
            "mean_probability": float(np.mean(baseline_proba)),
            "mean_abs_probability_shift": 0.0,
        }
    ]

    deleted = X.copy()
    for step, feature in enumerate(selected, start=1):
        deleted[feature] = ref[feature]
        proba = model.predict_proba(deleted)[:, 1]
        rows.append(
            {
                "curve": "deletion",
                "step": step,
                "feature": feature,
                "roc_auc": roc_auc_score(y, proba),
                "pr_auc": average_precision_score(y, proba),
                "mean_probability": float(np.mean(proba)),
                "mean_abs_probability_shift": float(np.abs(baseline_proba - proba).mean()),
            }
        )

    inserted = fully_masked.copy()
    for step, feature in enumerate(selected, start=1):
        inserted[feature] = X[feature]
        proba = model.predict_proba(inserted)[:, 1]
        rows.append(
            {
                "curve": "insertion",
                "step": step,
                "feature": feature,
                "roc_auc": roc_auc_score(y, proba),
                "pr_auc": average_precision_score(y, proba),
                "mean_probability": float(np.mean(proba)),
                "mean_abs_probability_shift": float(np.abs(baseline_proba - proba).mean()),
            }
        )

    return pd.DataFrame(rows)

