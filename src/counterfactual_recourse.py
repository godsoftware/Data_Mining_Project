"""Simple counterfactual and recourse utilities for credit-risk cases."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_recourse_candidates(
    model,
    X_reference: pd.DataFrame,
    row: pd.Series,
    actionable_directions: dict[str, int],
    target_threshold: float,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Try one-feature quantile moves and report probability reduction.

    `actionable_directions` uses -1 when lower values are expected to reduce
    risk and +1 when higher values are expected to reduce risk.
    """

    if quantiles is None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    baseline = pd.DataFrame([row])
    baseline_proba = float(model.predict_proba(baseline)[:, 1][0])
    rows = []

    for feature, direction in actionable_directions.items():
        if feature not in X_reference.columns or feature not in row.index:
            continue

        values = pd.to_numeric(X_reference[feature], errors="coerce").dropna()
        if values.empty:
            continue

        candidate_values = np.quantile(values, quantiles)
        if direction < 0:
            candidate_values = [value for value in candidate_values if value < row[feature]]
        else:
            candidate_values = [value for value in candidate_values if value > row[feature]]

        for candidate_value in candidate_values:
            candidate = row.copy()
            candidate[feature] = candidate_value
            proba = float(model.predict_proba(pd.DataFrame([candidate]))[:, 1][0])
            rows.append(
                {
                    "feature": feature,
                    "original_value": row[feature],
                    "candidate_value": candidate_value,
                    "baseline_probability": baseline_proba,
                    "candidate_probability": proba,
                    "probability_change": proba - baseline_proba,
                    "crosses_threshold": bool(proba < target_threshold),
                }
            )

    return pd.DataFrame(rows).sort_values(["crosses_threshold", "candidate_probability"], ascending=[False, True])

