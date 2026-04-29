"""Faithfulness analysis unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.explainability.faithfulness import (
    expected_cost,
    faithfulness_metrics,
    group_deletion_table,
    single_feature_perturbation_table,
)


class _ToyProbabilityModel:
    """Minimal probability model for perturbation tests."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk = pd.to_numeric(X["risk"], errors="coerce").fillna(0.0).to_numpy()
        helper = pd.to_numeric(X["helper"], errors="coerce").fillna(0.0).to_numpy()
        probability = np.clip(0.08 + 0.14 * risk + 0.04 * helper, 0.01, 0.99)
        return np.column_stack([1.0 - probability, probability])


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "risk": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "helper": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return X, y


def test_expected_cost_counts_false_negative_and_false_positive() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.6, 0.4, 0.2, 0.9])

    assert expected_cost(y_true, y_proba, threshold=0.5, fn_cost=5.0, fp_cost=1.0) == 6.0


def test_single_feature_perturbation_outputs_required_metrics() -> None:
    X, y = _toy_data()
    model = _ToyProbabilityModel()
    baseline_proba = model.predict_proba(X)[:, 1]
    baseline_metrics = faithfulness_metrics(y, baseline_proba, threshold=0.5)

    table = single_feature_perturbation_table(
        model,
        X,
        y,
        features=["risk", "helper"],
        replacements={"risk": 2.5, "helper": 0.5},
        threshold=0.5,
        baseline_proba=baseline_proba,
        baseline_metrics=baseline_metrics,
    )

    required = {"roc_auc_drop", "pr_auc_drop", "probability_shift", "recall_change", "cost_change"}
    assert set(table["feature"]) == {"risk", "helper"}
    assert required.issubset(table.columns)
    assert not table[list(required)].isna().any().any()


def test_group_deletion_ignores_missing_group_members() -> None:
    X, y = _toy_data()
    model = _ToyProbabilityModel()
    baseline_proba = model.predict_proba(X)[:, 1]
    baseline_metrics = faithfulness_metrics(y, baseline_proba, threshold=0.5)

    table = group_deletion_table(
        model,
        X,
        y,
        feature_groups={"risk_group": ["risk", "missing_feature"]},
        replacements={"risk": 2.5, "helper": 0.5},
        threshold=0.5,
        baseline_proba=baseline_proba,
        baseline_metrics=baseline_metrics,
    )

    assert table.loc[0, "group"] == "risk_group"
    assert table.loc[0, "n_features"] == 1
    assert table.loc[0, "feature_set"] == "risk"
