"""SCRE-Credit hybrid classifier tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.scre_credit import (
    SCRECreditHybridClassifier,
    SCREMetricError,
    build_scre_pareto_pool,
    build_scre_weight_table,
    optimize_validation_ensemble_weights,
)


def _metric_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["m1", "m2", "m3"],
            "pr_auc": [0.70, 0.75, 0.65],
            "roc_auc": [0.78, 0.80, 0.72],
            "recall": [0.60, 0.55, 0.70],
            "calibration_error": [0.12, 0.10, 0.14],
            "expected_cost": [100.0, 90.0, 130.0],
            "stability_score": [0.90, 0.80, 0.70],
            "faithfulness_score": [0.60, 0.90, 0.75],
        }
    )


def test_scre_credit_weights_sum_to_one():
    weights = build_scre_weight_table(_metric_table())
    assert np.isclose(weights["ensemble_weight"].sum(), 1.0)
    assert weights["ensemble_weight"].between(0.0, 1.0).all()


def test_scre_credit_probability_is_in_unit_interval():
    probabilities = pd.DataFrame(
        {
            "m1": [0.10, 0.20, 0.80, 0.90],
            "m2": [0.15, 0.25, 0.70, 0.85],
            "m3": [0.05, 0.30, 0.75, 0.95],
        }
    )
    y_true = pd.Series([0, 0, 1, 1])
    classifier = SCRECreditHybridClassifier(final_calibration_method="isotonic")
    classifier.fit_from_probabilities(probabilities, y_true, _metric_table())
    final_probability = classifier.predict_proba_from_base(probabilities)[:, 1]

    assert np.isclose(classifier.weight_table_["ensemble_weight"].sum(), 1.0)
    assert np.all(final_probability >= 0.0)
    assert np.all(final_probability <= 1.0)


def test_scre_credit_missing_metric_has_clear_error():
    bad_metrics = _metric_table().drop(columns=["faithfulness_score"])
    with pytest.raises(SCREMetricError, match="Missing SCRE-Credit metric columns"):
        build_scre_weight_table(bad_metrics)


def test_scre_pareto_pool_excludes_dominated_models():
    table = _metric_table().copy()
    table.loc[table["model"] == "m3", "recall"] = 0.50
    pareto = build_scre_pareto_pool(table)
    assert "m3" not in set(pareto["model"])
    assert set(pareto["model"]).issubset(set(table["model"]))
    assert pareto["test_set_used_for_selection"].eq(False).all()


def test_scre_optimized_weights_sum_to_one():
    probabilities = pd.DataFrame(
        {
            "m1": [0.10, 0.20, 0.80, 0.90, 0.30, 0.70],
            "m2": [0.15, 0.25, 0.70, 0.85, 0.35, 0.65],
        }
    )
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    metric_table = _metric_table().loc[_metric_table()["model"].isin(["m1", "m2"])].reset_index(drop=True)
    weights, audit = optimize_validation_ensemble_weights(
        probabilities,
        y_true,
        metric_table,
        n_random_candidates=10,
        n_top_candidates_for_calibration=5,
    )
    assert np.isclose(weights["ensemble_weight"].sum(), 1.0)
    assert weights["ensemble_weight"].between(0.0, 1.0).all()
    assert not audit.empty
