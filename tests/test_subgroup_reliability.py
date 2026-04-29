"""Subgroup reliability utility tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.evaluation.subgroup_reliability import (
    add_taiwan_age_group,
    expected_cost_from_counts,
    subgroup_reliability_table,
)


def test_add_taiwan_age_group_uses_required_bins() -> None:
    X = pd.DataFrame({"AGE": [21, 29, 30, 39, 40, 49, 50, 75]})

    grouped = add_taiwan_age_group(X)

    assert grouped["AGE_GROUP"].tolist() == ["<30", "<30", "30-39", "30-39", "40-49", "40-49", "50+", "50+"]


def test_expected_cost_from_counts_uses_fn_and_fp_costs() -> None:
    assert expected_cost_from_counts(false_negative=3, false_positive=4, fn_cost=5.0, fp_cost=1.0) == 19.0


def test_subgroup_reliability_table_contains_requested_metrics() -> None:
    X = pd.DataFrame({"SEX": [1, 1, 1, 2, 2, 2], "AGE_GROUP": ["<30", "<30", "30-39", "30-39", "50+", "50+"]})
    y = pd.Series([0, 1, 1, 0, 0, 1])
    proba = np.array([0.10, 0.80, 0.65, 0.30, 0.20, 0.70])

    table = subgroup_reliability_table(
        X,
        y,
        proba,
        subgroup_columns=["SEX", "AGE_GROUP"],
        threshold=0.5,
        min_group_size=1,
    )

    required = {
        "sample_size",
        "default_rate",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "f1",
        "fpr",
        "fnr",
        "brier_score",
        "ece",
        "expected_cost",
    }
    assert required.issubset(table.columns)
    assert set(table["subgroup_column"]) == {"SEX", "AGE_GROUP"}
    assert table["analysis_note"].str.contains("not legal fairness proof").all()
