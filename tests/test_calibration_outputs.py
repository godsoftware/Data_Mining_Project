"""Calibration analysis output tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import FIGURES_DIR, TABLES_DIR


def _assert_calibration_table(path: Path, dataset: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "dataset",
        "model",
        "source_phase",
        "model_family",
        "calibration_method",
        "calibration_fit_policy",
        "split",
        "test_set_used",
        "brier_score",
        "ece",
        "calibration_slope",
        "calibration_intercept",
        "roc_auc",
        "pr_auc",
        "expected_cost",
    }
    assert required.issubset(table.columns)
    assert set(table["dataset"]) == {dataset}
    assert set(table["calibration_method"]) == {"uncalibrated", "sigmoid", "isotonic"}
    assert (table.groupby("model")["calibration_method"].nunique() == 3).all()
    assert not table["test_set_used"].astype(bool).any()
    assert table["brier_score"].between(0, 1).all()
    assert table["ece"].between(0, 1).all()
    assert table["pr_auc"].between(0, 1).all()
    return table


def test_calibration_results_tables_exist_and_are_complete():
    taiwan = _assert_calibration_table(TABLES_DIR / "calibration_results_taiwan.csv", "taiwan")
    heloc = _assert_calibration_table(TABLES_DIR / "calibration_results_heloc.csv", "heloc")
    assert taiwan["model"].nunique() >= 13
    assert heloc["model"].nunique() >= 13


def test_calibration_figures_exist():
    for filename in ["calibration_curve_taiwan.png", "calibration_curve_heloc.png"]:
        figure = FIGURES_DIR / filename
        assert figure.exists()
        assert figure.stat().st_size > 0
