"""Cost-sensitive threshold and manual-review output tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import FIGURES_DIR, TABLES_DIR


def test_threshold_analysis_taiwan_has_required_grid_and_scenarios():
    table = pd.read_csv(TABLES_DIR / "threshold_analysis_taiwan.csv")
    required = {
        "dataset",
        "selection_split",
        "test_set_used",
        "model",
        "calibration_method",
        "scenario",
        "fn_cost",
        "fp_cost",
        "threshold",
        "expected_cost",
        "cost_improvement_pct_vs_0_50",
        "precision",
        "recall",
        "f1",
        "is_best_threshold_for_model_scenario",
    }
    assert required.issubset(table.columns)
    assert set(table["dataset"]) == {"taiwan"}
    assert set(table["scenario"]) == {"A_FN2_FP1", "B_FN5_FP1", "C_FN10_FP1"}
    assert table["threshold"].min() == 0.01
    assert table["threshold"].max() == 0.99
    assert table["threshold"].nunique() == 99
    assert not table["test_set_used"].astype(bool).any()
    assert table["model"].nunique() >= 13


def test_manual_review_band_results_taiwan_are_valid():
    table = pd.read_csv(TABLES_DIR / "manual_review_band_results_taiwan.csv")
    required = {
        "dataset",
        "selection_split",
        "test_set_used",
        "model",
        "calibration_method",
        "scenario",
        "t_low",
        "t_high",
        "manual_review_cost",
        "expected_policy_cost",
        "manual_review_rate",
        "review_capture_recall",
        "is_best_band_for_scenario",
    }
    assert required.issubset(table.columns)
    assert set(table["scenario"]) == {"A_FN2_FP1", "B_FN5_FP1", "C_FN10_FP1"}
    assert (table["t_low"] < table["t_high"]).all()
    assert table["t_low"].between(0.01, 0.99).all()
    assert table["t_high"].between(0.01, 0.99).all()
    assert not table["test_set_used"].astype(bool).any()
    assert table.groupby("scenario")["is_best_band_for_scenario"].sum().eq(1).all()


def test_cost_threshold_figures_exist():
    for filename in ["cost_curve_taiwan.png", "threshold_tradeoff_taiwan.png"]:
        figure = FIGURES_DIR / filename
        assert figure.exists()
        assert figure.stat().st_size > 0
