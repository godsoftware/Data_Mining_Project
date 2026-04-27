"""Baseline model output tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import BASELINE_MODELS_DIR, TABLES_DIR


EXPECTED_BASELINES = {
    "majority_baseline",
    "logistic_regression",
    "class_weighted_logistic_regression",
    "woe_scorecard_logistic_regression",
    "decision_tree",
    "random_forest",
}


def test_baseline_result_tables_have_all_models_and_final_test_rows():
    for dataset in ["taiwan", "heloc"]:
        table = pd.read_csv(TABLES_DIR / f"baseline_results_{dataset}.csv")
        assert set(table["model"]) == EXPECTED_BASELINES
        assert set(table["split"]) == {"validation", "test"}
        assert not table.loc[table["split"] == "validation", "test_set_used"].astype(bool).any()
        assert table.loc[table["split"] == "test", "test_set_used"].astype(bool).all()
        assert table["roc_auc"].between(0, 1).all()
        assert table["pr_auc"].between(0, 1).all()


def test_baseline_model_artifacts_exist():
    for dataset in ["taiwan", "heloc"]:
        dataset_dir = BASELINE_MODELS_DIR / dataset
        assert dataset_dir.exists()
        for model in EXPECTED_BASELINES:
            assert (dataset_dir / f"{model}.joblib").exists()
