"""Phase 7 model-pool output tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import BOOSTING_MODELS_DIR, MONOTONIC_MODELS_DIR, TABLES_DIR


EXPECTED_PHASE7_MODELS = {
    "xgboost",
    "lightgbm",
    "catboost",
    "smotenc_xgboost",
    "class_weighted_lightgbm",
    "monotonic_xgboost",
    "monotonic_lightgbm",
}


def test_phase7_result_tables_have_all_models_and_final_test_rows():
    for dataset in ["taiwan", "heloc"]:
        table = pd.read_csv(TABLES_DIR / f"model_results_{dataset}.csv")
        assert set(table["model"]) == EXPECTED_PHASE7_MODELS
        assert set(table["split"]) == {"validation", "test"}
        assert not table.loc[table["split"] == "validation", "test_set_used"].astype(bool).any()
        assert table.loc[table["split"] == "test", "test_set_used"].astype(bool).all()
        assert table["roc_auc"].between(0, 1).all()
        assert table["pr_auc"].between(0, 1).all()


def test_phase7_model_artifacts_exist():
    for dataset in ["taiwan", "heloc"]:
        for model in [
            "xgboost",
            "lightgbm",
            "catboost",
            "smotenc_xgboost",
            "class_weighted_lightgbm",
        ]:
            assert (BOOSTING_MODELS_DIR / dataset / f"{model}.joblib").exists()
        for model in ["monotonic_xgboost", "monotonic_lightgbm"]:
            assert (MONOTONIC_MODELS_DIR / dataset / f"{model}.joblib").exists()
