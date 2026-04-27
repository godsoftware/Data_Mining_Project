"""Hyperparameter optimization output tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import FIGURES_DIR, TABLES_DIR


def _optuna_status() -> pd.DataFrame:
    """Read the run-status file that distinguishes completed and long pending jobs."""

    status_path = TABLES_DIR / "optuna_run_status.csv"
    assert status_path.exists()
    status = pd.read_csv(status_path)
    assert {"dataset", "completed_trials_in_current_artifacts", "status"}.issubset(
        status.columns
    )
    return status


def _completed_optuna_datasets() -> set[str]:
    """Return datasets with completed trial artifacts in the current checkout."""

    status = _optuna_status()
    completed = status[
        (status["status"] == "completed")
        & (status["completed_trials_in_current_artifacts"] > 0)
    ]
    assert not completed.empty
    pending = status[status["status"] != "completed"]
    if not pending.empty:
        assert pending["status"].isin({"pending_long_run"}).all()
    return set(completed["dataset"])


def test_hyperparameter_search_results_exist_and_have_required_metrics():
    results = pd.read_csv(TABLES_DIR / "hyperparameter_search_results.csv")
    required = {
        "dataset",
        "trial_number",
        "primary_metric_pr_auc",
        "metric_roc_auc",
        "metric_recall",
        "metric_brier_score",
        "metric_fn_weighted_cost",
        "param_model_type",
    }
    assert required.issubset(results.columns)
    assert _completed_optuna_datasets().issubset(set(results["dataset"]))
    assert results["primary_metric_pr_auc"].between(0, 1).all()


def test_best_params_and_optuna_figures_exist():
    best = pd.read_csv(TABLES_DIR / "best_params.csv")
    assert _completed_optuna_datasets().issubset(set(best["dataset"]))
    assert not best["test_set_used"].astype(bool).any()
    assert (FIGURES_DIR / "optuna_optimization_history.png").exists()
    assert (FIGURES_DIR / "optuna_param_importance.png").exists()
