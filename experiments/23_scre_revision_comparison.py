"""Compare old SCRE, revised SCRE variants, and best single-model families."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import EXPERIMENT_LOGS_DIR, TABLES_DIR
from src.utils.logging_utils import create_experiment_logger


PRIMARY_COST_SCENARIO = "B_FN5_FP1"

MODEL_FAMILIES = {
    "Best CatBoost": ["catboost", "optuna_best_catboost"],
    "Best XGBoost": ["xgboost", "optuna_best_xgboost", "smotenc_xgboost"],
    "Best LightGBM": ["lightgbm", "class_weighted_lightgbm"],
    "Best Monotonic model": ["monotonic_xgboost", "monotonic_lightgbm"],
    "Best Scorecard": ["woe_scorecard_logistic_regression"],
    "Old SCRE-Credit": ["SCRE-Credit"],
}

INTERPRETABILITY_RANK = {
    "Best Scorecard": 1,
    "Best Monotonic model": 2,
    "Old SCRE-Credit": 3,
    "SCRE-Pareto": 3,
    "SCRE-Optimized": 3,
    "Best LightGBM": 4,
    "Best CatBoost": 5,
    "Best XGBoost": 5,
}

OUTPUT_COLUMNS = [
    "dataset",
    "model",
    "calibration_type",
    "threshold",
    "cost_scenario",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "expected_cost",
    "shap_stability",
    "faithfulness_score",
    "operational_rank",
    "auc_rank",
    "pr_auc_rank",
    "calibration_rank",
    "interpretability_rank",
    "final_comment",
]


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV with a clear error."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def select_validation_best(results: pd.DataFrame, candidate_models: list[str]) -> str:
    """Select the best source model within a family using validation-only metrics."""

    candidates = results.loc[
        (results["split"] == "validation")
        & (results["model"].isin(candidate_models))
        & (results["cost_scenario"] == PRIMARY_COST_SCENARIO)
    ].copy()
    if candidates.empty:
        raise ValueError(f"No validation candidates found for {candidate_models}.")
    candidates = candidates.sort_values(
        ["expected_cost", "pr_auc", "roc_auc", "brier_score", "ece"],
        ascending=[True, False, False, True, True],
    )
    return str(candidates.iloc[0]["model"])


def get_test_row(results: pd.DataFrame, source_model: str) -> pd.Series:
    """Return one final test row for a selected source model."""

    rows = results.loc[
        (results["split"] == "test")
        & (results["model"] == source_model)
        & (results["cost_scenario"] == PRIMARY_COST_SCENARIO)
    ]
    if rows.empty:
        raise ValueError(f"No test row found for selected source model {source_model!r}.")
    return rows.iloc[0]


def lookup_explanation_scores(dataset: str, source_model: str) -> tuple[float, float]:
    """Return comparable explanation diagnostic scores where they exist."""

    leaderboard_path = TABLES_DIR / f"final_leaderboard_{dataset}.csv"
    if leaderboard_path.exists():
        leaderboard = pd.read_csv(leaderboard_path)
        match = leaderboard.loc[leaderboard["model"] == source_model]
        if not match.empty:
            shap = match["shap_stability"].iloc[0] if "shap_stability" in match else np.nan
            faith = match["faithfulness_score"].iloc[0] if "faithfulness_score" in match else np.nan
            return float(shap) if pd.notna(shap) else np.nan, float(faith) if pd.notna(faith) else np.nan

    return np.nan, np.nan


def row_from_result(
    dataset: str,
    display_model: str,
    source_model: str,
    result_row: pd.Series,
    final_comment: str,
) -> dict[str, object]:
    """Convert a model result row into the final comparison schema."""

    shap_stability, faithfulness_score = lookup_explanation_scores(dataset, source_model)
    calibration_type = result_row.get("calibration_type", result_row.get("calibration_method", "unknown"))
    return {
        "dataset": dataset,
        "model": display_model,
        "calibration_type": calibration_type,
        "threshold": float(result_row["threshold"]),
        "cost_scenario": result_row.get("cost_scenario", PRIMARY_COST_SCENARIO),
        "accuracy": float(result_row["accuracy"]),
        "precision": float(result_row["precision"]),
        "recall": float(result_row["recall"]),
        "f1": float(result_row["f1"]),
        "specificity": float(result_row["specificity"]),
        "roc_auc": float(result_row["roc_auc"]),
        "pr_auc": float(result_row["pr_auc"]),
        "brier": float(result_row.get("brier", result_row.get("brier_score"))),
        "ece": float(result_row["ece"]),
        "expected_cost": float(result_row["expected_cost"]),
        "shap_stability": shap_stability,
        "faithfulness_score": faithfulness_score,
        "operational_rank": np.nan,
        "auc_rank": np.nan,
        "pr_auc_rank": np.nan,
        "calibration_rank": np.nan,
        "interpretability_rank": INTERPRETABILITY_RANK.get(display_model, np.nan),
        "final_comment": final_comment,
    }


def load_scre_variant_row(dataset: str, variant: str) -> pd.Series:
    """Load a final test row for a revised SCRE variant."""

    if variant == "SCRE-Pareto":
        table = read_csv(TABLES_DIR / f"scre_pareto_results_{dataset}.csv")
        rows = table.loc[(table["split"] == "test") & (table["model"] == "SCRE-Pareto")]
    elif variant == "SCRE-Optimized":
        table = read_csv(TABLES_DIR / f"scre_optimized_results_{dataset}.csv")
        rows = table.loc[
            (table["split"] == "test")
            & (table["model"] == "SCRE-Optimized")
            & (table["selected_final_setting"].astype(bool))
        ]
    else:
        raise ValueError(f"Unknown SCRE variant: {variant}")
    if rows.empty:
        raise ValueError(f"No final test row found for {dataset}/{variant}.")
    return rows.iloc[0]


def add_ranks(table: pd.DataFrame) -> pd.DataFrame:
    """Add operational, AUC, PR-AUC, and calibration ranks."""

    ranked = table.copy()
    ranked["operational_rank"] = ranked["expected_cost"].rank(method="min", ascending=True).astype(int)
    ranked["auc_rank"] = ranked["roc_auc"].rank(method="min", ascending=False).astype(int)
    ranked["pr_auc_rank"] = ranked["pr_auc"].rank(method="min", ascending=False).astype(int)
    brier_rank = ranked["brier"].rank(method="min", ascending=True)
    ece_rank = ranked["ece"].rank(method="min", ascending=True)
    ranked["calibration_rank"] = (brier_rank + ece_rank).rank(method="min", ascending=True).astype(int)
    ranked["interpretability_rank"] = ranked["interpretability_rank"].astype(int)
    return ranked


def conclusion_for_row(dataset: str, table: pd.DataFrame, row: pd.Series, source_note: str) -> str:
    """Create a compact, model-specific final comment."""

    operational_winner = table.sort_values(["expected_cost", "pr_auc"], ascending=[True, False]).iloc[0]
    pr_auc_winner = table.sort_values("pr_auc", ascending=False).iloc[0]
    calibration_winner = table.sort_values(["brier", "ece"], ascending=[True, True]).iloc[0]
    comments = [source_note]
    if row["model"] == operational_winner["model"]:
        comments.append("Operational winner on final test under FN=5, FP=1.")
    if row["model"] == pr_auc_winner["model"]:
        comments.append("Best PR-AUC in this comparison.")
    if row["model"] == calibration_winner["model"]:
        comments.append("Best Brier-first calibration profile in this comparison.")
    if row["model"] == "SCRE-Pareto":
        catboost_cost = float(table.loc[table["model"] == "Best CatBoost", "expected_cost"].iloc[0])
        outcome = "does not beat" if float(row["expected_cost"]) > catboost_cost else "beats or ties"
        comments.append(f"SCRE-Pareto {outcome} Best CatBoost on final-test expected cost.")
    if row["model"] == "SCRE-Optimized":
        catboost_cost = float(table.loc[table["model"] == "Best CatBoost", "expected_cost"].iloc[0])
        outcome = "does not beat" if float(row["expected_cost"]) > catboost_cost else "beats or ties"
        comments.append(f"SCRE-Optimized {outcome} Best CatBoost on final-test expected cost.")
    if row["model"].startswith("SCRE") or row["model"] == "Old SCRE-Credit":
        comments.append("Use as reliability-aware framework evidence, not as a blanket superiority claim.")
    if dataset == "heloc":
        comments.append("HELOC is external validation; domain shift can change the operational winner.")
    return " ".join(comments)


def build_dataset_comparison(dataset: str, logger) -> pd.DataFrame:
    """Build one dataset comparison table."""

    base_results = read_csv(TABLES_DIR / f"scre_credit_results_{dataset}.csv")
    rows: list[dict[str, object]] = []
    source_notes: dict[str, str] = {}

    for display_model, candidates in MODEL_FAMILIES.items():
        source_model = select_validation_best(base_results, candidates)
        test_row = get_test_row(base_results, source_model)
        note = (
            f"Source model selected on validation within family: {source_model}; "
            "reported metrics are final test evidence."
        )
        source_notes[display_model] = note
        rows.append(row_from_result(dataset, display_model, source_model, test_row, note))

    for variant in ["SCRE-Pareto", "SCRE-Optimized"]:
        test_row = load_scre_variant_row(dataset, variant)
        setting = test_row.get("setting_id", "")
        note = (
            "Weights, objective setting, and threshold selected on validation only; "
            f"reported row is final test evidence{f' for setting {setting}' if setting else ''}."
        )
        source_notes[variant] = note
        rows.append(row_from_result(dataset, variant, variant, test_row, note))

    table = add_ranks(pd.DataFrame(rows))
    table["final_comment"] = [
        conclusion_for_row(dataset, table, row, source_notes[str(row["model"])])
        for _, row in table.iterrows()
    ]
    table = table.sort_values(["operational_rank", "pr_auc_rank", "auc_rank"]).reset_index(drop=True)
    logger.info("%s comparison:\n%s", dataset, table[["model", "expected_cost", "pr_auc", "roc_auc"]].to_string(index=False))
    return table[OUTPUT_COLUMNS]


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> pd.DataFrame:
    """Write dataset-specific and global comparison tables."""

    taiwan_path = TABLES_DIR / "scre_revision_comparison_taiwan.csv"
    heloc_path = TABLES_DIR / "scre_revision_comparison_heloc.csv"
    global_path = TABLES_DIR / "scre_revision_global_comparison.csv"
    taiwan.to_csv(taiwan_path, index=False)
    heloc.to_csv(heloc_path, index=False)
    global_table = pd.concat([taiwan, heloc], ignore_index=True)
    global_table.to_csv(global_path, index=False)
    return global_table


def main() -> None:
    """Create final SCRE revision comparison tables."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger(
        "scre_revision_comparison",
        EXPERIMENT_LOGS_DIR / "23_scre_revision_comparison.log",
    )
    taiwan = build_dataset_comparison("taiwan", logger)
    heloc = build_dataset_comparison("heloc", logger)
    global_table = write_outputs(taiwan, heloc)

    for dataset, table in [("taiwan", taiwan), ("heloc", heloc)]:
        print(f"\n{dataset.upper()}")
        print(
            table[
                [
                    "model",
                    "expected_cost",
                    "operational_rank",
                    "pr_auc",
                    "pr_auc_rank",
                    "roc_auc",
                    "auc_rank",
                    "brier",
                    "ece",
                    "calibration_rank",
                ]
            ].to_string(index=False)
        )
    logger.info("Saved %s", TABLES_DIR / "scre_revision_comparison_taiwan.csv")
    logger.info("Saved %s", TABLES_DIR / "scre_revision_comparison_heloc.csv")
    logger.info("Saved %s", TABLES_DIR / "scre_revision_global_comparison.csv")
    logger.info("Global comparison rows=%s", len(global_table))


if __name__ == "__main__":
    main()
