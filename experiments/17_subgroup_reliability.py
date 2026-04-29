"""Run Prompt 19 Taiwan subgroup reliability analysis."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import FN_COST, FP_COST
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from src.config.paths import CALIBRATED_MODELS_DIR, FIGURES_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.evaluation.subgroup_reliability import (
    SUBGROUP_RELIABILITY_NOTE,
    add_taiwan_age_group,
    plot_subgroup_calibration,
    plot_subgroup_cost,
    plot_subgroup_recall_fnr,
    subgroup_reliability_table,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


DATASET = "taiwan"
MODEL_NAME = "lightgbm"
CALIBRATION_METHOD = "isotonic"
SUBGROUP_COLUMNS = ["SEX", "AGE_GROUP", "EDUCATION", "MARRIAGE"]


def parse_args() -> argparse.Namespace:
    """Parse subgroup reliability runtime options."""

    parser = argparse.ArgumentParser(description="Run Taiwan subgroup reliability analysis.")
    parser.add_argument("--model", default=MODEL_NAME, help="Calibrated model artifact prefix.")
    parser.add_argument("--calibration-method", default=CALIBRATION_METHOD)
    parser.add_argument("--min-group-size", type=int, default=1)
    return parser.parse_args()


def selected_threshold(model_name: str, default: float = 0.5) -> float:
    """Load validation-selected Taiwan threshold for the requested model."""

    result_path = TABLES_DIR / "scre_credit_results_taiwan.csv"
    if result_path.exists():
        results = pd.read_csv(result_path)
        match = results.loc[
            (results["model"] == model_name)
            & (results["split"] == "validation")
            & (results["model_role"] == "base_model")
        ]
        if not match.empty:
            return float(match.iloc[0]["threshold"])

    threshold_path = TABLES_DIR / "threshold_analysis.csv"
    if threshold_path.exists():
        threshold_table = pd.read_csv(threshold_path)
        match = threshold_table.loc[
            (threshold_table["dataset"] == DATASET)
            & (threshold_table["model"] == model_name)
            & (threshold_table["calibration_method"] == CALIBRATION_METHOD)
            & (threshold_table["scenario"] == "B_FN5_FP1")
            & (threshold_table["is_best_threshold_for_model_scenario"] == True)
        ]
        if not match.empty:
            return float(match.iloc[0]["threshold"])
    return default


def main() -> None:
    """Run subgroup reliability on the reserved Taiwan test split."""

    args = parse_args()
    run_id = start_run(
        "subgroup_reliability",
        dataset=DATASET,
        params={
            "model": args.model,
            "calibration_method": args.calibration_method,
            "subgroup_columns": ", ".join(SUBGROUP_COLUMNS),
            "interpretation": SUBGROUP_RELIABILITY_NOTE,
        },
        tags=["phase-17", "subgroup-reliability"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        model_path = CALIBRATED_MODELS_DIR / DATASET / f"{args.model}_{args.calibration_method}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing calibrated model artifact: {model_path}")
        model = joblib.load(model_path)

        df = pd.read_csv(TAIWAN_MODEL_READY)
        split = stratified_train_validation_test_split(df, target_column=TARGET_COLUMN)
        X_test = split.test.drop(columns=[TARGET_COLUMN]).copy()
        y_test = split.test[TARGET_COLUMN].copy()
        X_grouped = add_taiwan_age_group(X_test)

        threshold = selected_threshold(args.model)
        y_proba = model.predict_proba(X_test)[:, 1]
        table = subgroup_reliability_table(
            X_grouped,
            y_test,
            y_proba,
            subgroup_columns=SUBGROUP_COLUMNS,
            threshold=threshold,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
            min_group_size=args.min_group_size,
        )
        table.insert(0, "test_set_used_for_training_or_selection", False)
        table.insert(0, "selection_split", "validation")
        table.insert(0, "calibration_method", args.calibration_method)
        table.insert(0, "model", args.model)
        table.insert(0, "dataset", DATASET)

        output_path = TABLES_DIR / "subgroup_reliability_taiwan.csv"
        table.to_csv(output_path, index=False)
        plot_subgroup_recall_fnr(table, FIGURES_DIR / "subgroup_recall_fnr.png")
        plot_subgroup_calibration(table, FIGURES_DIR / "subgroup_calibration.png")
        plot_subgroup_cost(table, FIGURES_DIR / "subgroup_cost.png")

        print(table.to_string(index=False))
        finish_run(
            run_id,
            metrics={
                "subgroup_rows": int(len(table)),
                "max_fnr": float(table["fnr"].max()),
                "max_ece": float(table["ece"].max()),
                "max_expected_cost_per_1000": float(table["expected_cost_per_1000"].max()),
            },
            artifacts={
                "subgroup_reliability_taiwan": output_path,
                "subgroup_recall_fnr": FIGURES_DIR / "subgroup_recall_fnr.png",
                "subgroup_calibration": FIGURES_DIR / "subgroup_calibration.png",
                "subgroup_cost": FIGURES_DIR / "subgroup_cost.png",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
