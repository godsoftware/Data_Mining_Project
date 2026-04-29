"""Run Prompt 18 faithfulness sanity checks."""

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

from config.settings import FN_COST, FP_COST, RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from src.config.paths import CALIBRATED_MODELS_DIR, FIGURES_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.explainability.faithfulness import (
    TAIWAN_FEATURE_GROUPS,
    baseline_feature_values,
    faithfulness_metrics,
    group_deletion_table,
    plot_auc_drop_by_feature,
    plot_deletion_curve,
    plot_insertion_curve,
    random_feature_deletion_table,
    single_feature_perturbation_table,
    topk_deletion_table,
    topk_insertion_table,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


DATASET = "taiwan"
MODEL_NAME = "lightgbm"
CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]


def parse_args() -> argparse.Namespace:
    """Parse runtime options."""

    parser = argparse.ArgumentParser(description="Run faithfulness sanity checks.")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--max-k", type=int, default=10)
    return parser.parse_args()


def selected_threshold(default: float = 0.5) -> float:
    """Load LightGBM validation-selected threshold from SCRE comparison results."""

    result_path = TABLES_DIR / "scre_credit_results_taiwan.csv"
    if not result_path.exists():
        return default
    results = pd.read_csv(result_path)
    match = results.loc[
        (results["model"] == MODEL_NAME)
        & (results["split"] == "validation")
        & (results["model_role"] == "base_model")
    ]
    if match.empty:
        return default
    return float(match.iloc[0]["threshold"])


def ranked_raw_features(candidate_columns: list[str]) -> list[str]:
    """Return SHAP-ranked raw features followed by any remaining model columns."""

    shap_path = TABLES_DIR / "shap_top_features.csv"
    if not shap_path.exists():
        raise FileNotFoundError(f"Missing SHAP top features table: {shap_path}")
    shap_table = pd.read_csv(shap_path).sort_values("rank")
    features = []
    for feature in shap_table["raw_feature"].tolist():
        if feature in candidate_columns and feature not in features:
            features.append(feature)
    for feature in candidate_columns:
        if feature not in features:
            features.append(feature)
    return features


def main() -> None:
    """Run faithfulness checks and save outputs."""

    args = parse_args()
    run_id = start_run(
        "faithfulness_analysis",
        dataset=DATASET,
        params={
            "model": MODEL_NAME,
            "sample_size": args.sample_size,
            "max_k": args.max_k,
            "random_seed": RANDOM_SEED,
            "interpretation": "model behavior sanity check, not causal proof",
        },
        tags=["phase-16", "faithfulness"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        model_path = CALIBRATED_MODELS_DIR / DATASET / f"{MODEL_NAME}_uncalibrated.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        model = joblib.load(model_path)

        df = pd.read_csv(TAIWAN_MODEL_READY)
        split = stratified_train_validation_test_split(df, target_column=TARGET_COLUMN)
        X_train = split.train.drop(columns=[TARGET_COLUMN])
        X_test = split.test.drop(columns=[TARGET_COLUMN])
        y_test = split.test[TARGET_COLUMN]
        if len(X_test) > args.sample_size:
            sampled_index = X_test.sample(args.sample_size, random_state=RANDOM_SEED).index
            X_eval = X_test.loc[sampled_index].copy()
            y_eval = y_test.loc[sampled_index].copy()
        else:
            X_eval = X_test.copy()
            y_eval = y_test.copy()

        threshold = selected_threshold()
        replacements = baseline_feature_values(X_train, categorical_columns=CATEGORICAL_COLUMNS)
        baseline_proba = model.predict_proba(X_eval)[:, 1]
        baseline_metrics = faithfulness_metrics(
            y_eval,
            baseline_proba,
            threshold,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        top_features = ranked_raw_features(list(X_eval.columns))

        single = single_feature_perturbation_table(
            model,
            X_eval,
            y_eval,
            top_features,
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        deletion = topk_deletion_table(
            model,
            X_eval,
            y_eval,
            top_features,
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            max_k=args.max_k,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        insertion = topk_insertion_table(
            model,
            X_eval,
            y_eval,
            top_features,
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            max_k=args.max_k,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        random_deletion = random_feature_deletion_table(
            model,
            X_eval,
            y_eval,
            list(X_eval.columns),
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            random_seed=RANDOM_SEED,
            max_k=args.max_k,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )
        group_results = group_deletion_table(
            model,
            X_eval,
            y_eval,
            TAIWAN_FEATURE_GROUPS,
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
        )

        faithfulness_results = pd.concat(
            [single, deletion, insertion, random_deletion],
            ignore_index=True,
        )
        for column, value in [
            ("dataset", DATASET),
            ("model", MODEL_NAME),
            ("threshold", threshold),
            ("baseline_roc_auc", baseline_metrics["roc_auc"]),
            ("baseline_pr_auc", baseline_metrics["pr_auc"]),
            ("baseline_recall", baseline_metrics["recall"]),
            ("baseline_expected_cost", baseline_metrics["expected_cost"]),
            ("analysis_interpretation", "model behavior sanity check, not causal proof"),
        ]:
            faithfulness_results.insert(0, column, value)

        for column, value in [
            ("dataset", DATASET),
            ("model", MODEL_NAME),
            ("threshold", threshold),
            ("baseline_roc_auc", baseline_metrics["roc_auc"]),
            ("baseline_pr_auc", baseline_metrics["pr_auc"]),
            ("baseline_recall", baseline_metrics["recall"]),
            ("baseline_expected_cost", baseline_metrics["expected_cost"]),
            ("analysis_interpretation", "model behavior sanity check, not causal proof"),
        ]:
            group_results.insert(0, column, value)

        results_path = TABLES_DIR / "faithfulness_results.csv"
        group_path = TABLES_DIR / "faithfulness_group_results.csv"
        faithfulness_results.to_csv(results_path, index=False)
        group_results.to_csv(group_path, index=False)

        plot_deletion_curve(faithfulness_results, FIGURES_DIR / "deletion_curve.png")
        plot_insertion_curve(faithfulness_results, FIGURES_DIR / "insertion_curve.png")
        plot_auc_drop_by_feature(faithfulness_results, FIGURES_DIR / "auc_drop_by_feature.png")

        print(faithfulness_results.head(15).to_string(index=False))
        print(group_results.to_string(index=False))
        finish_run(
            run_id,
            metrics={
                "baseline_roc_auc": float(baseline_metrics["roc_auc"]),
                "baseline_pr_auc": float(baseline_metrics["pr_auc"]),
                "faithfulness_rows": int(len(faithfulness_results)),
                "group_rows": int(len(group_results)),
                "max_single_feature_roc_auc_drop": float(single["roc_auc_drop"].max()),
                "max_group_roc_auc_drop": float(group_results["roc_auc_drop"].max()),
            },
            artifacts={
                "faithfulness_results": results_path,
                "faithfulness_group_results": group_path,
                "deletion_curve": FIGURES_DIR / "deletion_curve.png",
                "insertion_curve": FIGURES_DIR / "insertion_curve.png",
                "auc_drop_by_feature": FIGURES_DIR / "auc_drop_by_feature.png",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
