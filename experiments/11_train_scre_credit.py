"""Train the Prompt 13 SCRE-Credit hybrid classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_TARGET
from run_calibration_analysis import _positive_class_probability
from src.config.paths import HELOC_MODEL_READY, MODELS_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.models.scre_credit import SCRECreditHybridClassifier


PRIMARY_COST_SCENARIO = "B_FN5_FP1"
NEUTRAL_STABILITY_SCORE = 1.0
NEUTRAL_FAITHFULNESS_SCORE = 1.0


DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "calibration_results": TABLES_DIR / "calibration_results_taiwan.csv",
        "threshold_analysis": TABLES_DIR / "threshold_analysis_taiwan.csv",
        "output": TABLES_DIR / "scre_credit_results_taiwan.csv",
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "calibration_results": TABLES_DIR / "calibration_results_heloc.csv",
        "threshold_analysis": TABLES_DIR / "threshold_analysis_heloc.csv",
        "output": TABLES_DIR / "scre_credit_results_heloc.csv",
    },
}


def _select_best_calibrated_row_per_model(calibration_results: pd.DataFrame) -> pd.DataFrame:
    """Select one calibrated probability source per base model."""

    eligible = calibration_results.loc[
        calibration_results["calibration_method"].isin(["sigmoid", "isotonic"])
    ].copy()
    if eligible.empty:
        raise ValueError("No sigmoid/isotonic calibrated model rows found.")

    eligible = eligible.sort_values(
        ["model", "brier_score", "ece", "pr_auc"],
        ascending=[True, True, True, False],
    )
    selected = eligible.drop_duplicates("model", keep="first").reset_index(drop=True)
    missing_paths = [
        path for path in selected["model_path"].tolist()
        if not isinstance(path, str) or not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing calibrated model artifacts: {missing_paths}")
    return selected


def _best_cost_rows(threshold_analysis: pd.DataFrame) -> pd.DataFrame:
    """Return validation cost-minimizing rows for the primary cost scenario."""

    scenario_rows = threshold_analysis.loc[
        threshold_analysis["scenario"] == PRIMARY_COST_SCENARIO
    ].copy()
    if scenario_rows.empty:
        raise ValueError(f"No threshold rows found for scenario {PRIMARY_COST_SCENARIO}.")
    idx = scenario_rows.groupby(["model", "calibration_method"])["expected_cost"].idxmin()
    return scenario_rows.loc[idx].copy()


def build_scre_metric_table(
    selected_calibration_rows: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Merge calibration and cost metrics into SCRE-Credit input form."""

    best_cost = _best_cost_rows(threshold_analysis)
    cost_keys = [
        "model",
        "calibration_method",
        "expected_cost",
        "threshold",
        "fn_cost",
        "fp_cost",
    ]
    merged = selected_calibration_rows.merge(
        best_cost[cost_keys],
        on=["model", "calibration_method"],
        how="left",
        suffixes=("", "_cost_selected"),
    )
    if merged["expected_cost_cost_selected"].isna().any():
        missing = merged.loc[merged["expected_cost_cost_selected"].isna(), ["model", "calibration_method"]]
        raise ValueError(f"Missing cost metrics for selected calibrated models:\n{missing}")

    return pd.DataFrame(
        {
            "model": merged["model"],
            "source_phase": merged["source_phase"],
            "model_family": merged["model_family"],
            "selected_calibration_method": merged["calibration_method"],
            "model_path": merged["model_path"],
            "pr_auc": merged["pr_auc"],
            "roc_auc": merged["roc_auc"],
            "recall": merged["recall"],
            "calibration_error": merged["brier_score"],
            "brier_score": merged["brier_score"],
            "ece": merged["ece"],
            "expected_cost": merged["expected_cost_cost_selected"],
            "base_selected_threshold": merged["threshold_cost_selected"],
            "cost_scenario": PRIMARY_COST_SCENARIO,
            "stability_score": NEUTRAL_STABILITY_SCORE,
            "faithfulness_score": NEUTRAL_FAITHFULNESS_SCORE,
            "stability_metric_source": "neutral_placeholder_until_model_level_stability_prompt",
            "faithfulness_metric_source": "neutral_placeholder_until_model_level_faithfulness_prompt",
        }
    )


def load_probability_frame(selected_rows: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Load calibrated model artifacts and predict positive-class probabilities."""

    probabilities = {}
    for row in selected_rows.itertuples(index=False):
        fitted_model = joblib.load(row.model_path)
        probabilities[row.model] = _positive_class_probability(fitted_model, X)
    return pd.DataFrame(probabilities, index=X.index)


def _metrics_row(
    dataset: str,
    model: str,
    split: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    model_role: str,
    calibration_method: str,
    threshold_source: str,
) -> dict:
    """Create one model-comparison metrics row."""

    metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
    return {
        "dataset": dataset,
        "model": model,
        "model_role": model_role,
        "split": split,
        "calibration_method": calibration_method,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": 5.0,
        "fp_cost": 1.0,
        "expected_cost": float(5.0 * metrics["fn"] + metrics["fp"]),
        "test_set_used_for_training_or_selection": False,
        **metrics,
    }


def evaluate_base_models(
    dataset: str,
    metric_table: pd.DataFrame,
    validation_probabilities: pd.DataFrame,
    test_probabilities: pd.DataFrame,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> list[dict]:
    """Evaluate all calibrated base models on validation and test."""

    rows = []
    for record in metric_table.itertuples(index=False):
        threshold = float(record.base_selected_threshold)
        for split_name, y_true, probabilities in [
            ("validation", y_validation, validation_probabilities),
            ("test", y_test, test_probabilities),
        ]:
            rows.append(
                _metrics_row(
                    dataset=dataset,
                    model=record.model,
                    split=split_name,
                    y_true=y_true,
                    y_proba=probabilities[record.model].to_numpy(),
                    threshold=threshold,
                    model_role="base_model",
                    calibration_method=record.selected_calibration_method,
                    threshold_source="validation_cost_min_B_FN5_FP1",
                )
            )
    return rows


def run_scre_for_dataset(dataset: str) -> tuple[SCRECreditHybridClassifier, pd.DataFrame, pd.DataFrame]:
    """Fit and evaluate SCRE-Credit for one dataset."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])

    X_validation = split.validation.drop(columns=[config["target"]])
    y_validation = split.validation[config["target"]]
    X_test = split.test.drop(columns=[config["target"]])
    y_test = split.test[config["target"]]

    calibration_results = pd.read_csv(config["calibration_results"])
    threshold_analysis = pd.read_csv(config["threshold_analysis"])
    selected_rows = _select_best_calibrated_row_per_model(calibration_results)
    metric_table = build_scre_metric_table(selected_rows, threshold_analysis)

    validation_probabilities = load_probability_frame(selected_rows, X_validation)
    test_probabilities = load_probability_frame(selected_rows, X_test)

    scre = SCRECreditHybridClassifier(final_calibration_method="isotonic", fn_cost=5.0, fp_cost=1.0)
    scre.fit_from_probabilities(validation_probabilities, y_validation, metric_table)

    result_rows = evaluate_base_models(
        dataset,
        metric_table,
        validation_probabilities,
        test_probabilities,
        y_validation,
        y_test,
    )
    result_rows.append(
        _metrics_row(
            dataset=dataset,
            model="SCRE-Credit",
            split="validation",
            y_true=y_validation,
            y_proba=scre.validation_probability_,
            threshold=scre.threshold_,
            model_role="scre_credit",
            calibration_method=f"weighted_sum_plus_{scre.final_calibration_method}",
            threshold_source="validation_cost_min_B_FN5_FP1",
        )
    )
    scre_test_probability = scre.predict_proba_from_base(test_probabilities)[:, 1]
    result_rows.append(
        _metrics_row(
            dataset=dataset,
            model="SCRE-Credit",
            split="test",
            y_true=y_test,
            y_proba=scre_test_probability,
            threshold=scre.threshold_,
            model_role="scre_credit",
            calibration_method=f"weighted_sum_plus_{scre.final_calibration_method}",
            threshold_source="validation_cost_min_B_FN5_FP1",
        )
    )

    results = pd.DataFrame(result_rows).sort_values(
        ["split", "model_role", "expected_cost", "pr_auc"],
        ascending=[True, True, True, False],
    )
    config["output"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config["output"], index=False)

    weights = scre.weight_table_.copy()
    weights.insert(0, "dataset", dataset)
    return scre, weights, results


def main() -> None:
    """Run SCRE-Credit Prompt 13 for Taiwan and HELOC."""

    run_id = start_run("train_scre_credit", dataset="both", tags=["phase-11", "scre-credit"])
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_bundle = {}
        all_weights = []
        metrics = {}
        artifacts = {}
        for dataset in ["taiwan", "heloc"]:
            print(f"Training SCRE-Credit hybrid classifier for {dataset}")
            classifier, weights, results = run_scre_for_dataset(dataset)
            model_bundle[dataset] = {
                "classifier": classifier,
                "model_names": classifier.model_names_,
                "threshold": classifier.threshold_,
                "final_calibration_method": classifier.final_calibration_method,
            }
            all_weights.append(weights)
            artifacts[f"{dataset}_scre_credit_results"] = DATASET_CONFIGS[dataset]["output"]
            test_row = results.loc[
                (results["split"] == "test") & (results["model"] == "SCRE-Credit")
            ].iloc[0]
            metrics[f"{dataset}_scre_pr_auc_test"] = float(test_row["pr_auc"])
            metrics[f"{dataset}_scre_expected_cost_test"] = float(test_row["expected_cost"])
            print(
                results.loc[results["split"] == "test", [
                    "dataset",
                    "model",
                    "model_role",
                    "threshold",
                    "expected_cost",
                    "pr_auc",
                    "roc_auc",
                    "recall",
                    "brier_score",
                ]]
                .sort_values(["expected_cost", "pr_auc"], ascending=[True, False])
                .head(8)
                .to_string(index=False)
            )

        weight_table = pd.concat(all_weights, ignore_index=True)
        weight_path = TABLES_DIR / "scre_model_weights.csv"
        weight_table.to_csv(weight_path, index=False)
        model_path = MODELS_DIR / "scre_credit_model.joblib"
        joblib.dump(model_bundle, model_path)

        artifacts["scre_model_weights"] = weight_path
        artifacts["scre_credit_model"] = model_path
        finish_run(run_id, metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
