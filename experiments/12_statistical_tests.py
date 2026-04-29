"""Run Prompt 14 statistical tests for SCRE-Credit model comparisons."""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import FN_COST, FP_COST, RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_TARGET
from run_calibration_analysis import _positive_class_probability
from src.config.paths import FIGURES_DIR, HELOC_MODEL_READY, MODELS_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.evaluation.statistical_tests import (
    BOOTSTRAP_METRICS,
    ModelPrediction,
    bootstrap_confidence_interval_row,
    clean_model_comparison_tests,
    comparison_test_row,
    mcnemar_test_row,
    plot_bootstrap_metric_distributions,
)


N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

DATASET_CONFIGS = {
    "taiwan": {"path": TAIWAN_MODEL_READY, "target": TARGET_COLUMN},
    "heloc": {"path": HELOC_MODEL_READY, "target": HELOC_TARGET},
}

COMPARISON_MODELS = {
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "CatBoost": "catboost",
    "Scorecard": "woe_scorecard_logistic_regression",
}


def stable_seed(*parts: str, base_seed: int = RANDOM_SEED) -> int:
    """Create a deterministic 32-bit seed from labels and the central seed."""

    payload = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def load_test_split(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the reserved test split for one dataset."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_test = split.test.drop(columns=[config["target"]])
    y_test = split.test[config["target"]].astype(int).reset_index(drop=True)
    return X_test.reset_index(drop=True), y_test


def load_base_probabilities(weights: pd.DataFrame, dataset: str, X_test: pd.DataFrame) -> pd.DataFrame:
    """Load calibrated base-model artifacts and predict test probabilities."""

    dataset_weights = weights.loc[weights["dataset"] == dataset].copy()
    probabilities = {}
    for row in dataset_weights.itertuples(index=False):
        model_path = Path(row.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing calibrated model artifact: {model_path}")
        fitted_model = joblib.load(model_path)
        probabilities[row.model] = _positive_class_probability(fitted_model, X_test)
    return pd.DataFrame(probabilities)


def build_model_predictions(dataset: str) -> tuple[np.ndarray, dict[str, ModelPrediction]]:
    """Create SCRE-Credit and comparator test-set prediction objects."""

    weights = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
    bundle_path = MODELS_DIR / "scre_credit_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing SCRE-Credit model bundle: {bundle_path}")

    model_bundle = joblib.load(bundle_path)
    classifier = model_bundle[dataset]["classifier"]
    X_test, y_test = load_test_split(dataset)
    base_probabilities = load_base_probabilities(weights, dataset, X_test)

    predictions: dict[str, ModelPrediction] = {}
    scre_proba = classifier.predict_proba_from_base(base_probabilities[classifier.model_names_])[:, 1]
    predictions["SCRE-Credit"] = ModelPrediction(
        model="SCRE-Credit",
        y_proba=scre_proba,
        threshold=float(classifier.threshold_),
    )

    dataset_weights = weights.loc[weights["dataset"] == dataset].set_index("model")
    for display_name, model_name in COMPARISON_MODELS.items():
        if model_name not in dataset_weights.index:
            raise KeyError(f"Required comparator {model_name!r} is missing for {dataset}.")
        predictions[display_name] = ModelPrediction(
            model=display_name,
            y_proba=base_probabilities[model_name].to_numpy(),
            threshold=float(dataset_weights.loc[model_name, "base_selected_threshold"]),
        )
    return y_test.to_numpy(), predictions


def build_bootstrap_ci_outputs(
    dataset: str,
    y_true: np.ndarray,
    predictions: dict[str, ModelPrediction],
) -> tuple[list[dict], list[dict]]:
    """Build bootstrap CI rows and SCRE distribution rows for plotting."""

    ci_rows = []
    distribution_rows = []
    for model_name, prediction in predictions.items():
        for metric in BOOTSTRAP_METRICS:
            row, values = bootstrap_confidence_interval_row(
                dataset=dataset,
                prediction=prediction,
                y_true=y_true,
                metric=metric,
                n_bootstrap=N_BOOTSTRAP,
                random_seed=stable_seed(dataset, model_name, metric, "ci"),
                confidence_level=CONFIDENCE_LEVEL,
                fn_cost=FN_COST,
                fp_cost=FP_COST,
            )
            ci_rows.append(row)
            if model_name == "SCRE-Credit":
                distribution_rows.extend(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "metric": metric,
                        "value": float(value),
                    }
                    for value in values
                )
    return ci_rows, distribution_rows


def build_model_comparison_outputs(
    dataset: str,
    y_true: np.ndarray,
    predictions: dict[str, ModelPrediction],
) -> list[dict]:
    """Build paired bootstrap, McNemar, and bootstrap AUC comparison rows."""

    rows = []
    scre = predictions["SCRE-Credit"]
    for comparator_name in COMPARISON_MODELS:
        comparator = predictions[comparator_name]
        for metric in BOOTSTRAP_METRICS:
            rows.append(
                comparison_test_row(
                    dataset=dataset,
                    y_true=y_true,
                    scre=scre,
                    comparator=comparator,
                    metric=metric,
                    n_bootstrap=N_BOOTSTRAP,
                    random_seed=stable_seed(dataset, comparator_name, metric, "paired"),
                    confidence_level=CONFIDENCE_LEVEL,
                    fn_cost=FN_COST,
                    fp_cost=FP_COST,
                    method="paired_bootstrap",
                )
            )
        rows.append(
            mcnemar_test_row(
                dataset=dataset,
                y_true=y_true,
                scre=scre,
                comparator=comparator,
                n_bootstrap=N_BOOTSTRAP,
                random_seed=stable_seed(dataset, comparator_name, "mcnemar"),
                confidence_level=CONFIDENCE_LEVEL,
            )
        )
        rows.append(
            comparison_test_row(
                dataset=dataset,
                y_true=y_true,
                scre=scre,
                comparator=comparator,
                metric="roc_auc",
                n_bootstrap=N_BOOTSTRAP,
                random_seed=stable_seed(dataset, comparator_name, "roc_auc", "auc_comparison"),
                confidence_level=CONFIDENCE_LEVEL,
                fn_cost=FN_COST,
                fp_cost=FP_COST,
                method="bootstrap_auc_comparison",
            )
        )
    return rows


def main() -> None:
    """Run all Prompt 14 statistical tests and save outputs."""

    run_id = start_run(
        "statistical_tests",
        dataset="both",
        params={
            "n_bootstrap": N_BOOTSTRAP,
            "confidence_level": CONFIDENCE_LEVEL,
            "random_seed": RANDOM_SEED,
        },
        tags=["phase-12", "statistical-testing"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        ci_rows = []
        comparison_rows = []
        distribution_rows = []

        for dataset in ["taiwan", "heloc"]:
            print(f"Running statistical tests for {dataset} with {N_BOOTSTRAP} bootstrap samples")
            y_true, predictions = build_model_predictions(dataset)
            dataset_ci_rows, dataset_distribution_rows = build_bootstrap_ci_outputs(
                dataset,
                y_true,
                predictions,
            )
            ci_rows.extend(dataset_ci_rows)
            distribution_rows.extend(dataset_distribution_rows)
            comparison_rows.extend(build_model_comparison_outputs(dataset, y_true, predictions))

        ci_table = pd.DataFrame(ci_rows)
        comparison_table = pd.DataFrame(comparison_rows)
        comparison_clean = clean_model_comparison_tests(comparison_table)
        distribution_table = pd.DataFrame(distribution_rows)

        ci_path = TABLES_DIR / "bootstrap_confidence_intervals.csv"
        comparison_path = TABLES_DIR / "model_comparison_tests.csv"
        comparison_clean_path = TABLES_DIR / "model_comparison_tests_clean.csv"
        figure_path = FIGURES_DIR / "bootstrap_metric_distributions.png"
        ci_table.to_csv(ci_path, index=False)
        comparison_table.to_csv(comparison_path, index=False)
        comparison_clean.to_csv(comparison_clean_path, index=False)
        plot_bootstrap_metric_distributions(distribution_table, figure_path)

        print(ci_table.head(12).to_string(index=False))
        print(
            comparison_table[
                [
                    "dataset",
                    "comparison",
                    "method",
                    "metric",
                    "mean",
                    "lower_ci",
                    "upper_ci",
                    "p_value",
                ]
            ]
            .head(16)
            .to_string(index=False)
        )

        finish_run(
            run_id,
            metrics={
                "bootstrap_ci_rows": int(len(ci_table)),
                "comparison_test_rows": int(len(comparison_table)),
                "comparison_test_clean_rows": int(len(comparison_clean)),
                "n_bootstrap": N_BOOTSTRAP,
            },
            artifacts={
                "bootstrap_confidence_intervals": ci_path,
                "model_comparison_tests": comparison_path,
                "model_comparison_tests_clean": comparison_clean_path,
                "bootstrap_metric_distributions": figure_path,
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
