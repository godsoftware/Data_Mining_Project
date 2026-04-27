"""Train standalone WOE/scorecard Logistic Regression baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET
from scorecard import make_scorecard_pipeline
from src.config.paths import HELOC_MODEL_READY, SCORECARD_MODELS_DIR, TABLES_DIR, TAIWAN_MODEL_READY


DATASETS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "categorical_columns": ["SEX", "EDUCATION", "MARRIAGE"],
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "categorical_columns": HELOC_CATEGORICAL_COLUMNS,
    },
}


def _positive_class_probability(estimator, X: pd.DataFrame) -> pd.Series:
    """Return probability for class 1."""

    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"Estimator does not expose positive class 1: {classes}")
    return probabilities[:, classes.index(1)]


def _contribution_rows(dataset: str, model_path: Path, estimator) -> list[dict]:
    """Extract scorecard-style feature contribution rows from a fitted pipeline."""

    woe = estimator.named_steps["woe"]
    model = estimator.named_steps["model"]
    rows: list[dict] = []
    for feature, coefficient in zip(woe.feature_names_in_, model.coef_[0]):
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "coefficient": float(coefficient),
                "absolute_coefficient": float(abs(coefficient)),
                "n_woe_bins_or_categories": int(len(woe.woe_maps_.get(feature, {}))),
                "model_path": str(model_path),
            }
        )
    return rows


def train_scorecard_for_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train one dataset's scorecard and return result/contribution tables."""

    config = DATASETS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_train = split.train.drop(columns=[config["target"]])
    y_train = split.train[config["target"]]
    evaluations = {
        "validation": (
            split.validation.drop(columns=[config["target"]]),
            split.validation[config["target"]],
            False,
        ),
        "test": (
            split.test.drop(columns=[config["target"]]),
            split.test[config["target"]],
            True,
        ),
    }

    estimator = make_scorecard_pipeline(categorical_columns=config["categorical_columns"])
    estimator.fit(X_train, y_train)

    model_dir = SCORECARD_MODELS_DIR / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "woe_scorecard_logistic_regression.joblib"
    joblib.dump(estimator, model_path)

    result_rows: list[dict] = []
    for split_name, (X_eval, y_eval, test_set_used) in evaluations.items():
        y_proba = _positive_class_probability(estimator, X_eval)
        result_rows.append(
            {
                "dataset": dataset,
                "model": "woe_scorecard_logistic_regression",
                "split": split_name,
                "train_rows": int(len(split.train)),
                "validation_rows": int(len(split.validation)),
                "test_rows_reserved": int(len(split.test)),
                "test_set_used": test_set_used,
                "fit_policy": "binning_and_woe_fit_on_train_only",
                "model_path": str(model_path),
                **binary_classification_metrics(y_eval, y_proba, threshold=0.5),
            }
        )

    return pd.DataFrame(result_rows), pd.DataFrame(_contribution_rows(dataset, model_path, estimator))


def main() -> None:
    """Train scorecard baselines for Taiwan and HELOC."""

    run_id = start_run("train_scorecard", dataset="both", tags=["phase-6", "scorecard"])
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        all_results = []
        all_contributions = []
        for dataset in DATASETS:
            results, contributions = train_scorecard_for_dataset(dataset)
            all_results.append(results)
            all_contributions.append(contributions)

        result_table = pd.concat(all_results, ignore_index=True)
        contribution_table = pd.concat(all_contributions, ignore_index=True)
        contribution_table = contribution_table.sort_values(
            ["dataset", "absolute_coefficient"],
            ascending=[True, False],
        )
        result_table.to_csv(TABLES_DIR / "scorecard_results.csv", index=False)
        contribution_table.to_csv(TABLES_DIR / "scorecard_feature_contributions.csv", index=False)
        finish_run(
            run_id,
            artifacts={
                "scorecard_results": TABLES_DIR / "scorecard_results.csv",
                "scorecard_feature_contributions": TABLES_DIR / "scorecard_feature_contributions.csv",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
