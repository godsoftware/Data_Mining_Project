"""Train Phase 6 baseline models for Taiwan and HELOC."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config.settings import RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET
from scorecard import make_scorecard_pipeline
from scre_credit import make_model_pipeline
from src.config.paths import BASELINE_MODELS_DIR, HELOC_MODEL_READY, TABLES_DIR, TAIWAN_MODEL_READY


DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "categorical_columns": ["SEX", "EDUCATION", "MARRIAGE"],
        "output": TABLES_DIR / "baseline_results_taiwan.csv",
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "categorical_columns": HELOC_CATEGORICAL_COLUMNS,
        "output": TABLES_DIR / "baseline_results_heloc.csv",
    },
}


def _safe_name(name: str) -> str:
    """Return a filesystem-friendly model name."""

    return name.lower().replace(" / ", "_").replace("/", "_").replace(" ", "_").replace("-", "_")


def _positive_class_probability(estimator, X: pd.DataFrame) -> pd.Series:
    """Return probability for class 1 even when estimator class ordering differs."""

    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"Estimator does not expose positive class 1: {classes}")
    return probabilities[:, classes.index(1)]


def build_baseline_models(
    X_train: pd.DataFrame,
    categorical_columns: list[str],
    random_state: int = RANDOM_SEED,
) -> list[tuple[str, object, str]]:
    """Create the Phase 6 baseline model list."""

    return [
        (
            "majority_baseline",
            DummyClassifier(strategy="most_frequent"),
            "Most-frequent-class dummy baseline fit on the train split.",
        ),
        (
            "logistic_regression",
            make_model_pipeline(
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=random_state,
                ),
                X_train,
                categorical_columns,
            ),
            "Standard Logistic Regression with train-fitted preprocessing.",
        ),
        (
            "class_weighted_logistic_regression",
            make_model_pipeline(
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                ),
                X_train,
                categorical_columns,
            ),
            "Logistic Regression with class_weight='balanced'.",
        ),
        (
            "woe_scorecard_logistic_regression",
            make_scorecard_pipeline(categorical_columns=categorical_columns),
            "WOE / scorecard-style Logistic Regression fit only on the train split.",
        ),
        (
            "decision_tree",
            make_model_pipeline(
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=100,
                    random_state=random_state,
                ),
                X_train,
                categorical_columns,
            ),
            "Shallow Decision Tree baseline with train-fitted preprocessing.",
        ),
        (
            "random_forest",
            make_model_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=20,
                    n_jobs=-1,
                    random_state=random_state,
                ),
                X_train,
                categorical_columns,
            ),
            "Random Forest baseline without hyperparameter tuning.",
        ),
    ]


def train_baselines_for_dataset(dataset: str) -> pd.DataFrame:
    """Train baseline models for one dataset and save validation results."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    target = config["target"]
    split = stratified_train_validation_test_split(df, target_column=target)

    X_train = split.train.drop(columns=[target])
    y_train = split.train[target]
    X_validation = split.validation.drop(columns=[target])
    y_validation = split.validation[target]
    X_test = split.test.drop(columns=[target])
    y_test = split.test[target]

    models = build_baseline_models(
        X_train,
        categorical_columns=config["categorical_columns"],
        random_state=RANDOM_SEED,
    )

    dataset_model_dir = BASELINE_MODELS_DIR / dataset
    dataset_model_dir.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for model_name, estimator, description in models:
        print(f"Training {dataset} baseline: {model_name}")
        estimator.fit(X_train, y_train)
        model_path = dataset_model_dir / f"{_safe_name(model_name)}.joblib"
        joblib.dump(estimator, model_path)

        for split_name, X_eval, y_eval, test_set_used in [
            ("validation", X_validation, y_validation, False),
            ("test", X_test, y_test, True),
        ]:
            y_proba = _positive_class_probability(estimator, X_eval)
            metrics = binary_classification_metrics(y_eval, y_proba, threshold=0.5)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "split": split_name,
                    "train_rows": int(len(split.train)),
                    "validation_rows": int(len(split.validation)),
                    "test_rows_reserved": int(len(split.test)),
                    "test_set_used": test_set_used,
                    "resampling": "none",
                    "preprocessing_fit_scope": "train_split_pipeline_fit",
                    "model_path": str(model_path),
                    "description": description,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).sort_values(["split", "roc_auc", "pr_auc"], ascending=[True, False, False])
    results.to_csv(config["output"], index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    args = parser.parse_args()

    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    run_id = start_run("train_baseline_models", dataset=args.dataset, tags=["phase-6", "baselines"])
    try:
        artifacts = {}
        metrics = {}
        for dataset in datasets:
            results = train_baselines_for_dataset(dataset)
            artifacts[f"{dataset}_baseline_results"] = DATASET_CONFIGS[dataset]["output"]
            best = results.loc[results["split"] == "validation"].iloc[0]
            metrics[f"{dataset}_best_validation_roc_auc"] = float(best["roc_auc"])
            metrics[f"{dataset}_best_validation_model"] = str(best["model"])
            print(results[["dataset", "model", "roc_auc", "pr_auc", "recall", "f1"]].to_string(index=False))

        finish_run(run_id, metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
