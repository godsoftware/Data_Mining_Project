"""Run enhanced model comparison with boosting and imbalance options."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from config.settings import RANDOM_SEED
from data_preprocessing import PROJECT_ROOT, TARGET_COLUMN
from evaluation import binary_classification_metrics
from model_training import (
    CATEGORICAL_COLUMNS,
    build_preprocessor,
    candidate_models,
    make_pipeline,
    save_model,
    split_features_target,
)


PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "taiwan_model_ready.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def make_smotenc_pipeline(model, X: pd.DataFrame, random_state: int):
    """Apply SMOTENC on raw training folds, then preprocess and fit the model."""

    categorical_indices = [
        X.columns.get_loc(column)
        for column in CATEGORICAL_COLUMNS
        if column in X.columns
    ]
    sampler = SMOTENC(
        categorical_features=categorical_indices,
        random_state=random_state,
    )
    return ImbalancedPipeline(
        steps=[
            ("sampler", sampler),
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def selection_score(metrics: dict[str, float]) -> float:
    """Decision score aligned with the report's model-selection logic."""

    return (
        0.25 * metrics["roc_auc"]
        + 0.20 * metrics["pr_auc"]
        + 0.20 * metrics["recall"]
        + 0.15 * metrics["f1"]
        + 0.10 * (1.0 - metrics["brier_score"])
        + 0.10 * metrics["specificity"]
    )


def main() -> None:
    """Train candidate models and save the selected final model."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split = split_features_target(df, target_column=TARGET_COLUMN, random_state=RANDOM_SEED)

    rows: list[dict] = []
    fitted_models: dict[tuple[str, str], object] = {}

    model_names = [
        "logistic_regression",
        "logistic_regression_balanced",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
    ]
    smotenc_model_names = [
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
    ]

    models = candidate_models(random_state=RANDOM_SEED)

    experiments: list[tuple[str, str]] = [
        ("no_resampling", model_name)
        for model_name in model_names
        if model_name in models
    ]
    experiments.extend(
        ("smotenc", model_name)
        for model_name in smotenc_model_names
        if model_name in models
    )

    for strategy, model_name in experiments:
        model = candidate_models(random_state=RANDOM_SEED)[model_name]
        print(f"Training {model_name} with {strategy}...")
        start = perf_counter()

        if strategy == "smotenc":
            pipeline = make_smotenc_pipeline(model, split.X_train, random_state=RANDOM_SEED)
        else:
            pipeline = make_pipeline(model, split.X_train)

        pipeline.fit(split.X_train, split.y_train)
        y_proba = pipeline.predict_proba(split.X_test)[:, 1]
        metrics = binary_classification_metrics(split.y_test, y_proba, threshold=0.5)
        metrics["model"] = model_name
        metrics["imbalance_strategy"] = strategy
        metrics["training_seconds"] = perf_counter() - start
        metrics["selection_score"] = selection_score(metrics)

        rows.append(metrics)
        fitted_models[(strategy, model_name)] = pipeline

    comparison = pd.DataFrame(rows).sort_values(
        ["selection_score", "pr_auc", "roc_auc"], ascending=False
    )
    comparison.to_csv(TABLES_DIR / "enhanced_model_comparison.csv", index=False)

    best = comparison.iloc[0]
    best_key = (best["imbalance_strategy"], best["model"])
    best_model = fitted_models[best_key]
    final_path = save_model(best_model, "final_model")
    joblib.dump(best_model, MODELS_DIR / f"{best['model']}_{best['imbalance_strategy']}.joblib")

    metadata = {
        "selected_model": best["model"],
        "imbalance_strategy": best["imbalance_strategy"],
        "selection_score": float(best["selection_score"]),
        "final_model_path": str(final_path),
        "selection_rule": (
            "0.25*ROC-AUC + 0.20*PR-AUC + 0.20*Recall + 0.15*F1 "
            "+ 0.10*(1-Brier) + 0.10*Specificity"
        ),
    }
    (TABLES_DIR / "final_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(comparison[
        [
            "model",
            "imbalance_strategy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "selection_score",
        ]
    ].to_string(index=False))
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
