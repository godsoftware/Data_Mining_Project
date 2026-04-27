"""Train Phase 7 black-box and monotonic-constrained model pools."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.settings import RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET
from monotonic_models import (
    HELOC_MONOTONIC_PRIORS,
    TAIWAN_MONOTONIC_PRIORS,
    monotonic_constraint_vector,
    xgboost_constraint_string,
)
from scre_credit import make_model_pipeline, make_numeric_model_pipeline
from src.config.paths import (
    BOOSTING_MODELS_DIR,
    HELOC_MODEL_READY,
    MONOTONIC_MODELS_DIR,
    TABLES_DIR,
    TAIWAN_MODEL_READY,
)

try:
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbalancedPipeline
except ImportError:  # pragma: no cover
    SMOTENC = None
    ImbalancedPipeline = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "categorical_columns": ["SEX", "EDUCATION", "MARRIAGE"],
        "monotonic_priors": TAIWAN_MONOTONIC_PRIORS,
        "output": TABLES_DIR / "model_results_taiwan.csv",
        "boosting_output": TABLES_DIR / "boosting_results_taiwan.csv",
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "categorical_columns": HELOC_CATEGORICAL_COLUMNS,
        "monotonic_priors": HELOC_MONOTONIC_PRIORS,
        "output": TABLES_DIR / "model_results_heloc.csv",
        "boosting_output": TABLES_DIR / "boosting_results_heloc.csv",
    },
}


@dataclass
class ModelSpec:
    """Phase 7 model specification."""

    name: str
    estimator: object
    family: str
    resampling: str = "none"
    monotonic: bool = False
    constraint_summary: str = ""


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    """Train-fitted imputer that preserves raw column order before SMOTENC."""

    def __init__(self, categorical_columns: list[str] | None = None) -> None:
        self.categorical_columns = categorical_columns or []

    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X).copy()
        self.columns_ = list(X_df.columns)
        categorical = set(self.categorical_columns)
        self.fill_values_: dict[str, float] = {}

        for column in self.columns_:
            series = pd.to_numeric(X_df[column], errors="coerce")
            if column in categorical:
                mode = series.dropna().mode()
                self.fill_values_[column] = float(mode.iloc[0]) if not mode.empty else 0.0
            else:
                median = series.median()
                self.fill_values_[column] = float(median) if pd.notna(median) else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        transformed = pd.DataFrame(index=X_df.index)
        for column in self.columns_:
            transformed[column] = pd.to_numeric(X_df[column], errors="coerce").fillna(
                self.fill_values_[column]
            )
        return transformed


def _safe_name(name: str) -> str:
    """Return a filesystem-friendly model name."""

    return name.lower().replace(" + ", "_").replace(" / ", "_").replace("/", "_").replace(" ", "_").replace("-", "_")


def _xgboost(random_state: int = RANDOM_SEED, monotone_constraints: str | None = None):
    if XGBClassifier is None:
        return None
    params = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    return XGBClassifier(**params)


def _lightgbm(random_state: int = RANDOM_SEED, class_weight=None, monotone_constraints: list[int] | None = None):
    if LGBMClassifier is None:
        return None
    params = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if class_weight is not None:
        params["class_weight"] = class_weight
    if monotone_constraints is not None:
        params["monotone_constraints"] = monotone_constraints
    return LGBMClassifier(**params)


def _catboost(random_state: int = RANDOM_SEED):
    if CatBoostClassifier is None:
        return None
    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.04,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
    )


def make_smotenc_xgboost_pipeline(
    X_train: pd.DataFrame,
    categorical_columns: list[str],
    random_state: int = RANDOM_SEED,
):
    """Create a train-fitted imputer + SMOTENC + XGBoost pipeline."""

    if SMOTENC is None or ImbalancedPipeline is None or XGBClassifier is None:
        return None

    categorical_indices = [
        X_train.columns.get_loc(column)
        for column in categorical_columns
        if column in X_train.columns
    ]
    if not categorical_indices:
        return None

    return ImbalancedPipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(categorical_columns=categorical_columns)),
            ("sampler", SMOTENC(categorical_features=categorical_indices, random_state=random_state)),
            ("model", _xgboost(random_state=random_state)),
        ]
    )


def build_phase7_models(
    dataset: str,
    X_train: pd.DataFrame,
    categorical_columns: list[str],
    monotonic_priors: dict[str, int],
    random_state: int = RANDOM_SEED,
) -> list[ModelSpec]:
    """Create all Phase 7 black-box and constrained model specs."""

    specs: list[ModelSpec] = []
    feature_names = list(X_train.columns)
    xgb_constraints = xgboost_constraint_string(feature_names, monotonic_priors)
    lgbm_constraints = monotonic_constraint_vector(feature_names, monotonic_priors)
    nonzero_constraints = {
        feature: monotonic_priors[feature]
        for feature in feature_names
        if monotonic_priors.get(feature, 0) != 0
    }

    xgb = _xgboost(random_state=random_state)
    if xgb is not None:
        specs.append(
            ModelSpec(
                name="xgboost",
                estimator=make_model_pipeline(xgb, X_train, categorical_columns),
                family="boosting",
            )
        )

    lgbm = _lightgbm(random_state=random_state)
    if lgbm is not None:
        specs.append(
            ModelSpec(
                name="lightgbm",
                estimator=make_model_pipeline(lgbm, X_train, categorical_columns),
                family="boosting",
            )
        )

    cat = _catboost(random_state=random_state)
    if cat is not None:
        specs.append(
            ModelSpec(
                name="catboost",
                estimator=make_model_pipeline(cat, X_train, categorical_columns),
                family="boosting",
            )
        )

    smotenc_xgb = make_smotenc_xgboost_pipeline(X_train, categorical_columns, random_state=random_state)
    if smotenc_xgb is not None:
        specs.append(
            ModelSpec(
                name="smotenc_xgboost",
                estimator=smotenc_xgb,
                family="boosting",
                resampling="SMOTENC train split only",
            )
        )

    weighted_lgbm = _lightgbm(random_state=random_state, class_weight="balanced")
    if weighted_lgbm is not None:
        specs.append(
            ModelSpec(
                name="class_weighted_lightgbm",
                estimator=make_model_pipeline(weighted_lgbm, X_train, categorical_columns),
                family="boosting",
            )
        )

    monotonic_xgb = _xgboost(random_state=random_state, monotone_constraints=xgb_constraints)
    if monotonic_xgb is not None:
        specs.append(
            ModelSpec(
                name="monotonic_xgboost",
                estimator=make_numeric_model_pipeline(monotonic_xgb),
                family="monotonic",
                monotonic=True,
                constraint_summary=str(nonzero_constraints),
            )
        )

    monotonic_lgbm = _lightgbm(random_state=random_state, monotone_constraints=lgbm_constraints)
    if monotonic_lgbm is not None:
        specs.append(
            ModelSpec(
                name="monotonic_lightgbm",
                estimator=make_numeric_model_pipeline(monotonic_lgbm),
                family="monotonic",
                monotonic=True,
                constraint_summary=str(nonzero_constraints),
            )
        )

    expected = {
        "xgboost",
        "lightgbm",
        "catboost",
        "smotenc_xgboost",
        "class_weighted_lightgbm",
        "monotonic_xgboost",
        "monotonic_lightgbm",
    }
    missing = expected - {spec.name for spec in specs}
    if missing:
        raise ImportError(f"Missing Phase 7 model dependencies for: {sorted(missing)}")

    return specs


def _positive_class_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"Estimator does not expose positive class 1: {classes}")
    return probabilities[:, classes.index(1)]


def train_phase7_for_dataset(dataset: str) -> pd.DataFrame:
    """Train all Phase 7 models for one dataset and save validation results."""

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

    specs = build_phase7_models(
        dataset,
        X_train,
        categorical_columns=config["categorical_columns"],
        monotonic_priors=config["monotonic_priors"],
        random_state=RANDOM_SEED,
    )

    rows: list[dict] = []
    for spec in specs:
        print(f"Training {dataset} Phase 7 model: {spec.name}")
        spec.estimator.fit(X_train, y_train)

        model_root = MONOTONIC_MODELS_DIR if spec.family == "monotonic" else BOOSTING_MODELS_DIR
        model_dir = model_root / dataset
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{_safe_name(spec.name)}.joblib"
        joblib.dump(spec.estimator, model_path)

        for split_name, X_eval, y_eval, test_set_used in [
            ("validation", X_validation, y_validation, False),
            ("test", X_test, y_test, True),
        ]:
            y_proba = _positive_class_probability(spec.estimator, X_eval)
            metrics = binary_classification_metrics(y_eval, y_proba, threshold=0.5)
            rows.append(
                {
                    "dataset": dataset,
                    "model": spec.name,
                    "model_family": spec.family,
                    "split": split_name,
                    "train_rows": int(len(split.train)),
                    "validation_rows": int(len(split.validation)),
                    "test_rows_reserved": int(len(split.test)),
                    "test_set_used": test_set_used,
                    "resampling": spec.resampling,
                    "monotonic_constraints": spec.monotonic,
                    "constraint_summary": spec.constraint_summary,
                    "model_path": str(model_path),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).sort_values(["split", "roc_auc", "pr_auc"], ascending=[True, False, False])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config["output"], index=False)
    results.to_csv(config["boosting_output"], index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    args = parser.parse_args()

    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    run_id = start_run("train_blackbox_constrained_models", dataset=args.dataset, tags=["phase-7", "model-pool"])
    try:
        artifacts = {}
        metrics = {}
        for dataset in datasets:
            results = train_phase7_for_dataset(dataset)
            artifacts[f"{dataset}_model_results"] = DATASET_CONFIGS[dataset]["output"]
            artifacts[f"{dataset}_boosting_results"] = DATASET_CONFIGS[dataset]["boosting_output"]
            best = results.loc[results["split"] == "validation"].iloc[0]
            metrics[f"{dataset}_best_validation_roc_auc"] = float(best["roc_auc"])
            metrics[f"{dataset}_best_validation_model"] = str(best["model"])
            print(
                results[
                    ["dataset", "model", "model_family", "roc_auc", "pr_auc", "recall", "f1"]
                ].to_string(index=False)
            )

        finish_run(run_id, metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
