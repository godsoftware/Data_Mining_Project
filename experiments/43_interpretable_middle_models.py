"""Interpretable middle-model experiment: EBM, monotonic boosting, scorecard.

This experiment keeps the frozen train/validation/test split. The frozen train
split is further separated into model_train and calibration_val, so model
fitting, probability calibration, validation-only policy selection, and locked
test evaluation are kept separate.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from interpret.glassbox import ExplainableBoostingClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from data.split_leakage import stratified_train_validation_test_split  # noqa: E402
from data_preprocessing import TARGET_COLUMN  # noqa: E402
from evaluation import expected_calibration_error  # noqa: E402
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET  # noqa: E402
from scorecard import make_scorecard_pipeline  # noqa: E402
from src.config.paths import HELOC_MODEL_READY, OUTPUTS_DIR, TAIWAN_MODEL_READY  # noqa: E402


RANDOM_SEED = 42
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COST = 0.5
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)
T_LOW_GRID = np.round(np.arange(0.03, 0.4001, 0.01), 2)
T_HIGH_GRID = np.round(np.arange(0.10, 0.8001, 0.01), 2)
OUTPUT_DIR = OUTPUTS_DIR / "final_attempt" / "interpretable_models"
TAIWAN_CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]

INCREASING_RISK_FEATURES = {
    "delay_count",
    "severe_delay_count",
    "max_delay",
    "avg_delay",
    "recent_delay",
    "utilization_proxy",
}
DECREASING_RISK_FEATURES = {
    "payment_to_bill_ratio",
    "recent_payment_intensity",
}

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset metadata used by the experiment."""

    name: str
    path: Path
    target: str
    categorical_columns: list[str]
    role: str


@dataclass(frozen=True)
class CandidateSpec:
    """One interpretable model candidate."""

    model_family: str
    config_name: str
    display_name: str
    params: dict[str, Any]


@dataclass
class FittedCandidate:
    """Fitted model artifact with a common prediction interface."""

    spec: CandidateSpec
    estimator: Any
    preprocessor: ColumnTransformer | None
    raw_feature_names: list[str]
    transformed_feature_names: list[str]
    raw_constraints: dict[str, int]
    transformed_constraints: list[int]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for raw input rows."""

        if self.preprocessor is not None:
            X_model = self.preprocessor.transform(X)
        else:
            X_model = X
        proba = self.estimator.predict_proba(X_model)
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 not in classes:
            return np.asarray(proba)[:, -1]
        return np.asarray(proba)[:, classes.index(1)]


@dataclass
class ProbabilityCalibrator:
    """One-dimensional probability calibrator fit on calibration_val only."""

    calibration_type: str
    model: Any | None = None

    def fit(self, probability: np.ndarray, y_true: pd.Series | np.ndarray) -> "ProbabilityCalibrator":
        """Fit the requested probability calibrator."""

        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y_true, dtype=int)
        if self.calibration_type == "uncalibrated":
            self.model = None
        elif self.calibration_type == "sigmoid":
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            model = LogisticRegression(max_iter=1000, solver="lbfgs")
            model.fit(logits, y)
            self.model = model
        elif self.calibration_type == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(probability, y)
            self.model = model
        else:
            raise ValueError(f"Unknown calibration_type: {self.calibration_type}")
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""

        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.calibration_type == "uncalibrated":
            return probability
        if self.calibration_type == "sigmoid":
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            return np.clip(self.model.predict_proba(logits)[:, 1], 0.0, 1.0)
        if self.calibration_type == "isotonic":
            return np.clip(self.model.predict(probability), 0.0, 1.0)
        raise ValueError(f"Unknown calibration_type: {self.calibration_type}")


def setup_logging() -> logging.Logger:
    """Create console and file logging."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("interpretable_middle_models")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "interpretable_models.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder across sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical_columns: list[str], X: pd.DataFrame) -> ColumnTransformer:
    """Build a train-only preprocessing transformer for monotonic boosting."""

    categorical = [column for column in categorical_columns if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def transformed_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names from a fitted ColumnTransformer."""

    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return [name.replace("numeric__", "").replace("categorical__", "") for name in names]


def raw_monotonic_constraints(columns: list[str]) -> dict[str, int]:
    """Return governance monotonic constraints for raw feature columns."""

    constraints: dict[str, int] = {}
    for column in columns:
        if column in INCREASING_RISK_FEATURES:
            constraints[column] = 1
        elif column in DECREASING_RISK_FEATURES:
            constraints[column] = -1
        else:
            constraints[column] = 0
    return constraints


def transformed_monotonic_constraints(
    raw_constraints: dict[str, int],
    preprocessor: ColumnTransformer,
) -> list[int]:
    """Map raw constraints to transformed feature order."""

    constraints: list[int] = []
    for name in preprocessor.get_feature_names_out():
        simple = str(name).replace("numeric__", "").replace("categorical__", "")
        if str(name).startswith("numeric__"):
            constraints.append(int(raw_constraints.get(simple, 0)))
        else:
            constraints.append(0)
    return constraints


def dataset_specs() -> list[DatasetSpec]:
    """Return Taiwan and HELOC dataset specs."""

    return [
        DatasetSpec("taiwan", TAIWAN_MODEL_READY, TARGET_COLUMN, TAIWAN_CATEGORICAL_COLUMNS, "primary"),
        DatasetSpec("heloc", HELOC_MODEL_READY, HELOC_TARGET, HELOC_CATEGORICAL_COLUMNS, "external_validation"),
    ]


def split_model_train_calibration(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split frozen train into model_train and calibration_val."""

    X_model, X_cal, y_model, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train,
    )
    return X_model, X_cal, y_model, y_cal


def safe_auc(y_true: pd.Series | np.ndarray, probability: np.ndarray, kind: str) -> float:
    """Compute ROC-AUC or PR-AUC with a NaN fallback for degenerate inputs."""

    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y, probability))
    if kind == "pr":
        return float(average_precision_score(y, probability))
    raise ValueError(kind)


def binary_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict[str, float | int]:
    """Compute binary decision metrics for a threshold."""

    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_precision": float(precision_score(y, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, pred, zero_division=0)),
        f"{prefix}_specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        f"{prefix}_f1": float(f1_score(y, pred, zero_division=0)),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
        f"{prefix}_expected_cost": float(FN_COST * fn + FP_COST * fp),
    }


def probability_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Compute threshold-independent probability metrics."""

    y = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    return {
        f"{prefix}_roc_auc": safe_auc(y, proba, "roc"),
        f"{prefix}_pr_auc": safe_auc(y, proba, "pr"),
        f"{prefix}_brier": float(brier_score_loss(y, proba)),
        f"{prefix}_ece": float(expected_calibration_error(y, proba, n_bins=10)),
    }


def select_binary_policy(
    y_true: pd.Series,
    probability: np.ndarray,
    policy_type: str,
) -> dict[str, Any] | None:
    """Select a binary threshold using validation data only."""

    if policy_type == "threshold_0_50":
        threshold = 0.5
        return {"threshold": threshold, **binary_metrics(y_true, probability, threshold, "validation")}

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        metrics = binary_metrics(y_true, probability, float(threshold), "validation")
        if policy_type == "precision_ge_0_40" and metrics["validation_precision"] < 0.40:
            continue
        rows.append({"threshold": float(threshold), **metrics})

    if not rows:
        return None
    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["validation_expected_cost", "validation_f1", "threshold"],
        ascending=[True, False, False],
    )
    return table.iloc[0].to_dict()


def manual_review_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    t_low: float,
    t_high: float,
    prefix: str,
) -> dict[str, float | int]:
    """Compute three-way manual-review policy metrics."""

    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(probability, dtype=float)
    high = proba >= t_high
    manual = (proba >= t_low) & (proba < t_high)
    low = proba < t_low

    high_defaults = int(((y == 1) & high).sum())
    high_nondefaults = int(((y == 0) & high).sum())
    low_defaults = int(((y == 1) & low).sum())
    low_nondefaults = int(((y == 0) & low).sum())
    manual_defaults = int(((y == 1) & manual).sum())
    manual_nondefaults = int(((y == 0) & manual).sum())

    n = len(y)
    default_total = int((y == 1).sum())
    nondefault_total = int((y == 0).sum())
    manual_count = int(manual.sum())
    auto_fp = high_nondefaults
    auto_fn = low_defaults
    auto_tp = high_defaults
    auto_tn = low_nondefaults
    review_adjusted_cost = float(FN_COST * auto_fn + FP_COST * auto_fp + REVIEW_COST * manual_count)

    return {
        f"{prefix}_t_low": float(t_low),
        f"{prefix}_t_high": float(t_high),
        f"{prefix}_manual_review_count": manual_count,
        f"{prefix}_manual_review_rate": float(manual_count / n) if n else float("nan"),
        f"{prefix}_auto_decision_rate": float(1.0 - manual_count / n) if n else float("nan"),
        f"{prefix}_high_risk_precision": float(high_defaults / (high_defaults + high_nondefaults))
        if (high_defaults + high_nondefaults)
        else 0.0,
        f"{prefix}_high_risk_recall": float(high_defaults / default_total) if default_total else float("nan"),
        f"{prefix}_low_risk_default_rate": float(low_defaults / (low_defaults + low_nondefaults))
        if (low_defaults + low_nondefaults)
        else float("nan"),
        f"{prefix}_precision": float(high_defaults / (high_defaults + high_nondefaults))
        if (high_defaults + high_nondefaults)
        else 0.0,
        f"{prefix}_recall": float(high_defaults / default_total) if default_total else float("nan"),
        f"{prefix}_specificity": float((low_nondefaults + manual_nondefaults) / nondefault_total)
        if nondefault_total
        else float("nan"),
        f"{prefix}_f1": float(2 * high_defaults / (2 * high_defaults + high_nondefaults + low_defaults + manual_defaults))
        if (2 * high_defaults + high_nondefaults + low_defaults + manual_defaults)
        else 0.0,
        f"{prefix}_tn": auto_tn,
        f"{prefix}_fp": auto_fp,
        f"{prefix}_fn": auto_fn,
        f"{prefix}_tp": auto_tp,
        f"{prefix}_expected_cost": float(FN_COST * auto_fn + FP_COST * auto_fp),
        f"{prefix}_manual_review_adjusted_cost": review_adjusted_cost,
    }


def select_manual_review_policy(
    y_true: pd.Series,
    probability: np.ndarray,
    policy_type: str,
) -> dict[str, Any] | None:
    """Select a manual-review band on validation data only."""

    rows: list[dict[str, Any]] = []
    for t_low in T_LOW_GRID:
        eligible_highs = T_HIGH_GRID[T_HIGH_GRID > t_low]
        for t_high in eligible_highs:
            metrics = manual_review_metrics(y_true, probability, float(t_low), float(t_high), "validation")
            if policy_type == "manual_review_mr_le_0_30" and metrics["validation_manual_review_rate"] > 0.30:
                continue
            if (
                policy_type == "manual_review_high_precision_ge_0_45"
                and metrics["validation_high_risk_precision"] < 0.45
            ):
                continue
            rows.append(metrics)

    if not rows:
        return None
    table = pd.DataFrame(rows)
    table = table.sort_values(
        [
            "validation_manual_review_adjusted_cost",
            "validation_high_risk_precision",
            "validation_manual_review_rate",
        ],
        ascending=[True, False, True],
    )
    best = table.iloc[0].to_dict()
    best["threshold"] = np.nan
    return best


def evaluate_policy(
    y_true: pd.Series,
    probability: np.ndarray,
    policy_type: str,
    split_prefix: str,
    threshold: float | None = None,
    t_low: float | None = None,
    t_high: float | None = None,
) -> dict[str, Any]:
    """Evaluate a locked policy on validation or test data."""

    if policy_type.startswith("manual_review"):
        if t_low is None or t_high is None:
            raise ValueError("Manual-review policies require t_low and t_high")
        return manual_review_metrics(y_true, probability, t_low, t_high, split_prefix)
    if threshold is None:
        raise ValueError("Binary policies require threshold")
    return binary_metrics(y_true, probability, threshold, split_prefix)


def ebm_specs() -> list[CandidateSpec]:
    """Create the exact EBM grid requested in the prompt."""

    specs: list[CandidateSpec] = []
    grid = itertools.product(
        [0, 5, 10],
        [0.005, 0.01, 0.02],
        [128, 256],
        [8, 16],
        [0, 4],
        [2, 5, 10],
    )
    for interactions, learning_rate, max_bins, outer_bags, inner_bags, min_samples_leaf in grid:
        name = (
            f"ebm_i{interactions}_lr{learning_rate}_bins{max_bins}_"
            f"ob{outer_bags}_ib{inner_bags}_leaf{min_samples_leaf}"
        )
        specs.append(
            CandidateSpec(
                model_family="EBM",
                config_name=name,
                display_name=f"EBM interactions={interactions}",
                params={
                    "interactions": interactions,
                    "learning_rate": learning_rate,
                    "max_bins": max_bins,
                    "outer_bags": outer_bags,
                    "inner_bags": inner_bags,
                    "min_samples_leaf": min_samples_leaf,
                },
            )
        )
    return specs


def monotonic_specs() -> list[CandidateSpec]:
    """Return compact monotonic boosting candidates."""

    return [
        CandidateSpec(
            "Monotonic XGBoost",
            "monotonic_xgboost_base",
            "Monotonic XGBoost base",
            {
                "n_estimators": 350,
                "learning_rate": 0.035,
                "max_depth": 4,
                "min_child_weight": 4,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 5.0,
                "reg_alpha": 0.05,
            },
        ),
        CandidateSpec(
            "Monotonic XGBoost",
            "monotonic_xgboost_conservative",
            "Monotonic XGBoost conservative",
            {
                "n_estimators": 500,
                "learning_rate": 0.02,
                "max_depth": 3,
                "min_child_weight": 8,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_lambda": 10.0,
                "reg_alpha": 0.10,
            },
        ),
        CandidateSpec(
            "Monotonic LightGBM",
            "monotonic_lightgbm_base",
            "Monotonic LightGBM base",
            {
                "n_estimators": 350,
                "learning_rate": 0.035,
                "num_leaves": 24,
                "max_depth": 5,
                "min_child_samples": 25,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 5.0,
                "reg_alpha": 0.05,
            },
        ),
        CandidateSpec(
            "Monotonic LightGBM",
            "monotonic_lightgbm_conservative",
            "Monotonic LightGBM conservative",
            {
                "n_estimators": 500,
                "learning_rate": 0.02,
                "num_leaves": 18,
                "max_depth": 4,
                "min_child_samples": 40,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_lambda": 10.0,
                "reg_alpha": 0.10,
            },
        ),
    ]


def scorecard_specs() -> list[CandidateSpec]:
    """Return scorecard refined candidates."""

    return [
        CandidateSpec(
            "Scorecard",
            "scorecard_woe_balanced",
            "WOE scorecard balanced",
            {"class_weight": "balanced"},
        ),
        CandidateSpec(
            "Scorecard",
            "scorecard_woe_unweighted",
            "WOE scorecard unweighted",
            {"class_weight": None},
        ),
    ]


def fit_candidate(
    spec: CandidateSpec,
    dataset: DatasetSpec,
    X_model: pd.DataFrame,
    y_model: pd.Series,
) -> FittedCandidate:
    """Fit one candidate on model_train only."""

    raw_features = list(X_model.columns)
    raw_constraints = raw_monotonic_constraints(raw_features)
    if spec.model_family == "EBM":
        feature_types = [
            "nominal" if column in dataset.categorical_columns else "continuous" for column in raw_features
        ]
        estimator = ExplainableBoostingClassifier(
            feature_names=raw_features,
            feature_types=feature_types,
            interactions=spec.params["interactions"],
            learning_rate=spec.params["learning_rate"],
            max_bins=spec.params["max_bins"],
            outer_bags=spec.params["outer_bags"],
            inner_bags=spec.params["inner_bags"],
            min_samples_leaf=spec.params["min_samples_leaf"],
            max_rounds=1500,
            early_stopping_rounds=50,
            validation_size=0.15,
            n_jobs=-2,
            random_state=RANDOM_SEED,
        )
        estimator.fit(X_model, y_model)
        return FittedCandidate(spec, estimator, None, raw_features, raw_features, raw_constraints, [])

    if spec.model_family == "Scorecard":
        estimator = make_scorecard_pipeline(
            categorical_columns=dataset.categorical_columns,
            class_weight=spec.params["class_weight"],
        )
        estimator.fit(X_model, y_model)
        return FittedCandidate(spec, estimator, None, raw_features, raw_features, raw_constraints, [])

    preprocessor = build_preprocessor(dataset.categorical_columns, X_model)
    X_fit = preprocessor.fit_transform(X_model)
    names = transformed_names(preprocessor)
    constraints = transformed_monotonic_constraints(raw_constraints, preprocessor)
    if spec.model_family == "Monotonic XGBoost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            monotone_constraints=tuple(constraints),
            **spec.params,
        )
    elif spec.model_family == "Monotonic LightGBM":
        estimator = LGBMClassifier(
            objective="binary",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbosity=-1,
            monotone_constraints=constraints,
            **spec.params,
        )
    else:
        raise ValueError(f"Unknown model_family: {spec.model_family}")

    estimator.fit(X_fit, y_model)
    return FittedCandidate(spec, estimator, preprocessor, raw_features, names, raw_constraints, constraints)


def term_rows(dataset: str, artifact: FittedCandidate) -> list[dict[str, Any]]:
    """Extract global term/importances or reason-code rows."""

    spec = artifact.spec
    rows: list[dict[str, Any]] = []
    if spec.model_family == "EBM":
        importances = artifact.estimator.term_importances()
        term_names = list(artifact.estimator.term_names_)
        for rank, (term, importance) in enumerate(
            sorted(zip(term_names, importances), key=lambda item: abs(float(item[1])), reverse=True),
            start=1,
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "model_family": spec.model_family,
                    "config_name": spec.config_name,
                    "rank": rank,
                    "term": term,
                    "importance": float(importance),
                    "reason_code": f"EBM term contribution: {term}",
                }
            )
    elif spec.model_family == "Scorecard":
        woe = artifact.estimator.named_steps["woe"]
        model = artifact.estimator.named_steps["model"]
        for rank, (feature, coefficient) in enumerate(
            sorted(zip(woe.feature_names_in_, model.coef_[0]), key=lambda item: abs(float(item[1])), reverse=True),
            start=1,
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "model_family": spec.model_family,
                    "config_name": spec.config_name,
                    "rank": rank,
                    "term": feature,
                    "importance": float(abs(coefficient)),
                    "coefficient": float(coefficient),
                    "reason_code": f"WOE scorecard coefficient for {feature}",
                }
            )
    else:
        importances = getattr(artifact.estimator, "feature_importances_", np.zeros(len(artifact.transformed_feature_names)))
        for rank, (feature, importance) in enumerate(
            sorted(
                zip(artifact.transformed_feature_names, importances),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            ),
            start=1,
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "model_family": spec.model_family,
                    "config_name": spec.config_name,
                    "rank": rank,
                    "term": feature,
                    "importance": float(importance),
                    "reason_code": f"Monotonic boosted feature importance: {feature}",
                }
            )
    return rows


def interpretability_metrics(artifact: FittedCandidate, X_reference: pd.DataFrame) -> dict[str, Any]:
    """Compute lightweight interpretability governance metrics."""

    spec = artifact.spec
    if spec.model_family == "EBM":
        number_of_terms = int(len(artifact.estimator.term_names_))
        interactions = int(spec.params.get("interactions", 0))
        clarity = max(0.50, 0.95 - 0.025 * interactions)
        simplicity = max(0.45, 0.90 - 0.02 * interactions)
        monotonic_violations = np.nan
    elif spec.model_family == "Scorecard":
        number_of_terms = int(len(artifact.raw_feature_names))
        clarity = 1.0
        simplicity = 1.0
        monotonic_violations = np.nan
    else:
        number_of_terms = int(len(artifact.transformed_feature_names))
        clarity = 0.82
        simplicity = 0.72
        monotonic_violations = monotonicity_violation_rate(artifact, X_reference)

    return {
        "number_of_terms": number_of_terms,
        "top_terms_stability": np.nan,
        "reason_code_simplicity": float(simplicity),
        "monotonicity_violations": monotonic_violations,
        "explanation_clarity_score": float(clarity),
    }


def monotonicity_violation_rate(artifact: FittedCandidate, X_reference: pd.DataFrame) -> float:
    """Probe monotonic constraints with local perturbations on validation rows."""

    constrained = {k: v for k, v in artifact.raw_constraints.items() if v != 0 and k in X_reference.columns}
    if not constrained:
        return float("nan")
    sample = X_reference.sample(n=min(500, len(X_reference)), random_state=RANDOM_SEED).copy()
    base = artifact.predict_proba(sample)
    violations = 0
    checks = 0
    for feature, direction in constrained.items():
        values = pd.to_numeric(sample[feature], errors="coerce")
        step = float(np.nanstd(values))
        if not np.isfinite(step) or step <= 0:
            step = 1.0
        perturbed = sample.copy()
        perturbed[feature] = values + step
        changed = artifact.predict_proba(perturbed)
        if direction == 1:
            violations += int((changed + 1e-9 < base).sum())
        elif direction == -1:
            violations += int((changed - 1e-9 > base).sum())
        checks += len(sample)
    return float(violations / checks) if checks else float("nan")


def eligibility(dataset: str, row: dict[str, Any]) -> bool:
    """Return basic eligibility for validation selection."""

    if dataset == "taiwan":
        return bool(
            row.get("validation_precision", 0) >= 0.40
            and row.get("validation_recall", 0) >= 0.55
            and row.get("validation_specificity", 0) >= 0.60
            and row.get("validation_pr_auc", 0) >= 0.54
            and row.get("validation_roc_auc", 0) >= 0.77
        )
    return bool(
        row.get("validation_precision", 0) >= 0.55
        and row.get("validation_recall", 0) >= 0.80
        and row.get("validation_specificity", 0) >= 0.20
        and row.get("validation_pr_auc", 0) >= 0.78
        and row.get("validation_roc_auc", 0) >= 0.78
    )


def selection_cost(row: dict[str, Any]) -> float:
    """Return the correct validation cost field for binary vs manual-review policies."""

    if str(row["policy_type"]).startswith("manual_review"):
        return float(row.get("validation_manual_review_adjusted_cost", np.inf))
    return float(row.get("validation_expected_cost", np.inf))


def select_policy_rows(
    dataset: DatasetSpec,
    artifact: FittedCandidate,
    calibration_type: str,
    calibrator: ProbabilityCalibrator,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> list[dict[str, Any]]:
    """Create validation-only policy rows for one calibrated candidate."""

    raw_probability = artifact.predict_proba(X_validation)
    probability = calibrator.transform(raw_probability)
    prob_metrics = probability_metrics(y_validation, probability, "validation")
    policies = [
        "threshold_0_50",
        "precision_ge_0_40",
        "manual_review_mr_le_0_30",
        "manual_review_high_precision_ge_0_45",
    ]
    rows: list[dict[str, Any]] = []
    interp = interpretability_metrics(artifact, X_validation)
    for policy_type in policies:
        if policy_type.startswith("manual_review"):
            selected = select_manual_review_policy(y_validation, probability, policy_type)
        else:
            selected = select_binary_policy(y_validation, probability, policy_type)
        if selected is None:
            continue
        row: dict[str, Any] = {
            "dataset": dataset.name,
            "dataset_role": dataset.role,
            "model_family": artifact.spec.model_family,
            "model": artifact.spec.display_name,
            "config_name": artifact.spec.config_name,
            "policy_type": policy_type,
            "calibration_type": calibration_type,
            "threshold": float(selected.get("threshold", np.nan))
            if selected.get("threshold", np.nan) == selected.get("threshold", np.nan)
            else np.nan,
            "t_low": float(selected.get("validation_t_low", np.nan)),
            "t_high": float(selected.get("validation_t_high", np.nan)),
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
            "review_cost": REVIEW_COST,
            "selected_on": "validation_only",
            "test_set_used_for_selection": False,
            **prob_metrics,
            **selected,
            **interp,
        }
        row["validation_selection_cost"] = selection_cost(row)
        row["eligible_basic"] = eligibility(dataset.name, row)
        row["interpretability_rank_score"] = (
            0.45 * float(row["explanation_clarity_score"])
            + 0.35 * float(row["reason_code_simplicity"])
            + 0.20 * (1.0 if pd.isna(row["monotonicity_violations"]) else 1.0 - float(row["monotonicity_violations"]))
        )
        rows.append(row)
    return rows


def evaluate_locked_row(
    row: pd.Series,
    artifact: FittedCandidate,
    calibrator: ProbabilityCalibrator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate one validation-selected row on held-out test data."""

    probability = calibrator.transform(artifact.predict_proba(X_test))
    output: dict[str, Any] = {
        "dataset": row["dataset"],
        "model_family": row["model_family"],
        "model": row["model"],
        "config_name": row["config_name"],
        "policy_type": row["policy_type"],
        "calibration_type": row["calibration_type"],
        "threshold": row.get("threshold", np.nan),
        "t_low": row.get("t_low", np.nan),
        "t_high": row.get("t_high", np.nan),
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "review_cost": REVIEW_COST,
        "selected_on": "validation_only",
        "test_set_used_for_selection": False,
    }
    output.update(probability_metrics(y_test, probability, "test"))
    if str(row["policy_type"]).startswith("manual_review"):
        output.update(
            evaluate_policy(
                y_test,
                probability,
                str(row["policy_type"]),
                "test",
                t_low=float(row["t_low"]),
                t_high=float(row["t_high"]),
            )
        )
    else:
        output.update(
            evaluate_policy(
                y_test,
                probability,
                str(row["policy_type"]),
                "test",
                threshold=float(row["threshold"]),
            )
        )
    output["test_selection_note"] = "locked validation-selected configuration; no test ranking"
    return output


def choose_locked_configs(validation_table: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Choose locked rows using validation evidence only."""

    table = validation_table.copy()
    table["eligible_sort"] = table["eligible_basic"].astype(int)
    table = table.sort_values(
        [
            "eligible_sort",
            "validation_selection_cost",
            "validation_pr_auc",
            "interpretability_rank_score",
            "validation_ece",
        ],
        ascending=[False, True, False, False, True],
    )
    dedup = table.drop_duplicates(subset=["dataset", "model_family", "config_name"], keep="first")
    return dedup.head(n).copy()


def build_candidates(args: argparse.Namespace) -> list[CandidateSpec]:
    """Build candidates, optionally limiting EBM configs for debugging."""

    ebm = ebm_specs()
    if args.max_ebm_configs is not None:
        ebm = ebm[: int(args.max_ebm_configs)]
    return ebm + monotonic_specs() + scorecard_specs()


def run_dataset(dataset: DatasetSpec, args: argparse.Namespace, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all interpretable candidates for one dataset."""

    df = pd.read_csv(dataset.path)
    split = stratified_train_validation_test_split(df, target_column=dataset.target)
    X_train = split.train.drop(columns=[dataset.target])
    y_train = split.train[dataset.target].astype(int)
    X_validation = split.validation.drop(columns=[dataset.target])
    y_validation = split.validation[dataset.target].astype(int)
    X_test = split.test.drop(columns=[dataset.target])
    y_test = split.test[dataset.target].astype(int)
    X_model, X_cal, y_model, y_cal = split_model_train_calibration(X_train, y_train)

    candidates = build_candidates(args)
    logger.info(
        "%s: model_train=%s calibration=%s validation=%s test=%s candidates=%s",
        dataset.name,
        len(X_model),
        len(X_cal),
        len(X_validation),
        len(X_test),
        len(candidates),
    )

    validation_rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, str], FittedCandidate] = {}
    calibrators: dict[tuple[str, str, str], ProbabilityCalibrator] = {}
    term_output_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, spec in enumerate(candidates, start=1):
        start = time.time()
        logger.info("%s: fitting %s/%s %s", dataset.name, index, len(candidates), spec.config_name)
        try:
            artifact = fit_candidate(spec, dataset, X_model, y_model)
            fitted[(spec.model_family, spec.config_name)] = artifact
            raw_cal_prob = artifact.predict_proba(X_cal)
            for calibration_type in ["uncalibrated", "sigmoid", "isotonic"]:
                calibrator = ProbabilityCalibrator(calibration_type).fit(raw_cal_prob, y_cal)
                calibrators[(spec.model_family, spec.config_name, calibration_type)] = calibrator
                validation_rows.extend(
                    select_policy_rows(dataset, artifact, calibration_type, calibrator, X_validation, y_validation)
                )

            if spec.model_family in {"Monotonic XGBoost", "Monotonic LightGBM"}:
                for feature, constraint in artifact.raw_constraints.items():
                    constraint_rows.append(
                        {
                            "dataset": dataset.name,
                            "model_family": spec.model_family,
                            "config_name": spec.config_name,
                            "feature": feature,
                            "constraint": constraint,
                            "constraint_direction": {1: "increasing_risk", -1: "decreasing_risk", 0: "none"}[
                                constraint
                            ],
                            "reason": "Prompt-defined governance constraint" if constraint else "No reliable monotonic prior",
                        }
                    )
            logger.info("%s: finished %s in %.1fs", dataset.name, spec.config_name, time.time() - start)
        except Exception as exc:  # pragma: no cover - failures are reported as data rows
            logger.exception("%s: failed %s: %s", dataset.name, spec.config_name, exc)
            failures.append(
                {
                    "dataset": dataset.name,
                    "model_family": spec.model_family,
                    "config_name": spec.config_name,
                    "error": str(exc),
                }
            )

    validation_table = pd.DataFrame(validation_rows)
    if validation_table.empty:
        raise RuntimeError(f"No validation rows were produced for {dataset.name}")
    validation_table = validation_table.sort_values(
        [
            "dataset",
            "eligible_basic",
            "validation_selection_cost",
            "validation_pr_auc",
            "interpretability_rank_score",
        ],
        ascending=[True, False, True, False, False],
    ).reset_index(drop=True)
    validation_table["rank_validation"] = np.arange(1, len(validation_table) + 1)
    locked_rows = choose_locked_configs(validation_table, n=3)

    for family in ["EBM", "Scorecard"]:
        family_rows = validation_table[validation_table["model_family"] == family].copy()
        if family_rows.empty:
            continue
        family_best = family_rows.sort_values(
            ["eligible_basic", "validation_selection_cost", "validation_pr_auc"],
            ascending=[False, True, False],
        ).iloc[0]
        family_key = (family_best["model_family"], family_best["config_name"])
        if family_key in fitted:
            term_output_rows.extend(term_rows(dataset.name, fitted[family_key]))

    locked_outputs: list[dict[str, Any]] = []
    for _, row in locked_rows.iterrows():
        key = (row["model_family"], row["config_name"])
        cal_key = (row["model_family"], row["config_name"], row["calibration_type"])
        artifact = fitted[key]
        calibrator = calibrators[cal_key]
        locked_outputs.append(evaluate_locked_row(row, artifact, calibrator, X_test, y_test))
        if row["model_family"] not in {"EBM", "Scorecard"} and not any(
            existing.get("dataset") == dataset.name
            and existing.get("model_family") == row["model_family"]
            and existing.get("config_name") == row["config_name"]
            for existing in term_output_rows
        ):
            term_output_rows.extend(term_rows(dataset.name, artifact)[:30])

        model_path = OUTPUT_DIR / f"{dataset.name}_{row['config_name']}_{row['calibration_type']}.joblib"
        joblib.dump({"artifact": artifact, "calibrator": calibrator, "locked_row": row.to_dict()}, model_path)

    failures_path = OUTPUT_DIR / f"interpretable_failures_{dataset.name}.csv"
    failure_columns = ["dataset", "model_family", "config_name", "error"]
    pd.DataFrame(failures, columns=failure_columns).to_csv(failures_path, index=False)
    return validation_table, pd.DataFrame(locked_outputs), term_output_rows, constraint_rows


def write_summary(
    validation_all: pd.DataFrame,
    locked_all: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    """Write the human-readable experiment summary."""

    lines: list[str] = [
        "# Interpretable Middle Models Summary",
        "",
        "## Protocol",
        "- Frozen train/validation/test split was preserved.",
        "- Model fitting used model_train, calibration used calibration_val, and policy selection used validation only.",
        "- Held-out test rows are locked validation-selected evidence only; they are not used for ranking.",
        "- No new features or datasets were created.",
        "- EBM used the requested grid; max_rounds=1500 with early stopping was fixed for runtime control.",
        "",
    ]

    for dataset in sorted(validation_all["dataset"].unique()):
        subset = validation_all[validation_all["dataset"] == dataset].copy()
        best_by_family = (
            subset.sort_values(
                ["eligible_basic", "validation_selection_cost", "validation_pr_auc"],
                ascending=[False, True, False],
            )
            .groupby("model_family", as_index=False)
            .head(1)
        )
        lines.extend([f"## {dataset.upper()} validation best by family", ""])
        for _, row in best_by_family.iterrows():
            lines.append(
                "- {family}: {config}, policy={policy}, calibration={cal}, "
                "precision={precision:.3f}, recall={recall:.3f}, specificity={specificity:.3f}, "
                "PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, cost={cost:.1f}, eligible={eligible}".format(
                    family=row["model_family"],
                    config=row["config_name"],
                    policy=row["policy_type"],
                    cal=row["calibration_type"],
                    precision=row["validation_precision"],
                    recall=row["validation_recall"],
                    specificity=row["validation_specificity"],
                    pr_auc=row["validation_pr_auc"],
                    roc_auc=row["validation_roc_auc"],
                    cost=row["validation_selection_cost"],
                    eligible=row["eligible_basic"],
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Questions",
            "",
            "1. Is EBM better than Scorecard?",
            "   - The answer is dataset- and policy-dependent. Use the validation best-by-family rows above; EBM is retained only when it improves validation cost/PR-AUC without losing governance clarity.",
            "2. Did EBM approach CatBoost?",
            "   - EBM is evaluated as a middle model, not as a black-box replacement. Compare its PR-AUC and manual-review cost against the V2 CatBoost baseline in final selection.",
            "3. Did monotonic models lose performance?",
            "   - Monotonic candidates are reported separately with measured monotonicity violation probes. Any performance loss should be weighed against governance value.",
            "4. Did monotonic models give safer explanations?",
            "   - They provide explicit sign constraints for delay, utilization, and payment-intensity features; this is safer for governance than unconstrained feature effects.",
            "5. Final interpretable model: Scorecard, EBM, or monotonic model?",
            "   - This prompt does not replace the final validation-only selector. It provides candidates and locked evidence for the final attempt.",
            "6. Automatic rejection?",
            "   - No. These models remain screening/manual-review candidates unless precision and governance thresholds are explicitly satisfied.",
            "",
            "## Failure log",
            f"- Failed model rows: {len(failures)}",
        ]
    )
    if not failures.empty:
        for _, row in failures.head(20).iterrows():
            lines.append(f"- {row.get('dataset')}: {row.get('config_name')} failed: {row.get('error')}")

    (OUTPUT_DIR / "interpretable_models_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run interpretable middle-model experiment.")
    parser.add_argument(
        "--dataset",
        choices=["taiwan", "heloc", "both"],
        default="both",
        help="Dataset to run.",
    )
    parser.add_argument(
        "--max-ebm-configs",
        type=int,
        default=None,
        help="Optional debug cap. Omit for the full requested EBM grid.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the experiment and write all requested outputs."""

    args = parse_args()
    logger = setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = dataset_specs()
    if args.dataset != "both":
        specs = [spec for spec in specs if spec.name == args.dataset]

    all_validation: list[pd.DataFrame] = []
    all_locked: list[pd.DataFrame] = []
    all_terms: list[dict[str, Any]] = []
    all_constraints: list[dict[str, Any]] = []
    all_failures: list[pd.DataFrame] = []

    for spec in specs:
        validation, locked, terms, constraints = run_dataset(spec, args, logger)
        all_validation.append(validation)
        all_locked.append(locked)
        all_terms.extend(terms)
        all_constraints.extend(constraints)
        failure_path = OUTPUT_DIR / f"interpretable_failures_{spec.name}.csv"
        if failure_path.exists() and failure_path.stat().st_size > 0:
            all_failures.append(pd.read_csv(failure_path))

    validation_all = pd.concat(all_validation, ignore_index=True)
    locked_all = pd.concat(all_locked, ignore_index=True)
    failures = pd.concat(all_failures, ignore_index=True) if all_failures else pd.DataFrame()

    validation_all.to_csv(OUTPUT_DIR / "interpretable_models_validation_all.csv", index=False)
    best_configs = validation_all.sort_values(
        ["dataset", "eligible_basic", "validation_selection_cost", "validation_pr_auc", "interpretability_rank_score"],
        ascending=[True, False, True, False, False],
    ).reset_index(drop=True)
    best_configs.to_csv(OUTPUT_DIR / "interpretable_models_best_configs.csv", index=False)

    if "taiwan" in locked_all["dataset"].values:
        locked_all[locked_all["dataset"] == "taiwan"].to_csv(
            OUTPUT_DIR / "interpretable_models_locked_test_taiwan.csv",
            index=False,
        )
    if "heloc" in locked_all["dataset"].values:
        locked_all[locked_all["dataset"] == "heloc"].to_csv(
            OUTPUT_DIR / "interpretable_models_locked_test_heloc.csv",
            index=False,
        )

    terms = pd.DataFrame(all_terms)
    if not terms.empty:
        terms[(terms["dataset"] == "taiwan") & (terms["model_family"] == "EBM")].to_csv(
            OUTPUT_DIR / "ebm_global_terms_taiwan.csv",
            index=False,
        )
        terms[(terms["dataset"] == "heloc") & (terms["model_family"] == "EBM")].to_csv(
            OUTPUT_DIR / "ebm_global_terms_heloc.csv",
            index=False,
        )
        terms[terms["model_family"] == "Scorecard"].to_csv(OUTPUT_DIR / "scorecard_reason_codes.csv", index=False)
        terms[terms["model_family"].str.startswith("Monotonic", na=False)].to_csv(
            OUTPUT_DIR / "monotonic_global_terms.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(OUTPUT_DIR / "ebm_global_terms_taiwan.csv", index=False)
        pd.DataFrame().to_csv(OUTPUT_DIR / "ebm_global_terms_heloc.csv", index=False)
        pd.DataFrame().to_csv(OUTPUT_DIR / "scorecard_reason_codes.csv", index=False)

    pd.DataFrame(all_constraints).to_csv(OUTPUT_DIR / "monotonic_constraint_report.csv", index=False)
    write_summary(validation_all, locked_all, failures)

    audit = {
        "datasets": [spec.name for spec in specs],
        "ebm_grid_size_requested": len(ebm_specs()),
        "ebm_grid_size_run": len(ebm_specs()) if args.max_ebm_configs is None else args.max_ebm_configs,
        "test_set_used_for_selection": False,
        "model_fit_split": "model_train_from_frozen_train",
        "calibration_fit_split": "calibration_val_from_frozen_train",
        "policy_selection_split": "validation",
        "locked_test_rows": int(len(locked_all)),
        "validation_rows": int(len(validation_all)),
        "failed_rows": int(len(failures)),
    }
    (OUTPUT_DIR / "interpretable_models_protocol_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    logger.info("Completed interpretable middle-model experiment. Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
