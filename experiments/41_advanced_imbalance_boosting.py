"""Advanced imbalance and calibration experiment for boosting models.

The experiment uses existing processed features only. The original frozen train
split is split again into model_train and calibration_val so probability
calibration is fit separately from validation-only threshold/policy selection.
The held-out test split is evaluated only for the locked validation-selected
configuration.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from data.split_leakage import stratified_train_validation_test_split  # noqa: E402
from data_preprocessing import TARGET_COLUMN  # noqa: E402
from evaluation import expected_calibration_error  # noqa: E402
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET  # noqa: E402
from src.config.paths import HELOC_MODEL_READY, OUTPUTS_DIR, TAIWAN_MODEL_READY  # noqa: E402


RANDOM_SEED = 42
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COST = 0.5
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)
T_LOW_GRID = np.round(np.arange(0.03, 0.4001, 0.01), 2)
T_HIGH_GRID = np.round(np.arange(0.10, 0.8001, 0.01), 2)
OUTPUT_DIR = OUTPUTS_DIR / "final_attempt" / "imbalance_boosting"
PROTOCOL_DIR = OUTPUTS_DIR / "final_attempt" / "protocol"
TAIWAN_CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset metadata."""

    name: str
    path: Path
    target: str
    categorical_columns: list[str]
    role: str


@dataclass(frozen=True)
class VariantSpec:
    """One boosting class-imbalance configuration."""

    model_family: str
    variant_name: str
    imbalance_strategy: str
    estimator: Any
    native_catboost: bool = False


class CatBoostNativeClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible wrapper for CatBoost native categorical handling."""

    _estimator_type = "classifier"

    def __init__(
        self,
        categorical_columns: tuple[str, ...] = (),
        iterations: int = 400,
        learning_rate: float = 0.04,
        depth: int = 5,
        auto_class_weights: str | None = None,
        class_weights: tuple[float, float] | None = None,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.categorical_columns = categorical_columns
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.auto_class_weights = auto_class_weights
        self.class_weights = class_weights
        self.random_state = random_state

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        for column in self.categorical_columns:
            if column in frame.columns:
                frame[column] = frame[column].astype("Int64", errors="ignore").astype(str).fillna("__missing__")
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "CatBoostNativeClassifier":
        frame = self._prepare_X(X)
        cat_features = [frame.columns.get_loc(column) for column in self.categorical_columns if column in frame.columns]
        params: dict[str, Any] = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": self.random_state,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        if self.auto_class_weights is not None:
            params["auto_class_weights"] = self.auto_class_weights
        if self.class_weights is not None:
            params["class_weights"] = list(self.class_weights)
        self.model_ = CatBoostClassifier(**params)
        self.model_.fit(frame, y, cat_features=cat_features)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._prepare_X(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self._prepare_X(X)).astype(int)


@dataclass
class ProbabilityCalibrator:
    """One-dimensional probability calibrator fit on calibration_val only."""

    calibration_type: str
    model: Any | None = None

    def fit(self, probability: np.ndarray, y_true: pd.Series | np.ndarray) -> "ProbabilityCalibrator":
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
    """Configure output logging."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("advanced_imbalance_boosting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "advanced_imbalance.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def one_hot_encoder() -> OneHotEncoder:
    """Version-compatible one-hot encoder."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical_columns: list[str], X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing fitted inside the model pipeline."""

    categorical = [column for column in categorical_columns if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def xgboost_estimator(**params: Any) -> XGBClassifier:
    """Create XGBoost estimator."""

    base = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    base.update(params)
    return XGBClassifier(**base)


def lightgbm_estimator(**params: Any) -> LGBMClassifier:
    """Create LightGBM estimator."""

    base = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    base.update(params)
    return LGBMClassifier(**base)


def build_variant_specs(categorical_columns: list[str]) -> list[VariantSpec]:
    """Build all requested boosting imbalance variants."""

    cat_cols = tuple(categorical_columns)
    variants: list[VariantSpec] = [
        VariantSpec("CatBoost", "catboost_baseline", "baseline", CatBoostNativeClassifier(categorical_columns=cat_cols), True),
        VariantSpec(
            "CatBoost",
            "catboost_auto_balanced",
            "auto_class_weights='Balanced'",
            CatBoostNativeClassifier(categorical_columns=cat_cols, auto_class_weights="Balanced"),
            True,
        ),
        VariantSpec(
            "CatBoost",
            "catboost_auto_sqrtbalanced",
            "auto_class_weights='SqrtBalanced'",
            CatBoostNativeClassifier(categorical_columns=cat_cols, auto_class_weights="SqrtBalanced"),
            True,
        ),
    ]
    for weight in [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        variants.append(
            VariantSpec(
                "CatBoost",
                f"catboost_class_weight_pos_{weight:g}",
                f"custom class_weights positive_weight={weight:g}",
                CatBoostNativeClassifier(categorical_columns=cat_cols, class_weights=(1.0, weight)),
                True,
            )
        )

    variants.append(VariantSpec("XGBoost", "xgboost_baseline", "baseline", xgboost_estimator()))
    for weight in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        variants.append(
            VariantSpec("XGBoost", f"xgboost_scale_pos_weight_{weight:g}", f"scale_pos_weight={weight:g}", xgboost_estimator(scale_pos_weight=weight))
        )
    for step in [0, 1, 3, 5]:
        variants.append(
            VariantSpec("XGBoost", f"xgboost_max_delta_step_{step}", f"max_delta_step={step}", xgboost_estimator(max_delta_step=step))
        )
    for weight in [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        for step in [1, 3, 5]:
            variants.append(
                VariantSpec(
                    "XGBoost",
                    f"xgboost_spw_{weight:g}_mds_{step}",
                    f"scale_pos_weight={weight:g}; max_delta_step={step}",
                    xgboost_estimator(scale_pos_weight=weight, max_delta_step=step),
                )
            )

    variants.extend(
        [
            VariantSpec("LightGBM", "lightgbm_baseline", "baseline", lightgbm_estimator()),
            VariantSpec("LightGBM", "lightgbm_is_unbalance", "is_unbalance=True", lightgbm_estimator(is_unbalance=True)),
            VariantSpec("LightGBM", "lightgbm_class_weight_balanced", "class_weight='balanced'", lightgbm_estimator(class_weight="balanced")),
        ]
    )
    for weight in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        variants.append(
            VariantSpec("LightGBM", f"lightgbm_scale_pos_weight_{weight:g}", f"scale_pos_weight={weight:g}", lightgbm_estimator(scale_pos_weight=weight))
        )
    return variants


def build_estimator(variant: VariantSpec, X: pd.DataFrame, categorical_columns: list[str]) -> Any:
    """Create a fitted-ready estimator or preprocessing pipeline."""

    if variant.native_catboost:
        return variant.estimator
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical_columns, X)),
            ("model", variant.estimator),
        ]
    )


def positive_class_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probability."""

    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"Positive class 1 missing from classes: {classes}")
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def expected_cost(fn: int, fp: int, fn_cost: float = FN_COST, fp_cost: float = FP_COST) -> float:
    """Return FN/FP weighted expected cost."""

    return float(fn_cost * fn + fp_cost * fp)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute binary threshold metrics."""

    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": float(expected_calibration_error(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "expected_cost": expected_cost(int(fn), int(fp)),
    }


def binary_policy_grid(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Evaluate all binary thresholds."""

    return pd.DataFrame([binary_metrics(y_true, probability, float(threshold)) for threshold in THRESHOLDS])


def select_binary_policy(grid: pd.DataFrame, policy: str) -> tuple[pd.Series | None, str]:
    """Select one threshold on validation only."""

    if policy == "threshold_0_50":
        selected = grid.iloc[(grid["threshold"] - 0.50).abs().argsort()].iloc[0]
        return selected, "fixed threshold 0.50"
    if policy == "cost_optimal_FN5_FP1":
        ordered = grid.sort_values(["expected_cost", "f1", "pr_auc", "threshold"], ascending=[True, False, False, False])
        return ordered.iloc[0], "min validation binary expected cost FN=5 FP=1"

    feasible = grid.copy()
    if policy == "precision_ge_0_40":
        feasible = feasible.loc[feasible["precision"] >= 0.40]
        reason = "precision >= 0.40"
    elif policy == "precision_ge_0_45":
        feasible = feasible.loc[feasible["precision"] >= 0.45]
        reason = "precision >= 0.45"
    elif policy == "specificity_ge_0_65":
        feasible = feasible.loc[feasible["specificity"] >= 0.65]
        reason = "specificity >= 0.65"
    elif policy == "balanced_precision_0_40_recall_0_60":
        feasible = feasible.loc[(feasible["precision"] >= 0.40) & (feasible["recall"] >= 0.60)]
        reason = "precision >= 0.40 and recall >= 0.60"
    else:
        raise ValueError(f"Unsupported binary policy: {policy}")

    if feasible.empty:
        return None, f"no feasible threshold for {reason}"
    ordered = feasible.sort_values(
        ["expected_cost", "f1", "pr_auc", "precision", "recall", "threshold"],
        ascending=[True, False, False, False, False, False],
    )
    return ordered.iloc[0], f"{reason}; then min validation binary expected cost"


def manual_review_metrics(y_true: np.ndarray, probability: np.ndarray, t_low: float, t_high: float) -> dict[str, Any]:
    """Evaluate one three-way manual-review band."""

    low = probability < t_low
    high = probability >= t_high
    manual = ~(low | high)
    default = y_true == 1
    non_default = ~default
    total = len(y_true)
    total_defaults = int(default.sum())
    total_nondefaults = int(non_default.sum())

    low_count = int(low.sum())
    manual_count = int(manual.sum())
    high_count = int(high.sum())
    auto_fn = int(np.sum(low & default))
    auto_fp = int(np.sum(high & non_default))
    auto_tp = int(np.sum(high & default))
    auto_tn = int(np.sum(low & non_default))
    manual_defaults = int(np.sum(manual & default))
    manual_nondefaults = int(np.sum(manual & non_default))
    high_precision = auto_tp / high_count if high_count else 0.0
    high_recall = auto_tp / total_defaults if total_defaults else 0.0
    high_specificity = 1.0 - (auto_fp / total_nondefaults) if total_nondefaults else 0.0
    review_adjusted_cost = float(auto_fp * FP_COST + auto_fn * FN_COST + manual_count * REVIEW_COST)
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "precision": float(high_precision),
        "recall": float(high_recall),
        "specificity": float(high_specificity),
        "f1": float(2 * high_precision * high_recall / (high_precision + high_recall)) if (high_precision + high_recall) else 0.0,
        "accuracy": np.nan,
        "tn": auto_tn,
        "fp": auto_fp,
        "fn": auto_fn,
        "tp": auto_tp,
        "manual_review_count": manual_count,
        "manual_review_rate": float(manual_count / total),
        "auto_decision_rate": float((low_count + high_count) / total),
        "low_risk_count": low_count,
        "high_risk_count": high_count,
        "low_risk_default_rate": float(auto_fn / low_count) if low_count else np.nan,
        "high_risk_default_rate": float(auto_tp / high_count) if high_count else np.nan,
        "manual_review_default_rate": float(manual_defaults / manual_count) if manual_count else np.nan,
        "manual_review_defaults": manual_defaults,
        "manual_review_nondefaults": manual_nondefaults,
        "expected_cost": review_adjusted_cost,
        "review_adjusted_cost": review_adjusted_cost,
    }


def select_manual_review_policy(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict[str, Any] | None, str]:
    """Select manual-review band under MR rate <= 0.30 on validation."""

    rows: list[dict[str, Any]] = []
    for t_low in T_LOW_GRID:
        for t_high in T_HIGH_GRID:
            if float(t_low) >= float(t_high):
                continue
            metrics = manual_review_metrics(y_true, probability, float(t_low), float(t_high))
            if metrics["manual_review_rate"] <= 0.30:
                rows.append(metrics)
    if not rows:
        return None, "no feasible manual-review band with MR rate <= 0.30"
    grid = pd.DataFrame(rows)
    ordered = grid.sort_values(
        ["review_adjusted_cost", "precision", "recall", "specificity", "manual_review_rate", "t_high"],
        ascending=[True, False, False, False, True, False],
    )
    return ordered.iloc[0].to_dict(), "MR rate <= 0.30; then min validation review-adjusted cost"


def dataset_eligibility(dataset: str, precision: float, recall: float, specificity: float) -> bool:
    """Return prompt-defined dataset eligibility."""

    if dataset == "taiwan":
        return bool(precision >= 0.40 and recall >= 0.55 and specificity >= 0.60)
    if dataset == "heloc":
        return bool(precision >= 0.60 and recall >= 0.80 and specificity >= 0.20)
    raise ValueError(dataset)


def load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load frozen split and create a separate calibration split from train."""

    frame = pd.read_csv(spec.path)
    split = stratified_train_validation_test_split(frame, target_column=spec.target)
    train = split.train.reset_index(drop=True)
    validation = split.validation.reset_index(drop=True)
    test = split.test.reset_index(drop=True)
    model_train, calibration_val = train_test_split(
        train,
        test_size=0.20,
        stratify=train[spec.target],
        random_state=RANDOM_SEED,
    )
    model_train = model_train.reset_index(drop=True)
    calibration_val = calibration_val.reset_index(drop=True)
    return (
        model_train.drop(columns=[spec.target]),
        model_train[spec.target].astype(int),
        calibration_val.drop(columns=[spec.target]),
        calibration_val[spec.target].astype(int),
        validation.drop(columns=[spec.target]),
        validation[spec.target].astype(int),
        test.drop(columns=[spec.target]),
        test[spec.target].astype(int),
    )


def add_probability_metrics(row: dict[str, Any], y_true: pd.Series, probability: np.ndarray, prefix: str) -> None:
    """Add threshold-independent probability metrics to a row."""

    y = np.asarray(y_true, dtype=int)
    row[f"{prefix}_roc_auc"] = float(roc_auc_score(y, probability))
    row[f"{prefix}_pr_auc"] = float(average_precision_score(y, probability))
    row[f"{prefix}_brier"] = float(brier_score_loss(y, probability))
    row[f"{prefix}_ece"] = float(expected_calibration_error(y, probability))


def evaluate_validation_policies(
    dataset: str,
    variant: VariantSpec,
    calibration_type: str,
    y_validation: pd.Series,
    probability: np.ndarray,
    fit_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all prompt policies on validation only."""

    y = np.asarray(y_validation, dtype=int)
    base = {
        "dataset": dataset,
        "model_family": variant.model_family,
        "variant_name": variant.variant_name,
        "imbalance_strategy": variant.imbalance_strategy,
        "calibration_type": calibration_type,
        "fit_seconds": fit_seconds,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "review_cost": REVIEW_COST,
        "calibration_fit_split": "calibration_val_from_train_only",
        "policy_selection_split": "validation",
        "test_set_used_for_selection": False,
        "new_features_created": False,
    }
    add_probability_metrics(base, y_validation, probability, "validation")

    rows: list[dict[str, Any]] = []
    grid = binary_policy_grid(y, probability)
    for policy in [
        "threshold_0_50",
        "cost_optimal_FN5_FP1",
        "precision_ge_0_40",
        "precision_ge_0_45",
        "specificity_ge_0_65",
        "balanced_precision_0_40_recall_0_60",
    ]:
        selected, reason = select_binary_policy(grid, policy)
        row = base.copy()
        row["policy_type"] = policy
        row["policy_family"] = "binary_threshold"
        row["selection_reason"] = reason
        row["policy_feasible"] = selected is not None
        row["t_low"] = np.nan
        row["t_high"] = np.nan
        row["manual_review_rate"] = 0.0
        row["auto_decision_rate"] = 1.0
        row["review_adjusted_cost"] = np.nan
        if selected is not None:
            for key, value in selected.items():
                row[f"validation_{key}"] = value
            row["validation_operational_cost"] = row["validation_expected_cost"]
            row["eligible"] = dataset_eligibility(dataset, row["validation_precision"], row["validation_recall"], row["validation_specificity"])
        else:
            row["eligible"] = False
        rows.append(row)

    manual_selected, reason = select_manual_review_policy(y, probability)
    manual_row = base.copy()
    manual_row["policy_type"] = "manual_review_band_mr_le_0_30"
    manual_row["policy_family"] = "manual_review"
    manual_row["selection_reason"] = reason
    manual_row["policy_feasible"] = manual_selected is not None
    if manual_selected is not None:
        for key, value in manual_selected.items():
            if key in {"t_low", "t_high"}:
                manual_row[key] = value
            else:
                manual_row[f"validation_{key}"] = value
        manual_row["manual_review_rate"] = manual_selected["manual_review_rate"]
        manual_row["auto_decision_rate"] = manual_selected["auto_decision_rate"]
        manual_row["review_adjusted_cost"] = manual_selected["review_adjusted_cost"]
        manual_row["validation_operational_cost"] = manual_selected["review_adjusted_cost"]
        manual_row["threshold"] = np.nan
        manual_row["validation_threshold"] = np.nan
        manual_row["eligible"] = dataset_eligibility(dataset, manual_selected["precision"], manual_selected["recall"], manual_selected["specificity"])
    else:
        manual_row["eligible"] = False
    rows.append(manual_row)

    comparison = base.copy()
    comparison["validation_best_binary_cost"] = min(
        [row.get("validation_expected_cost", np.nan) for row in rows if row.get("policy_family") == "binary_threshold" and row.get("policy_feasible")]
        or [np.nan]
    )
    comparison["validation_best_manual_review_cost"] = manual_row.get("review_adjusted_cost", np.nan)
    comparison["validation_best_manual_review_rate"] = manual_row.get("manual_review_rate", np.nan)
    return rows, comparison


def evaluate_locked_test(
    locked: pd.Series,
    y_test: pd.Series,
    probability: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one locked validation-selected policy on held-out test."""

    y = np.asarray(y_test, dtype=int)
    row = {
        "dataset": locked["dataset"],
        "model_family": locked["model_family"],
        "variant_name": locked["variant_name"],
        "imbalance_strategy": locked["imbalance_strategy"],
        "calibration_type": locked["calibration_type"],
        "policy_type": locked["policy_type"],
        "policy_family": locked["policy_family"],
        "test_set_used_for_selection": False,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "review_cost": REVIEW_COST,
    }
    add_probability_metrics(row, y_test, probability, "test")
    if locked["policy_family"] == "manual_review":
        metrics = manual_review_metrics(y, probability, float(locked["t_low"]), float(locked["t_high"]))
        row["threshold"] = np.nan
        row["t_low"] = float(locked["t_low"])
        row["t_high"] = float(locked["t_high"])
        for key, value in metrics.items():
            row[f"test_{key}"] = value
        row["test_operational_cost"] = metrics["review_adjusted_cost"]
    else:
        threshold = float(locked["validation_threshold"])
        metrics = binary_metrics(y, probability, threshold)
        row["threshold"] = threshold
        row["t_low"] = np.nan
        row["t_high"] = np.nan
        for key, value in metrics.items():
            row[f"test_{key}"] = value
        row["test_manual_review_rate"] = 0.0
        row["test_auto_decision_rate"] = 1.0
        row["test_review_adjusted_cost"] = np.nan
        row["test_operational_cost"] = metrics["expected_cost"]
    row["validation_operational_cost"] = locked["validation_operational_cost"]
    row["validation_precision"] = locked["validation_precision"]
    row["validation_recall"] = locked["validation_recall"]
    row["validation_specificity"] = locked["validation_specificity"]
    return row


def select_locked(validation: pd.DataFrame, dataset: str) -> pd.Series:
    """Select the final policy using validation evidence only."""

    pool = validation.loc[
        validation["dataset"].eq(dataset)
        & validation["policy_feasible"].eq(True)
        & validation["eligible"].eq(True)
    ].copy()
    if pool.empty:
        pool = validation.loc[validation["dataset"].eq(dataset) & validation["policy_feasible"].eq(True)].copy()
    pool["policy_preference"] = np.where(pool["policy_family"].eq("manual_review"), 0, 1)
    pool = pool.sort_values(
        [
            "validation_operational_cost",
            "policy_preference",
            "validation_pr_auc",
            "validation_ece",
            "validation_f1",
            "validation_fp",
        ],
        ascending=[True, True, False, True, False, True],
        na_position="last",
    )
    selected = pool.iloc[0].copy()
    selected["locked_by_validation_only"] = True
    return selected


def write_summary(
    validation: pd.DataFrame,
    best: pd.DataFrame,
    locked_tests: pd.DataFrame,
    calibration: pd.DataFrame,
    v2_baseline: pd.DataFrame | None,
) -> None:
    """Write markdown summary."""

    lines: list[str] = ["# Advanced Imbalance Boosting Summary", ""]
    lines.extend(
        [
            "## Protocol",
            "- No new features were created.",
            "- Base models were fit on `model_train`, calibration was fit on `calibration_val`, and policy selection used the frozen validation split.",
            "- Test set was used only for the validation-locked configuration.",
            "- LightGBM probability-quality caveat is active: weighted LightGBM rows must be judged with Brier/ECE after calibration.",
            "",
        ]
    )

    for dataset in sorted(best["dataset"].dropna().unique()):
        lines.append(f"## {dataset.upper()} best validation policies")
        top = best.loc[best["dataset"].eq(dataset)].head(8)
        for _, row in top.iterrows():
            mr = row.get("manual_review_rate", np.nan)
            lines.append(
                f"- {row['model_family']} / {row['variant_name']} / {row['calibration_type']} / {row['policy_type']}: "
                f"precision={row['validation_precision']:.4f}, recall={row['validation_recall']:.4f}, "
                f"specificity={row['validation_specificity']:.4f}, PR-AUC={row['validation_pr_auc']:.4f}, "
                f"cost={row['validation_operational_cost']:.1f}, MR rate={mr if pd.notna(mr) else 0:.3f}."
            )
        lines.append("")

    lines.extend(["## 1. Did class weighting improve precision?", ""])
    for dataset in sorted(validation["dataset"].dropna().unique()):
        base = validation.loc[
            validation["dataset"].eq(dataset)
            & validation["variant_name"].str.contains("baseline", na=False)
            & validation["policy_type"].eq("cost_optimal_FN5_FP1")
        ]["validation_precision"].max()
        weighted = validation.loc[
            validation["dataset"].eq(dataset)
            & ~validation["variant_name"].str.contains("baseline", na=False)
            & validation["policy_type"].eq("cost_optimal_FN5_FP1")
        ]["validation_precision"].max()
        lines.append(f"- {dataset}: baseline best precision={base:.4f}, weighted best precision={weighted:.4f}.")

    lines.extend(["", "## 2. What happened to recall?", ""])
    for dataset in sorted(validation["dataset"].dropna().unique()):
        selected = best.loc[best["dataset"].eq(dataset)].iloc[0]
        lines.append(
            f"- {dataset}: selected validation recall={selected['validation_recall']:.4f}; this is the recall retained after constraints/calibration."
        )

    lines.extend(["", "## 3. Did calibration degrade or improve?", ""])
    for dataset in sorted(calibration["dataset"].dropna().unique()):
        view = calibration.loc[calibration["dataset"].eq(dataset)].groupby("calibration_type")[
            ["validation_brier", "validation_ece"]
        ].median()
        for calibration_type, row in view.iterrows():
            lines.append(f"- {dataset} / {calibration_type}: median Brier={row['validation_brier']:.4f}, median ECE={row['validation_ece']:.4f}.")

    lines.extend(["", "## 4. Sigmoid vs isotonic", ""])
    for dataset in sorted(calibration["dataset"].dropna().unique()):
        med = calibration.loc[calibration["dataset"].eq(dataset)].groupby("calibration_type")["validation_ece"].median().sort_values()
        lines.append(f"- {dataset}: lowest median ECE calibration = {med.index[0]} ({med.iloc[0]:.4f}).")

    lines.extend(["", "## 5-7. Best variant by family", ""])
    for dataset in sorted(best["dataset"].dropna().unique()):
        lines.append(f"### {dataset}")
        for family in ["CatBoost", "XGBoost", "LightGBM"]:
            family_rows = best.loc[best["dataset"].eq(dataset) & best["model_family"].eq(family)]
            if family_rows.empty:
                lines.append(f"- {family}: no feasible row.")
            else:
                row = family_rows.iloc[0]
                lines.append(
                    f"- {family}: {row['variant_name']} with {row['calibration_type']} and {row['policy_type']} "
                    f"(cost={row['validation_operational_cost']:.1f}, PR-AUC={row['validation_pr_auc']:.4f})."
                )

    lines.extend(["", "## 8. Did any variant beat V2 policy?", ""])
    if v2_baseline is not None and not v2_baseline.empty:
        for _, baseline in v2_baseline.iterrows():
            dataset = baseline["dataset"]
            selected = best.loc[best["dataset"].eq(dataset)].iloc[0]
            v2_cost = baseline["validation_review_adjusted_cost"]
            lines.append(
                f"- {dataset}: best advanced validation operational cost={selected['validation_operational_cost']:.1f}; "
                f"V2 review-adjusted validation cost={v2_cost:.1f}. "
                f"{'Advanced variant is lower on this validation cost definition.' if selected['validation_operational_cost'] < v2_cost else 'V2 remains lower or not directly beaten.'}"
            )
    else:
        lines.append("- V2 baseline file not found.")

    lines.extend(
        [
            "",
            "## Locked held-out test evidence",
        ]
    )
    for _, row in locked_tests.iterrows():
        lines.append(
            f"- {row['dataset']}: {row['model_family']} / {row['variant_name']} / {row['calibration_type']} / {row['policy_type']} "
            f"test precision={row['test_precision']:.4f}, recall={row['test_recall']:.4f}, "
            f"specificity={row['test_specificity']:.4f}, operational cost={row['test_operational_cost']:.1f}."
        )

    (OUTPUT_DIR / "advanced_imbalance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(datasets: list[str], logger: logging.Logger) -> None:
    """Run the full advanced imbalance experiment."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = {
        "taiwan": DatasetSpec("taiwan", TAIWAN_MODEL_READY, TARGET_COLUMN, TAIWAN_CATEGORICAL_COLUMNS, "primary"),
        "heloc": DatasetSpec("heloc", HELOC_MODEL_READY, HELOC_TARGET, HELOC_CATEGORICAL_COLUMNS, "external_validation"),
    }
    validation_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    test_probability_cache: dict[tuple[str, str, str], np.ndarray] = {}
    split_cache: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = {}

    for dataset_name in datasets:
        dataset_spec = specs[dataset_name]
        logger.info("Loading %s", dataset_name)
        X_model_train, y_model_train, X_cal, y_cal, X_val, y_val, X_test, y_test = load_dataset(dataset_spec)
        split_cache[dataset_name] = (X_model_train, y_model_train, X_cal, y_cal, X_val, y_val, X_test, y_test)
        logger.info(
            "%s split sizes: model_train=%d calibration_val=%d validation=%d test=%d",
            dataset_name,
            len(y_model_train),
            len(y_cal),
            len(y_val),
            len(y_test),
        )
        variants = build_variant_specs(dataset_spec.categorical_columns)
        for idx, variant in enumerate(variants, start=1):
            logger.info("[%s] Variant %d/%d: %s / %s", dataset_name, idx, len(variants), variant.model_family, variant.variant_name)
            started = time.perf_counter()
            estimator = build_estimator(variant, X_model_train, dataset_spec.categorical_columns)
            try:
                estimator.fit(X_model_train, y_model_train)
                raw_cal = positive_class_probability(estimator, X_cal)
                raw_val = positive_class_probability(estimator, X_val)
                raw_test = positive_class_probability(estimator, X_test)
                fit_seconds = time.perf_counter() - started
            except Exception as exc:  # pragma: no cover - diagnostic path
                logger.exception("[%s] Variant failed: %s", dataset_name, variant.variant_name)
                failure_row = {
                    "dataset": dataset_name,
                    "model_family": variant.model_family,
                    "variant_name": variant.variant_name,
                    "imbalance_strategy": variant.imbalance_strategy,
                    "status": "failed",
                    "error_message": str(exc),
                    "test_set_used_for_selection": False,
                }
                validation_rows.append(failure_row)
                pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "advanced_imbalance_validation_all.csv", index=False)
                continue

            for calibration_type in ["uncalibrated", "sigmoid", "isotonic"]:
                calibrator = ProbabilityCalibrator(calibration_type).fit(raw_cal, y_cal)
                val_probability = calibrator.transform(raw_val)
                test_probability = calibrator.transform(raw_test)
                key = (dataset_name, variant.variant_name, calibration_type)
                test_probability_cache[key] = test_probability
                rows, comparison = evaluate_validation_policies(dataset_name, variant, calibration_type, y_val, val_probability, fit_seconds)
                validation_rows.extend(rows)
                add_probability_metrics(comparison, y_cal, calibrator.transform(raw_cal), "calibration_val")
                calibration_rows.append(comparison)

            pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "advanced_imbalance_validation_all.csv", index=False)
            pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / "advanced_imbalance_calibration_comparison.csv", index=False)

    validation_table = pd.DataFrame(validation_rows)
    calibration_table = pd.DataFrame(calibration_rows)
    success = validation_table.loc[validation_table.get("policy_feasible", False).eq(True)].copy()
    success["eligible_priority"] = np.where(success["eligible"].eq(True), 0, 1)
    success = success.sort_values(
        [
            "dataset",
            "eligible_priority",
            "validation_operational_cost",
            "validation_pr_auc",
            "validation_ece",
            "validation_f1",
        ],
        ascending=[True, True, True, False, True, False],
        na_position="last",
    )
    success["rank_validation_operational"] = success.groupby("dataset").cumcount() + 1
    success.to_csv(OUTPUT_DIR / "advanced_imbalance_best_configs.csv", index=False)

    locked_rows: list[dict[str, Any]] = []
    locked_config_rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        locked = select_locked(validation_table, dataset_name)
        locked_config_rows.append(locked.to_dict())
        _, _, _, _, _, _, _, y_test = split_cache[dataset_name]
        probability = test_probability_cache[(dataset_name, locked["variant_name"], locked["calibration_type"])]
        locked_rows.append(evaluate_locked_test(locked, y_test, probability))
    locked_test = pd.DataFrame(locked_rows)
    pd.DataFrame(locked_config_rows).to_csv(OUTPUT_DIR / "advanced_imbalance_locked_configs.csv", index=False)

    for dataset_name in datasets:
        locked_test.loc[locked_test["dataset"].eq(dataset_name)].to_csv(
            OUTPUT_DIR / f"advanced_imbalance_locked_test_{dataset_name}.csv",
            index=False,
        )
    if "taiwan" not in datasets:
        pd.DataFrame().to_csv(OUTPUT_DIR / "advanced_imbalance_locked_test_taiwan.csv", index=False)
    if "heloc" not in datasets:
        pd.DataFrame().to_csv(OUTPUT_DIR / "advanced_imbalance_locked_test_heloc.csv", index=False)

    v2_path = PROTOCOL_DIR / "v2_baseline_freeze.csv"
    v2 = pd.read_csv(v2_path) if v2_path.exists() else None
    write_summary(validation_table, success, locked_test, calibration_table, v2)

    audit = {
        "datasets": datasets,
        "validation_rows": int(len(validation_table)),
        "calibration_rows": int(len(calibration_table)),
        "test_columns_used_for_selection": False,
        "calibration_fit_split": "calibration_val_from_train_only",
        "policy_selection_split": "validation",
        "test_usage": "locked evaluation only",
    }
    (OUTPUT_DIR / "advanced_imbalance_protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    logger = setup_logging()
    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    logger.info("Starting advanced imbalance boosting for %s", datasets)
    run_experiment(datasets, logger)
    logger.info("Advanced imbalance boosting complete. Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
