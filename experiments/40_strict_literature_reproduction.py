"""Strict leakage-free reproduction of literature-inspired credit scoring setups.

This experiment intentionally keeps the frozen 60/20/20 train/validation/test
split. Resampling is placed inside imbalanced-learn pipelines and is therefore
fit only on train folds during CV and on the frozen train split for validation.
The held-out test split is evaluated only for the locked validation-selected
configuration.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from catboost import CatBoostClassifier  # noqa: E402
from imblearn.combine import SMOTETomek  # noqa: E402
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbalancedPipeline  # noqa: E402
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
N_SPLITS = 5
N_REPEATS = 3
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)
OUTPUT_DIR = OUTPUTS_DIR / "final_attempt" / "literature_reproduction"
PROTOCOL_DIR = OUTPUTS_DIR / "final_attempt" / "protocol"

TAIWAN_CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset metadata for the strict reproduction experiment."""

    name: str
    path: Path
    target: str
    categorical_columns: list[str]
    role: str


@dataclass(frozen=True)
class ConfigSpec:
    """One literature-inspired model/resampling configuration."""

    family: str
    config_name: str
    display_name: str
    estimator_factory: Callable[[], Any]
    sampler_factory: Callable[[], Any] | None = None
    native_catboost: bool = False
    literature_group: str = ""


class CatBoostNativeClassifier(ClassifierMixin, BaseEstimator):
    """Small sklearn-compatible wrapper for CatBoost native categorical support."""

    _estimator_type = "classifier"

    def __init__(
        self,
        categorical_columns: tuple[str, ...] = (),
        iterations: int = 300,
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
        """Return a CatBoost-safe copy with categorical columns encoded as strings."""

        frame = X.copy()
        for column in self.categorical_columns:
            if column in frame.columns:
                frame[column] = frame[column].astype("Int64", errors="ignore").astype(str).fillna("__missing__")
        return frame

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "CatBoostNativeClassifier":
        """Fit CatBoost using native categorical column indices."""

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
        """Return class probabilities."""

        return self.model_.predict_proba(self._prepare_X(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions."""

        return self.model_.predict(self._prepare_X(X)).astype(int)


def setup_logging() -> logging.Logger:
    """Configure a file and console logger for this experiment."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("strict_literature_reproduction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(OUTPUT_DIR / "literature_reproduction.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def one_hot_encoder() -> OneHotEncoder:
    """Create a dense one-hot encoder compatible across sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical_columns: list[str], X: pd.DataFrame) -> ColumnTransformer:
    """Build a train-fitted preprocessing transformer used before resampling."""

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


def gbdt_estimator() -> HistGradientBoostingClassifier:
    """Create a moderate-capacity histogram GBDT baseline."""

    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def xgboost_estimator() -> XGBClassifier:
    """Create a standard XGBoost classifier for reproduction tests."""

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def lightgbm_estimator() -> LGBMClassifier:
    """Create a standard LightGBM classifier for reproduction tests."""

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def smote_tomek_sampler() -> SMOTETomek:
    """Create SMOTE-Tomek with deterministic inner SMOTE."""

    return SMOTETomek(smote=SMOTE(random_state=RANDOM_SEED, k_neighbors=5), random_state=RANDOM_SEED)


def kmeans_smote_sampler() -> KMeansSMOTE:
    """Create a deterministic KMeansSMOTE sampler with conservative clusters."""

    kmeans = MiniBatchKMeans(n_clusters=12, random_state=RANDOM_SEED, n_init=3, batch_size=2048)
    return KMeansSMOTE(
        random_state=RANDOM_SEED,
        k_neighbors=5,
        cluster_balance_threshold=0.01,
        kmeans_estimator=kmeans,
    )


def build_config_specs(categorical_columns: list[str]) -> list[ConfigSpec]:
    """Return all requested strict literature-inspired configurations."""

    samplers: list[tuple[str, str, Callable[[], Any] | None]] = [
        ("no_resampling", "no resampling", None),
        ("random_over_sampler", "RandomOverSampler", lambda: RandomOverSampler(random_state=RANDOM_SEED)),
        ("smote", "SMOTE", lambda: SMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
        ("borderline_smote", "BorderlineSMOTE", lambda: BorderlineSMOTE(random_state=RANDOM_SEED, k_neighbors=5)),
        ("smote_tomek", "SMOTE-Tomek", smote_tomek_sampler),
        ("adasyn", "ADASYN", lambda: ADASYN(random_state=RANDOM_SEED, n_neighbors=5)),
        ("kmeans_smote", "KMeansSMOTE", kmeans_smote_sampler),
    ]
    specs: list[ConfigSpec] = []
    for suffix, label, sampler_factory in samplers:
        specs.append(
            ConfigSpec(
                family="GBDT",
                config_name=f"gbdt_{suffix}",
                display_name=f"GBDT + {label}",
                estimator_factory=gbdt_estimator,
                sampler_factory=sampler_factory,
                literature_group="A_GBDT_resampling",
            )
        )

    xgb_samplers = [samplers[0], samplers[2], samplers[3], samplers[4], samplers[6]]
    for suffix, label, sampler_factory in xgb_samplers:
        specs.append(
            ConfigSpec(
                family="XGBoost",
                config_name=f"xgboost_{suffix}",
                display_name=f"XGBoost + {label}",
                estimator_factory=xgboost_estimator,
                sampler_factory=sampler_factory,
                literature_group="B_XGBoost_resampling",
            )
        )

    lgbm_samplers = [samplers[0], samplers[2], samplers[3], samplers[4], samplers[6]]
    for suffix, label, sampler_factory in lgbm_samplers:
        specs.append(
            ConfigSpec(
                family="LightGBM",
                config_name=f"lightgbm_{suffix}",
                display_name=f"LightGBM + {label}",
                estimator_factory=lightgbm_estimator,
                sampler_factory=sampler_factory,
                literature_group="C_LightGBM_resampling",
            )
        )

    cat_cols = tuple(categorical_columns)
    specs.extend(
        [
            ConfigSpec(
                family="CatBoost",
                config_name="catboost_baseline",
                display_name="CatBoost baseline",
                estimator_factory=lambda: CatBoostNativeClassifier(categorical_columns=cat_cols),
                native_catboost=True,
                literature_group="D_CatBoost_class_strategy",
            ),
            ConfigSpec(
                family="CatBoost",
                config_name="catboost_class_weights",
                display_name="CatBoost class_weights",
                estimator_factory=lambda: CatBoostNativeClassifier(categorical_columns=cat_cols, class_weights=(1.0, 3.5)),
                native_catboost=True,
                literature_group="D_CatBoost_class_strategy",
            ),
            ConfigSpec(
                family="CatBoost",
                config_name="catboost_balanced",
                display_name="CatBoost Balanced",
                estimator_factory=lambda: CatBoostNativeClassifier(categorical_columns=cat_cols, auto_class_weights="Balanced"),
                native_catboost=True,
                literature_group="D_CatBoost_class_strategy",
            ),
            ConfigSpec(
                family="CatBoost",
                config_name="catboost_sqrtbalanced",
                display_name="CatBoost SqrtBalanced",
                estimator_factory=lambda: CatBoostNativeClassifier(categorical_columns=cat_cols, auto_class_weights="SqrtBalanced"),
                native_catboost=True,
                literature_group="D_CatBoost_class_strategy",
            ),
        ]
    )
    return specs


def build_estimator(spec: ConfigSpec, X: pd.DataFrame, categorical_columns: list[str]) -> Any:
    """Build one estimator, placing resampling inside the train-fold pipeline."""

    estimator = spec.estimator_factory()
    if spec.native_catboost:
        return estimator
    steps: list[tuple[str, Any]] = [("preprocessor", build_preprocessor(categorical_columns, X))]
    if spec.sampler_factory is not None:
        steps.append(("sampler", spec.sampler_factory()))
    steps.append(("model", estimator))
    return ImbalancedPipeline(steps=steps)


def positive_class_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability from a fitted estimator."""

    probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"Estimator classes do not include positive class 1: {classes}")
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def expected_cost(fn: int, fp: int, fn_cost: float = FN_COST, fp_cost: float = FP_COST) -> float:
    """Return FN/FP weighted cost."""

    return float(fn_cost * fn + fp_cost * fp)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute threshold-dependent and probability metrics."""

    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
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


def threshold_grid(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Evaluate the full threshold grid on validation."""

    return pd.DataFrame([binary_metrics(y_true, probability, float(threshold)) for threshold in THRESHOLDS])


def select_policy(grid: pd.DataFrame, policy_name: str) -> tuple[pd.Series | None, str]:
    """Select one validation threshold under the requested policy."""

    if policy_name == "threshold_0_50":
        row = grid.iloc[(grid["threshold"] - 0.50).abs().argsort()].iloc[0]
        return row, "fixed threshold 0.50"
    if policy_name == "f1_optimal":
        ordered = grid.sort_values(["f1", "expected_cost", "pr_auc", "threshold"], ascending=[False, True, False, False])
        return ordered.iloc[0], "max validation F1"
    if policy_name == "cost_optimal_FN5_FP1":
        ordered = grid.sort_values(["expected_cost", "f1", "pr_auc", "threshold"], ascending=[True, False, False, False])
        return ordered.iloc[0], "min validation expected cost FN=5 FP=1"

    feasible = grid.copy()
    if policy_name == "precision_ge_0_40":
        feasible = feasible.loc[feasible["precision"] >= 0.40]
        reason = "precision >= 0.40 then min cost"
    elif policy_name == "precision_ge_0_45":
        feasible = feasible.loc[feasible["precision"] >= 0.45]
        reason = "precision >= 0.45 then min cost"
    elif policy_name == "specificity_ge_0_65":
        feasible = feasible.loc[feasible["specificity"] >= 0.65]
        reason = "specificity >= 0.65 then min cost"
    elif policy_name == "balanced_precision_0_40_recall_0_60":
        feasible = feasible.loc[(feasible["precision"] >= 0.40) & (feasible["recall"] >= 0.60)]
        reason = "precision >= 0.40 and recall >= 0.60 then min cost"
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    if feasible.empty:
        return None, f"no feasible threshold for {reason}"
    ordered = feasible.sort_values(
        ["expected_cost", "f1", "pr_auc", "recall", "specificity", "threshold"],
        ascending=[True, False, False, False, False, False],
    )
    return ordered.iloc[0], reason


def cv_expected_cost_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return negative expected cost for sklearn scoring."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    _ = tn, tp
    return -expected_cost(int(fn), int(fp))


def run_cv(estimator: Any, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Run 5x3 repeated stratified CV on the frozen train split."""

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "neg_expected_cost_0_50": make_scorer(cv_expected_cost_scorer),
    }
    scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    result: dict[str, Any] = {
        "cv_n_splits": N_SPLITS,
        "cv_n_repeats": N_REPEATS,
        "cv_total_folds": N_SPLITS * N_REPEATS,
    }
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if metric == "neg_expected_cost_0_50":
            result["cv_expected_cost_0_50_mean"] = float(-mean_value)
            result["cv_expected_cost_0_50_std"] = std_value
        else:
            result[f"cv_{metric}_mean"] = mean_value
            result[f"cv_{metric}_std"] = std_value
    result["cv_fit_time_mean"] = float(np.mean(scores["fit_time"]))
    result["cv_score_time_mean"] = float(np.mean(scores["score_time"]))
    return result


def load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load a processed dataset and return frozen train/validation/test splits."""

    frame = pd.read_csv(spec.path)
    split = stratified_train_validation_test_split(frame, target_column=spec.target)
    train = split.train.reset_index(drop=True)
    validation = split.validation.reset_index(drop=True)
    test = split.test.reset_index(drop=True)
    return (
        train.drop(columns=[spec.target]),
        train[spec.target].astype(int),
        validation.drop(columns=[spec.target]),
        validation[spec.target].astype(int),
        test.drop(columns=[spec.target]),
        test[spec.target].astype(int),
    )


def max_precision_at_recall(grid: pd.DataFrame, min_recall: float = 0.60) -> float:
    """Return the best precision among thresholds preserving a recall floor."""

    feasible = grid.loc[grid["recall"] >= min_recall]
    if feasible.empty:
        return float("nan")
    return float(feasible["precision"].max())


def summarize_config(
    dataset_spec: DatasetSpec,
    spec: ConfigSpec,
    validation_probability: np.ndarray,
    y_validation: pd.Series,
    cv_metrics: dict[str, Any],
    fit_seconds: float,
    status: str,
    failure_stage: str | None = None,
    error_message: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build validation policy rows and one config-level summary row."""

    base = {
        "dataset": dataset_spec.name,
        "dataset_role": dataset_spec.role,
        "literature_group": spec.literature_group,
        "model_family": spec.family,
        "config_name": spec.config_name,
        "config_display_name": spec.display_name,
        "resampling_strategy": "none" if spec.sampler_factory is None else spec.display_name.split(" + ", 1)[-1],
        "native_catboost_categorical": bool(spec.native_catboost),
        "status": status,
        "failure_stage": failure_stage,
        "error_message": error_message,
        "fit_seconds_train_validation": fit_seconds,
        "test_set_used_for_selection": False,
        "resampling_inside_train_fold": True,
        "split_before_resampling": True,
    }
    base.update(cv_metrics)

    if status != "success":
        row = copy.deepcopy(base)
        row.update({"threshold_policy": None, "policy_feasible": False})
        return [row], row

    y = np.asarray(y_validation, dtype=int)
    grid = threshold_grid(y, validation_probability)
    policy_names = [
        "threshold_0_50",
        "f1_optimal",
        "cost_optimal_FN5_FP1",
        "precision_ge_0_40",
        "precision_ge_0_45",
        "specificity_ge_0_65",
        "balanced_precision_0_40_recall_0_60",
    ]
    policy_rows: list[dict[str, Any]] = []
    for policy_name in policy_names:
        selected, reason = select_policy(grid, policy_name)
        row = copy.deepcopy(base)
        row["threshold_policy"] = policy_name
        row["threshold_selection_reason"] = reason
        row["policy_feasible"] = selected is not None
        row["calibration_type"] = "uncalibrated"
        row["fn_cost"] = FN_COST
        row["fp_cost"] = FP_COST
        if selected is not None:
            for metric_name, value in selected.items():
                row[f"validation_{metric_name}"] = value
        policy_rows.append(row)

    cost_row = next(row for row in policy_rows if row["threshold_policy"] == "cost_optimal_FN5_FP1")
    summary = copy.deepcopy(base)
    summary.update(
        {
            "validation_pr_auc": float(average_precision_score(y, validation_probability)),
            "validation_roc_auc": float(roc_auc_score(y, validation_probability)),
            "validation_brier": float(brier_score_loss(y, validation_probability)),
            "validation_ece": float(expected_calibration_error(y, validation_probability)),
            "validation_precision_at_recall_0_60": max_precision_at_recall(grid, 0.60),
            "validation_best_f1": float(grid["f1"].max()),
            "validation_best_specificity_at_cost_policy": cost_row.get("validation_specificity"),
            "validation_expected_cost_cost_policy": cost_row.get("validation_expected_cost"),
            "validation_recall_cost_policy": cost_row.get("validation_recall"),
            "validation_precision_cost_policy": cost_row.get("validation_precision"),
            "validation_threshold_cost_policy": cost_row.get("validation_threshold"),
        }
    )
    return policy_rows, summary


def select_locked_config(config_summaries: pd.DataFrame, validation_rows: pd.DataFrame, dataset: str) -> dict[str, Any]:
    """Select one locked configuration and one threshold policy using validation only."""

    candidates = config_summaries.loc[
        config_summaries["dataset"].eq(dataset) & config_summaries["status"].eq("success")
    ].copy()
    candidates = candidates.sort_values(
        [
            "validation_pr_auc",
            "validation_roc_auc",
            "validation_best_f1",
            "validation_expected_cost_cost_policy",
            "validation_ece",
        ],
        ascending=[False, False, False, True, True],
    )
    if candidates.empty:
        raise ValueError(f"No successful configs for {dataset}.")
    config = candidates.iloc[0].to_dict()

    policy_pool = validation_rows.loc[
        validation_rows["dataset"].eq(dataset)
        & validation_rows["config_name"].eq(config["config_name"])
        & validation_rows["status"].eq("success")
        & validation_rows["policy_feasible"].eq(True)
    ].copy()
    policy_pool = policy_pool.sort_values(
        [
            "validation_expected_cost",
            "validation_f1",
            "validation_pr_auc",
            "validation_precision",
            "validation_threshold",
        ],
        ascending=[True, False, False, False, False],
    )
    selected_policy = policy_pool.iloc[0].to_dict()
    return {
        "dataset": dataset,
        "config_name": config["config_name"],
        "config_display_name": config["config_display_name"],
        "model_family": config["model_family"],
        "literature_group": config["literature_group"],
        "threshold_policy": selected_policy["threshold_policy"],
        "selected_threshold_validation": float(selected_policy["validation_threshold"]),
        "validation_pr_auc": float(config["validation_pr_auc"]),
        "validation_roc_auc": float(config["validation_roc_auc"]),
        "validation_brier": float(config["validation_brier"]),
        "validation_ece": float(config["validation_ece"]),
        "validation_precision": float(selected_policy["validation_precision"]),
        "validation_recall": float(selected_policy["validation_recall"]),
        "validation_specificity": float(selected_policy["validation_specificity"]),
        "validation_f1": float(selected_policy["validation_f1"]),
        "validation_expected_cost": float(selected_policy["validation_expected_cost"]),
        "selection_metric_primary": "validation PR-AUC",
        "threshold_selection_metric": "validation expected cost among feasible policies",
        "test_set_used_for_selection": False,
    }


def evaluate_locked_test(
    locked: dict[str, Any],
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    """Evaluate the validation-locked configuration once on natural test data."""

    probability = positive_class_probability(estimator, X_test)
    metrics = binary_metrics(np.asarray(y_test, dtype=int), probability, locked["selected_threshold_validation"])
    row = {
        "dataset": dataset_spec.name,
        "dataset_role": dataset_spec.role,
        "config_name": locked["config_name"],
        "config_display_name": locked["config_display_name"],
        "model_family": locked["model_family"],
        "literature_group": locked["literature_group"],
        "threshold_policy": locked["threshold_policy"],
        "threshold_locked_on_validation": locked["selected_threshold_validation"],
        "calibration_type": "uncalibrated",
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "test_set_used_for_selection": False,
        "test_distribution": "natural held-out split; no resampling",
        "test_positive_rate": float(np.mean(y_test)),
        "test_rows": int(len(y_test)),
    }
    for metric_name, value in metrics.items():
        row[f"test_{metric_name}"] = value
    for key, value in locked.items():
        if key.startswith("validation_") or key in {"selection_metric_primary", "threshold_selection_metric"}:
            row[key] = value
    return row


def write_summary(
    validation_rows: pd.DataFrame,
    best_configs: pd.DataFrame,
    locked_tests: pd.DataFrame,
    v2_baseline: pd.DataFrame | None,
) -> None:
    """Write the markdown interpretation requested by the prompt."""

    successful = validation_rows.loc[validation_rows["status"].eq("success")].copy()
    kmeans = successful.loc[successful["config_name"].str.contains("kmeans_smote", na=False)]
    no_resampling = successful.loc[successful["config_name"].str.contains("no_resampling|catboost_baseline", regex=True, na=False)]
    kmeans_help_lines = []
    for dataset in sorted(successful["dataset"].dropna().unique()):
        k_best = kmeans.loc[kmeans["dataset"].eq(dataset)]["validation_pr_auc"].max()
        n_best = no_resampling.loc[no_resampling["dataset"].eq(dataset)]["validation_pr_auc"].max()
        if pd.notna(k_best) and pd.notna(n_best):
            delta = k_best - n_best
            verdict = "helped" if delta > 0.002 else "did not materially help"
            kmeans_help_lines.append(f"- {dataset}: best KMeansSMOTE PR-AUC={k_best:.4f}, no-resampling/baseline PR-AUC={n_best:.4f}; {verdict}.")
        else:
            kmeans_help_lines.append(f"- {dataset}: KMeansSMOTE comparison unavailable because at least one run failed.")

    v2_lines: list[str] = []
    if v2_baseline is not None and not v2_baseline.empty:
        for _, row in v2_baseline.iterrows():
            dataset = row["dataset"]
            locked = locked_tests.loc[locked_tests["dataset"].eq(dataset)]
            if locked.empty:
                continue
            locked_row = locked.iloc[0]
            v2_lines.append(
                "- "
                f"{dataset}: reproduction locked test PR-AUC={locked_row['test_pr_auc']:.4f}, "
                f"recall={locked_row['test_recall']:.4f}, precision={locked_row['test_precision']:.4f}. "
                f"V2 operational baseline is a manual-review policy ({row['selected_model']}); "
                "binary reproduction costs are not directly comparable to review-adjusted V2 costs."
            )

    fail_rows = validation_rows.loc[validation_rows["status"].ne("success")]
    fail_text = "None."
    if not fail_rows.empty:
        fail_text = "\n".join(
            f"- {row['dataset']} / {row['config_display_name']}: {row.get('failure_stage')} - {row.get('error_message')}"
            for _, row in fail_rows.drop_duplicates(["dataset", "config_name"]).iterrows()
        )

    top_lines: list[str] = []
    for dataset in sorted(best_configs["dataset"].dropna().unique()):
        rows = best_configs.loc[best_configs["dataset"].eq(dataset)].head(5)
        top_lines.append(f"### {dataset.upper()} top validation configs")
        for _, row in rows.iterrows():
            top_lines.append(
                f"- {row['config_display_name']}: PR-AUC={row['validation_pr_auc']:.4f}, "
                f"ROC-AUC={row['validation_roc_auc']:.4f}, "
                f"cost-policy precision={row['validation_precision_cost_policy']:.4f}, "
                f"recall={row['validation_recall_cost_policy']:.4f}, "
                f"cost={row['validation_expected_cost_cost_policy']:.1f}"
            )

    content = f"""# Strict Literature Reproduction Summary

## Protocol status
- Split policy: frozen 60/20/20 stratified train/validation/test split.
- CV: RepeatedStratifiedKFold with n_splits={N_SPLITS}, n_repeats={N_REPEATS}, random_state={RANDOM_SEED}.
- Resampling: inside imbalanced-learn pipelines only; no split-before-SMOTE violation.
- Test usage: held out until the validation-selected locked configuration was fixed.
- New features: none.

## 1. Did literature-inspired resampling improve AUC/PR-AUC?
{chr(10).join(top_lines)}

Overall, resampling was evaluated as a protocol stress test, not as a score-inflation exercise. Improvements must be interpreted on validation first and then checked once on the natural held-out test distribution for the locked configuration.

## 2. Did KMeansSMOTE produce a literature-style jump?
{chr(10).join(kmeans_help_lines)}

## 3. Did natural test precision/recall balance improve?
{chr(10).join(v2_lines) if v2_lines else "- V2 baseline comparison unavailable."}

## 4. Did resampling hurt calibration?
Calibration was measured with Brier score and ECE for every validation configuration. Any resampled model with improved PR-AUC but worse ECE/Brier should be treated as a screening model requiring calibration review, not as a ready automatic decision model.

## 5. Is the best config better than V2 manual-review policy?
Not as an operational replacement by default. V2 uses a corrected manual-review cost model and capacity-aware three-way policy; this reproduction experiment mainly tests whether high literature-style binary-model scores survive a leakage-free protocol.

## 6. Is there any leakage-like high score?
No result should be treated as leakage evidence solely because it is high. Leakage suspicion would require implausibly high test AUC/PR-AUC, train/test distribution contamination, or split-before-resampling. This script explicitly prevents split-before-resampling and does not resample test.

## 7. If high scores are not reproduced, what is the likely reason?
The likely explanation is protocol difference: many optimistic literature scores can arise from resampling before splitting, test-set balancing, weak separation between model selection and final evaluation, or reporting threshold-insensitive metrics without natural-distribution precision/cost checks.

## Failed configurations
{fail_text}
"""
    (OUTPUT_DIR / "literature_reproduction_summary.md").write_text(content, encoding="utf-8")


def run_experiment(datasets: list[str], logger: logging.Logger) -> None:
    """Run the full strict reproduction workflow."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_specs = {
        "taiwan": DatasetSpec("taiwan", TAIWAN_MODEL_READY, TARGET_COLUMN, TAIWAN_CATEGORICAL_COLUMNS, "primary"),
        "heloc": DatasetSpec("heloc", HELOC_MODEL_READY, HELOC_TARGET, HELOC_CATEGORICAL_COLUMNS, "external_validation"),
    }
    validation_rows: list[dict[str, Any]] = []
    config_summaries: list[dict[str, Any]] = []
    fitted_estimators: dict[tuple[str, str], Any] = {}
    split_cache: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]] = {}

    for dataset_name in datasets:
        dataset_spec = dataset_specs[dataset_name]
        logger.info("Loading %s from %s", dataset_name, dataset_spec.path)
        X_train, y_train, X_validation, y_validation, X_test, y_test = load_dataset(dataset_spec)
        split_cache[dataset_name] = (X_train, y_train, X_validation, y_validation, X_test, y_test)
        logger.info(
            "%s split sizes: train=%d validation=%d test=%d positive_rates=(%.4f, %.4f, %.4f)",
            dataset_name,
            len(y_train),
            len(y_validation),
            len(y_test),
            float(y_train.mean()),
            float(y_validation.mean()),
            float(y_test.mean()),
        )

        specs = build_config_specs(dataset_spec.categorical_columns)
        for index, spec in enumerate(specs, start=1):
            logger.info("[%s] Config %d/%d: %s", dataset_name, index, len(specs), spec.display_name)
            started = time.perf_counter()
            estimator = build_estimator(spec, X_train, dataset_spec.categorical_columns)
            try:
                cv_metrics = run_cv(estimator, X_train, y_train)
                logger.info(
                    "[%s] %s CV PR-AUC %.4f ROC-AUC %.4f",
                    dataset_name,
                    spec.display_name,
                    cv_metrics.get("cv_pr_auc_mean", math.nan),
                    cv_metrics.get("cv_roc_auc_mean", math.nan),
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                elapsed = time.perf_counter() - started
                logger.exception("[%s] CV failed for %s", dataset_name, spec.display_name)
                rows, summary = summarize_config(dataset_spec, spec, np.array([]), y_validation, {}, elapsed, "failed", "cv", str(exc))
                validation_rows.extend(rows)
                config_summaries.append(summary)
                pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "literature_reproduction_validation_all.csv", index=False)
                pd.DataFrame(config_summaries).to_csv(OUTPUT_DIR / "literature_reproduction_config_summaries.csv", index=False)
                continue

            try:
                estimator.fit(X_train, y_train)
                probability = positive_class_probability(estimator, X_validation)
                elapsed = time.perf_counter() - started
                rows, summary = summarize_config(dataset_spec, spec, probability, y_validation, cv_metrics, elapsed, "success")
                validation_rows.extend(rows)
                config_summaries.append(summary)
                fitted_estimators[(dataset_name, spec.config_name)] = estimator
                logger.info(
                    "[%s] %s validation PR-AUC %.4f ROC-AUC %.4f ECE %.4f",
                    dataset_name,
                    spec.display_name,
                    summary["validation_pr_auc"],
                    summary["validation_roc_auc"],
                    summary["validation_ece"],
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                elapsed = time.perf_counter() - started
                logger.exception("[%s] Train/validation failed for %s", dataset_name, spec.display_name)
                rows, summary = summarize_config(dataset_spec, spec, np.array([]), y_validation, cv_metrics, elapsed, "failed", "train_validation", str(exc))
                validation_rows.extend(rows)
                config_summaries.append(summary)

            pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "literature_reproduction_validation_all.csv", index=False)
            pd.DataFrame(config_summaries).to_csv(OUTPUT_DIR / "literature_reproduction_config_summaries.csv", index=False)

    validation_table = pd.DataFrame(validation_rows)
    config_table = pd.DataFrame(config_summaries)
    config_table = config_table.sort_values(
        ["dataset", "validation_pr_auc", "validation_roc_auc", "validation_best_f1", "validation_expected_cost_cost_policy"],
        ascending=[True, False, False, False, True],
        na_position="last",
    )
    config_table["rank_by_validation_pr_auc"] = config_table.groupby("dataset")["validation_pr_auc"].rank(
        ascending=False,
        method="first",
    )
    config_table.to_csv(OUTPUT_DIR / "literature_reproduction_best_configs.csv", index=False)

    locked_rows: list[dict[str, Any]] = []
    locked_configs: list[dict[str, Any]] = []
    for dataset_name in datasets:
        locked = select_locked_config(config_table, validation_table, dataset_name)
        locked_configs.append(locked)
        estimator = fitted_estimators.get((dataset_name, locked["config_name"]))
        if estimator is None:
            dataset_spec = dataset_specs[dataset_name]
            X_train, y_train, _, _, _, _ = split_cache[dataset_name]
            estimator = build_estimator(
                next(spec for spec in build_config_specs(dataset_spec.categorical_columns) if spec.config_name == locked["config_name"]),
                X_train,
                dataset_spec.categorical_columns,
            )
            estimator.fit(X_train, y_train)
        _, _, _, _, X_test, y_test = split_cache[dataset_name]
        locked_rows.append(evaluate_locked_test(locked, estimator, X_test, y_test, dataset_specs[dataset_name]))

    locked_test = pd.DataFrame(locked_rows)
    locked_config_table = pd.DataFrame(locked_configs)
    locked_config_table.to_csv(OUTPUT_DIR / "literature_reproduction_locked_configs.csv", index=False)

    for dataset_name in datasets:
        dataset_locked = locked_test.loc[locked_test["dataset"].eq(dataset_name)]
        dataset_locked.to_csv(OUTPUT_DIR / f"literature_reproduction_locked_test_{dataset_name}.csv", index=False)
    if "taiwan" not in datasets:
        pd.DataFrame().to_csv(OUTPUT_DIR / "literature_reproduction_locked_test_taiwan.csv", index=False)
    if "heloc" not in datasets:
        pd.DataFrame().to_csv(OUTPUT_DIR / "literature_reproduction_locked_test_heloc.csv", index=False)

    v2_path = PROTOCOL_DIR / "v2_baseline_freeze.csv"
    v2_baseline = pd.read_csv(v2_path) if v2_path.exists() else None
    write_summary(validation_table, config_table, locked_test, v2_baseline)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["taiwan", "heloc", "both"],
        default="both",
        help="Dataset subset to run. Default runs Taiwan and HELOC.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    logger = setup_logging()
    args = parse_args()
    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    logger.info("Starting strict literature reproduction for datasets=%s", datasets)
    run_experiment(datasets, logger)
    logger.info("Strict literature reproduction complete. Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
