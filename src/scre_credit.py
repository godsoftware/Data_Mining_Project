"""SCRE-Credit: Stability- and Cost-Regularized Ensemble for Credit Risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import RANDOM_SEED
from evaluation import binary_classification_metrics
from monotonic_models import (
    HELOC_MONOTONIC_PRIORS,
    TAIWAN_MONOTONIC_PRIORS,
    monotonic_constraint_vector,
    xgboost_constraint_string,
)
from scorecard import make_scorecard_pipeline

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


@dataclass
class ScreWeights:
    """Component weights used by the SCRE-Credit selection objective."""

    pr_auc: float = 0.20
    roc_auc: float = 0.15
    recall: float = 0.15
    calibration: float = 0.15
    cost: float = 0.15
    stability: float = 0.10
    faithfulness: float = 0.10


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_research_preprocessor(
    X: pd.DataFrame,
    categorical_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Build a missing-value-safe preprocessor for both Taiwan and HELOC."""

    categorical_set = set(categorical_columns or [])
    categorical = [column for column in X.columns if column in categorical_set]
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
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_pipeline(model, X: pd.DataFrame, categorical_columns: list[str] | None = None) -> Pipeline:
    """Create a standard preprocessing + estimator pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_research_preprocessor(X, categorical_columns)),
            ("model", model),
        ]
    )
    try:
        pipeline.set_output(transform="pandas")
    except ValueError:
        pass
    return pipeline


def make_numeric_model_pipeline(model) -> Pipeline:
    """Create an all-numeric pipeline that preserves original feature order."""

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    try:
        pipeline.set_output(transform="pandas")
    except ValueError:
        pass
    return pipeline


def make_smotenc_xgboost_pipeline(
    X: pd.DataFrame,
    categorical_columns: list[str] | None,
    random_state: int,
):
    """Create SMOTENC + preprocessing + XGBoost for the Taiwan full pool."""

    if SMOTENC is None or ImbalancedPipeline is None or XGBClassifier is None:
        return None

    categorical_indices = [
        X.columns.get_loc(column)
        for column in (categorical_columns or [])
        if column in X.columns
    ]
    if not categorical_indices:
        return None

    sampler = SMOTENC(categorical_features=categorical_indices, random_state=random_state)
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    return ImbalancedPipeline(
        steps=[
            ("sampler", sampler),
            ("preprocessor", build_research_preprocessor(X, categorical_columns)),
            ("model", model),
        ]
    )


def build_candidate_models(
    X: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    random_state: int = RANDOM_SEED,
    include_catboost: bool = True,
    include_scorecard: bool = True,
    model_profile: str = "taiwan_full",
) -> list[tuple[str, BaseEstimator]]:
    """Return candidate estimators for SCRE-Credit."""

    candidates: list[tuple[str, BaseEstimator]] = []
    full_pool = model_profile == "taiwan_full"
    priors = HELOC_MONOTONIC_PRIORS if model_profile == "heloc_reduced" else TAIWAN_MONOTONIC_PRIORS

    if include_scorecard:
        candidates.append(
            (
                "M2_woe_scorecard_logistic",
                make_scorecard_pipeline(categorical_columns=categorical_columns),
            )
        )

    candidates.insert(
        0,
        (
            "M1_logistic_regression",
            make_model_pipeline(
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=random_state,
                ),
                X,
                categorical_columns,
            ),
        ),
    )

    if full_pool:
        candidates.append(
            (
                "M3_random_forest",
                make_model_pipeline(
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                    X,
                    categorical_columns,
                ),
            )
        )

    if XGBClassifier is not None:
        candidates.append(
            (
                "M4_xgboost" if full_pool else "M3_xgboost",
                make_model_pipeline(
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.04,
                        max_depth=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                    X,
                    categorical_columns,
                ),
            )
        )
        candidates.append(
            (
                "M7_monotonic_xgboost" if full_pool else "M5_monotonic_xgboost",
                make_numeric_model_pipeline(
                    XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.04,
                        max_depth=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=3,
                        eval_metric="logloss",
                        monotone_constraints=xgboost_constraint_string(list(X.columns), priors),
                        random_state=random_state,
                        n_jobs=-1,
                    )
                ),
            )
        )

    if LGBMClassifier is not None:
        candidates.append(
            (
                "M5_lightgbm" if full_pool else "M4_lightgbm",
                make_model_pipeline(
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.04,
                        num_leaves=31,
                        min_child_samples=40,
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    X,
                    categorical_columns,
                ),
            )
        )
        candidates.append(
            (
                "M8_monotonic_lightgbm" if full_pool else "M5_monotonic_lightgbm",
                make_numeric_model_pipeline(
                    LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.04,
                        num_leaves=31,
                        min_child_samples=40,
                        monotone_constraints=monotonic_constraint_vector(list(X.columns), priors),
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=-1,
                    )
                ),
            )
        )
        if full_pool:
            candidates.append(
                (
                    "M10_class_weighted_lightgbm",
                    make_model_pipeline(
                        LGBMClassifier(
                            n_estimators=500,
                            learning_rate=0.04,
                            num_leaves=31,
                            min_child_samples=40,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                            verbose=-1,
                        ),
                        X,
                        categorical_columns,
                    ),
                )
            )

    if full_pool and include_catboost and CatBoostClassifier is not None:
        candidates.append(
            (
                "M6_catboost",
                make_model_pipeline(
                    CatBoostClassifier(
                        iterations=500,
                        learning_rate=0.04,
                        depth=5,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=random_state,
                        verbose=False,
                    ),
                    X,
                    categorical_columns,
                ),
            )
        )

    if full_pool:
        smotenc_xgb = make_smotenc_xgboost_pipeline(X, categorical_columns, random_state)
        if smotenc_xgb is not None:
            candidates.append(("M9_smotenc_xgboost", smotenc_xgb))

    return sorted(candidates, key=lambda item: _candidate_sort_key(item[0]))


def _candidate_sort_key(name: str) -> tuple[int, str]:
    marker = name.split("_", 1)[0]
    if marker.startswith("M") and marker[1:].isdigit():
        return int(marker[1:]), name
    return 99, name


class ScreCreditEnsemble(BaseEstimator, ClassifierMixin):
    """Reliability-weighted voting ensemble for credit risk.

    The ensemble learns model weights on an internal validation split. A model
    receives more weight when its normalized reliability score is higher.
    """

    def __init__(
        self,
        estimators: Iterable[tuple[str, BaseEstimator]],
        objective_weights: ScreWeights | None = None,
        fn_cost: float = 5.0,
        fp_cost: float = 1.0,
        review_cost: float = 0.5,
        threshold: float = 0.5,
        validation_size: float = 0.2,
        random_state: int = RANDOM_SEED,
        stability_scores: dict[str, float] | None = None,
        faithfulness_scores: dict[str, float] | None = None,
        min_review_capture_recall: float = 0.75,
        max_review_rate: float = 0.45,
    ) -> None:
        self.estimators = list(estimators)
        self.objective_weights = objective_weights or ScreWeights()
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.review_cost = review_cost
        self.threshold = threshold
        self.validation_size = validation_size
        self.random_state = random_state
        self.stability_scores = stability_scores or {}
        self.faithfulness_scores = faithfulness_scores or {}
        self.min_review_capture_recall = min_review_capture_recall
        self.max_review_rate = max_review_rate

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit candidate models and learn SCRE-Credit ensemble weights."""

        X_df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).astype(int)
        self.classes_ = np.asarray([0, 1])

        X_train, X_val, y_train, y_val = train_test_split(
            X_df,
            y_series,
            test_size=self.validation_size,
            stratify=y_series,
            random_state=self.random_state,
        )

        rows = []
        fitted_for_full_data = []
        validation_probabilities: dict[str, np.ndarray] = {}

        for name, estimator in self.estimators:
            validation_estimator = clone(estimator)
            validation_estimator.fit(X_train, y_train)
            proba = validation_estimator.predict_proba(X_val)[:, 1]
            validation_probabilities[name] = proba
            metrics = binary_classification_metrics(y_val, proba, threshold=self.threshold)
            expected_cost = self.fn_cost * metrics["fn"] + self.fp_cost * metrics["fp"]
            rows.append(
                {
                    "model": name,
                    "expected_cost": expected_cost,
                    "stability_raw": float(self.stability_scores.get(name, 1.0)),
                    "faithfulness_raw": float(self.faithfulness_scores.get(name, 1.0)),
                    **metrics,
                }
            )

            full_estimator = clone(estimator)
            full_estimator.fit(X_df, y_series)
            fitted_for_full_data.append((name, full_estimator))

        validation_results = self._add_reliability_scores(pd.DataFrame(rows))
        self.validation_results_ = validation_results.sort_values(
            "reliability_score",
            ascending=False,
        )
        reliability_sum = float(self.validation_results_["reliability_score"].sum())
        if reliability_sum <= 0:
            self.validation_results_["ensemble_weight"] = 1.0 / len(self.validation_results_)
        else:
            self.validation_results_["ensemble_weight"] = (
                self.validation_results_["reliability_score"] / reliability_sum
            )
        self.weights_ = self.validation_results_["ensemble_weight"].to_numpy()
        self.estimators_ = fitted_for_full_data

        weight_map = dict(
            zip(
                self.validation_results_["model"],
                self.validation_results_["ensemble_weight"],
            )
        )
        validation_ensemble_proba = np.zeros(len(y_val), dtype=float)
        for name, proba in validation_probabilities.items():
            validation_ensemble_proba += float(weight_map[name]) * proba
        self.validation_y_true_ = y_val.reset_index(drop=True)
        self.validation_proba_ = validation_ensemble_proba

        threshold_search = optimize_three_way_thresholds(
            y_val,
            validation_ensemble_proba,
            fn_cost=self.fn_cost,
            fp_cost=self.fp_cost,
            review_cost=self.review_cost,
            min_review_capture_recall=self.min_review_capture_recall,
            max_review_rate=self.max_review_rate,
        )
        self.three_way_threshold_search_ = threshold_search
        best_policy = threshold_search.iloc[0].to_dict()
        self.t_low_ = float(best_policy["t_low"])
        self.t_high_ = float(best_policy["t_high"])
        self.three_way_validation_summary_ = best_policy

        self.weight_table_ = self.validation_results_[
            [
                "model",
                "reliability_score",
                "ensemble_weight",
                "pr_auc_norm",
                "roc_auc_norm",
                "recall_norm",
                "calibration_norm",
                "cost_norm",
                "stability_norm",
                "faithfulness_norm",
            ]
        ].copy()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average default probabilities."""

        X_df = pd.DataFrame(X).copy()
        proba = np.zeros(len(X_df), dtype=float)
        estimator_map = dict(self.estimators_)

        for _, row in self.weight_table_.iterrows():
            name = row["model"]
            weight = float(row["ensemble_weight"])
            proba += weight * estimator_map[name].predict_proba(X_df)[:, 1]

        return np.column_stack([1.0 - proba, proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard decisions using the configured threshold."""

        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_risk_band(self, X: pd.DataFrame) -> np.ndarray:
        """Return three-level risk decisions: low, manual review, high."""

        proba = self.predict_proba(X)[:, 1]
        return risk_band_labels(proba, self.t_low_, self.t_high_)

    def _add_reliability_scores(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add normalized reliability components and final R_m."""

        weights = self.objective_weights
        scored = table.copy()
        scored["pr_auc_norm"] = normalize_higher_better(scored["pr_auc"])
        scored["roc_auc_norm"] = normalize_higher_better(scored["roc_auc"])
        scored["recall_norm"] = normalize_higher_better(scored["recall"])
        scored["calibration_norm"] = normalize_lower_better(scored["brier_score"])
        scored["cost_norm"] = normalize_lower_better(scored["expected_cost"])
        scored["stability_norm"] = normalize_higher_better(scored["stability_raw"])
        scored["faithfulness_norm"] = normalize_higher_better(scored["faithfulness_raw"])
        scored["reliability_score"] = (
            weights.pr_auc * scored["pr_auc_norm"]
            + weights.roc_auc * scored["roc_auc_norm"]
            + weights.recall * scored["recall_norm"]
            + weights.calibration * scored["calibration_norm"]
            + weights.cost * scored["cost_norm"]
            + weights.stability * scored["stability_norm"]
            + weights.faithfulness * scored["faithfulness_norm"]
        )
        return scored


def normalize_higher_better(values: pd.Series) -> pd.Series:
    """Min-max normalize a higher-is-better metric."""

    series = pd.Series(values, dtype=float)
    value_range = float(series.max() - series.min())
    if not np.isfinite(value_range) or value_range == 0:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / value_range


def normalize_lower_better(values: pd.Series) -> pd.Series:
    """Min-max normalize a lower-is-better metric into a higher-is-better score."""

    series = pd.Series(values, dtype=float)
    value_range = float(series.max() - series.min())
    if not np.isfinite(value_range) or value_range == 0:
        return pd.Series(np.ones(len(series)), index=series.index)
    return 1.0 - ((series - series.min()) / value_range)


def risk_band_labels(y_proba: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Map probabilities to low/manual/high risk labels."""

    y_proba = np.asarray(y_proba)
    labels = np.full(len(y_proba), "manual_review", dtype=object)
    labels[y_proba < t_low] = "low_risk"
    labels[y_proba > t_high] = "high_risk"
    return labels


def three_way_policy_metrics(
    y_true,
    y_proba,
    t_low: float,
    t_high: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    review_cost: float = 0.5,
) -> dict[str, float | int]:
    """Evaluate low/manual/high risk decisions."""

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    low = y_proba < t_low
    high = y_proba > t_high
    review = ~(low | high)

    positives = max(int((y_true == 1).sum()), 1)
    high_tp = int(np.sum(high & (y_true == 1)))
    high_fp = int(np.sum(high & (y_true == 0)))
    low_tn = int(np.sum(low & (y_true == 0)))
    low_fn = int(np.sum(low & (y_true == 1)))
    review_default = int(np.sum(review & (y_true == 1)))
    review_non_default = int(np.sum(review & (y_true == 0)))
    review_count = int(review.sum())
    expected_policy_cost = fn_cost * low_fn + fp_cost * high_fp + review_cost * review_count

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "expected_policy_cost": float(expected_policy_cost),
        "low_count": int(low.sum()),
        "manual_review_count": review_count,
        "high_count": int(high.sum()),
        "low_rate": float(low.mean()),
        "manual_review_rate": float(review.mean()),
        "high_rate": float(high.mean()),
        "low_false_negative": low_fn,
        "high_false_positive": high_fp,
        "low_true_negative": low_tn,
        "high_true_positive": high_tp,
        "manual_review_default": review_default,
        "manual_review_non_default": review_non_default,
        "high_risk_precision": high_tp / max(high_tp + high_fp, 1),
        "high_risk_recall": high_tp / positives,
        "review_capture_recall": (high_tp + review_default) / positives,
        "auto_decision_rate": float((low | high).mean()),
    }


def optimize_three_way_thresholds(
    y_true,
    y_proba,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    review_cost: float = 0.5,
    thresholds: np.ndarray | None = None,
    min_review_capture_recall: float = 0.75,
    max_review_rate: float = 0.45,
) -> pd.DataFrame:
    """Select low/high thresholds on validation by cost and recall trade-off."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    rows = []
    for t_low in thresholds:
        for t_high in thresholds:
            if t_low >= t_high:
                continue
            metrics = three_way_policy_metrics(
                y_true,
                y_proba,
                t_low,
                t_high,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
                review_cost=review_cost,
            )
            metrics["meets_review_capture_recall"] = (
                metrics["review_capture_recall"] >= min_review_capture_recall
            )
            metrics["meets_review_rate"] = metrics["manual_review_rate"] <= max_review_rate
            metrics["meets_constraints"] = (
                metrics["meets_review_capture_recall"] and metrics["meets_review_rate"]
            )
            rows.append(metrics)

    table = pd.DataFrame(rows)
    feasible = table[table["meets_constraints"]]
    if feasible.empty:
        feasible = table
    return feasible.sort_values(
        [
            "expected_policy_cost",
            "manual_review_rate",
            "t_low",
            "t_high",
        ]
    ).reset_index(drop=True)
