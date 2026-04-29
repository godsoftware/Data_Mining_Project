"""SCRE-Credit reliability-weighted probability ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SCRE_COMPONENT_WEIGHTS = {
    "pr_auc": 0.20,
    "roc_auc": 0.15,
    "recall": 0.15,
    "calibration_error": 0.15,
    "expected_cost": 0.15,
    "stability_score": 0.10,
    "faithfulness_score": 0.10,
}

HIGHER_IS_BETTER = ["pr_auc", "roc_auc", "recall", "stability_score", "faithfulness_score"]
LOWER_IS_BETTER = ["calibration_error", "expected_cost"]
REQUIRED_METRIC_COLUMNS = ["model", *HIGHER_IS_BETTER, *LOWER_IS_BETTER]
PARETO_HIGHER_IS_BETTER = ["pr_auc", "roc_auc", "recall"]
PARETO_LOWER_IS_BETTER = ["calibration_error", "expected_cost"]


class SCREMetricError(ValueError):
    """Raised when SCRE-Credit metric inputs are missing or invalid."""


@dataclass
class SCRECreditHybridClassifier(BaseEstimator, ClassifierMixin):
    """Combine calibrated base-model probabilities using SCRE-Credit weights.

    The class does not train base models. It expects a probability matrix whose
    columns are base-model names and a metric table with the same model names.
    Final probability calibration and threshold selection are fit on validation
    probabilities only.
    """

    component_weights: dict[str, float] = field(default_factory=lambda: SCRE_COMPONENT_WEIGHTS.copy())
    final_calibration_method: str = "isotonic"
    threshold_grid: np.ndarray | None = None
    fn_cost: float = 5.0
    fp_cost: float = 1.0

    def fit_from_probabilities(
        self,
        probability_frame: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        metric_table: pd.DataFrame,
    ) -> "SCRECreditHybridClassifier":
        """Fit ensemble weights, final calibrator, and cost-sensitive threshold."""

        probabilities = validate_probability_frame(probability_frame)
        y_array = np.asarray(y_true).astype(int)
        if len(probabilities) != len(y_array):
            raise ValueError("probability_frame and y_true must have the same number of rows.")

        self.weight_table_ = build_scre_weight_table(metric_table, self.component_weights)
        self.model_names_ = self.weight_table_["model"].tolist()
        missing_models = sorted(set(self.model_names_) - set(probabilities.columns))
        if missing_models:
            raise SCREMetricError(f"Missing probability columns for models: {missing_models}")

        raw_probability = weighted_probability_sum(
            probabilities[self.model_names_],
            self.weight_table_.set_index("model")["ensemble_weight"],
        )
        self.final_calibrator_ = fit_final_calibrator(
            raw_probability,
            y_array,
            method=self.final_calibration_method,
        )
        calibrated_probability = apply_final_calibrator(self.final_calibrator_, raw_probability)
        threshold_table = threshold_cost_table(
            y_array,
            calibrated_probability,
            thresholds=self.threshold_grid,
            fn_cost=self.fn_cost,
            fp_cost=self.fp_cost,
        )
        best_row = threshold_table.sort_values(["expected_cost", "threshold"]).iloc[0]

        self.classes_ = np.asarray([0, 1])
        self.threshold_ = float(best_row["threshold"])
        self.threshold_table_ = threshold_table
        self.validation_probability_ = calibrated_probability
        self.validation_raw_probability_ = raw_probability
        self.validation_threshold_summary_ = best_row.to_dict()
        return self

    def fit_from_weight_table(
        self,
        probability_frame: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        weight_table: pd.DataFrame,
    ) -> "SCRECreditHybridClassifier":
        """Fit final calibration and threshold from preselected ensemble weights.

        This is used by SCRE-Pareto and SCRE-Optimized, where the model pool
        and/or weights are selected upstream on validation data only.
        """

        probabilities = validate_probability_frame(probability_frame)
        y_array = np.asarray(y_true).astype(int)
        if len(probabilities) != len(y_array):
            raise ValueError("probability_frame and y_true must have the same number of rows.")

        self.weight_table_ = validate_weight_table(weight_table)
        self.model_names_ = self.weight_table_["model"].tolist()
        missing_models = sorted(set(self.model_names_) - set(probabilities.columns))
        if missing_models:
            raise SCREMetricError(f"Missing probability columns for models: {missing_models}")

        raw_probability = weighted_probability_sum(
            probabilities[self.model_names_],
            self.weight_table_.set_index("model")["ensemble_weight"],
        )
        self.final_calibrator_ = fit_final_calibrator(
            raw_probability,
            y_array,
            method=self.final_calibration_method,
        )
        calibrated_probability = apply_final_calibrator(self.final_calibrator_, raw_probability)
        threshold_table = threshold_cost_table(
            y_array,
            calibrated_probability,
            thresholds=self.threshold_grid,
            fn_cost=self.fn_cost,
            fp_cost=self.fp_cost,
        )
        best_row = threshold_table.sort_values(["expected_cost", "threshold"]).iloc[0]

        self.classes_ = np.asarray([0, 1])
        self.threshold_ = float(best_row["threshold"])
        self.threshold_table_ = threshold_table
        self.validation_probability_ = calibrated_probability
        self.validation_raw_probability_ = raw_probability
        self.validation_threshold_summary_ = best_row.to_dict()
        return self

    def predict_proba_from_base(self, probability_frame: pd.DataFrame) -> np.ndarray:
        """Return final calibrated probabilities from base-model probabilities."""

        self._check_is_fitted()
        probabilities = validate_probability_frame(probability_frame)
        missing_models = sorted(set(self.model_names_) - set(probabilities.columns))
        if missing_models:
            raise SCREMetricError(f"Missing probability columns for models: {missing_models}")
        raw_probability = weighted_probability_sum(
            probabilities[self.model_names_],
            self.weight_table_.set_index("model")["ensemble_weight"],
        )
        final_probability = apply_final_calibrator(self.final_calibrator_, raw_probability)
        return np.column_stack([1.0 - final_probability, final_probability])

    def predict_from_base(self, probability_frame: pd.DataFrame) -> np.ndarray:
        """Return hard labels using the selected cost-sensitive threshold."""

        return (self.predict_proba_from_base(probability_frame)[:, 1] >= self.threshold_).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Alias for sklearn-style use when X is a base-probability frame."""

        return self.predict_proba_from_base(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Alias for sklearn-style use when X is a base-probability frame."""

        return self.predict_from_base(X)

    def _check_is_fitted(self) -> None:
        """Raise a clear error if the hybrid classifier has not been fit."""

        if not hasattr(self, "weight_table_"):
            raise RuntimeError("SCRECreditHybridClassifier is not fit. Call fit_from_probabilities first.")


def validate_metric_table(metric_table: pd.DataFrame) -> pd.DataFrame:
    """Validate and return the metric table in SCRE-Credit canonical form."""

    missing = [column for column in REQUIRED_METRIC_COLUMNS if column not in metric_table.columns]
    if missing:
        raise SCREMetricError(f"Missing SCRE-Credit metric columns: {missing}")
    if metric_table["model"].duplicated().any():
        duplicates = metric_table.loc[metric_table["model"].duplicated(), "model"].tolist()
        raise SCREMetricError(f"Duplicate SCRE-Credit model rows: {duplicates}")

    validated = metric_table.copy()
    for column in [*HIGHER_IS_BETTER, *LOWER_IS_BETTER]:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
        if validated[column].isna().any():
            bad_models = validated.loc[validated[column].isna(), "model"].tolist()
            raise SCREMetricError(f"Metric {column!r} is missing or non-numeric for models: {bad_models}")
    return validated


def normalize_higher_better(values: pd.Series) -> pd.Series:
    """Min-max normalize a higher-is-better metric to [0, 1]."""

    series = pd.Series(values, dtype=float)
    value_range = float(series.max() - series.min())
    if not np.isfinite(value_range) or value_range == 0.0:
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    return (series - series.min()) / value_range


def normalize_lower_better(values: pd.Series) -> pd.Series:
    """Min-max normalize a lower-is-better metric to [0, 1]."""

    series = pd.Series(values, dtype=float)
    value_range = float(series.max() - series.min())
    if not np.isfinite(value_range) or value_range == 0.0:
        return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    return 1.0 - ((series - series.min()) / value_range)


def build_scre_weight_table(
    metric_table: pd.DataFrame,
    component_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute normalized metrics, reliability score R_m, and ensemble weight w_m."""

    weights = component_weights or SCRE_COMPONENT_WEIGHTS
    missing_weight_keys = sorted(set(SCRE_COMPONENT_WEIGHTS) - set(weights))
    if missing_weight_keys:
        raise SCREMetricError(f"Missing SCRE-Credit component weights: {missing_weight_keys}")

    table = validate_metric_table(metric_table)
    result = table.copy()
    for column in HIGHER_IS_BETTER:
        result[f"{column}_norm"] = normalize_higher_better(result[column])
    for column in LOWER_IS_BETTER:
        result[f"{column}_norm"] = normalize_lower_better(result[column])

    result["reliability_score"] = (
        weights["pr_auc"] * result["pr_auc_norm"]
        + weights["roc_auc"] * result["roc_auc_norm"]
        + weights["recall"] * result["recall_norm"]
        + weights["calibration_error"] * result["calibration_error_norm"]
        + weights["expected_cost"] * result["expected_cost_norm"]
        + weights["stability_score"] * result["stability_score_norm"]
        + weights["faithfulness_score"] * result["faithfulness_score_norm"]
    )
    reliability_sum = float(result["reliability_score"].sum())
    if reliability_sum <= 0 or not np.isfinite(reliability_sum):
        result["ensemble_weight"] = 1.0 / len(result)
    else:
        result["ensemble_weight"] = result["reliability_score"] / reliability_sum
    return result.sort_values("ensemble_weight", ascending=False).reset_index(drop=True)


def validate_weight_table(weight_table: pd.DataFrame) -> pd.DataFrame:
    """Validate a fixed model-weight table for SCRE-style ensembling."""

    missing = [column for column in ["model", "ensemble_weight"] if column not in weight_table.columns]
    if missing:
        raise SCREMetricError(f"Missing ensemble weight columns: {missing}")
    if weight_table["model"].duplicated().any():
        duplicates = weight_table.loc[weight_table["model"].duplicated(), "model"].tolist()
        raise SCREMetricError(f"Duplicate ensemble weight rows: {duplicates}")

    table = weight_table.copy()
    table["ensemble_weight"] = pd.to_numeric(table["ensemble_weight"], errors="coerce")
    if table["ensemble_weight"].isna().any():
        bad_models = table.loc[table["ensemble_weight"].isna(), "model"].tolist()
        raise SCREMetricError(f"Ensemble weight missing or non-numeric for models: {bad_models}")
    if (table["ensemble_weight"] < 0).any():
        bad_models = table.loc[table["ensemble_weight"] < 0, "model"].tolist()
        raise SCREMetricError(f"Negative ensemble weights are not allowed: {bad_models}")

    weight_sum = float(table["ensemble_weight"].sum())
    if weight_sum <= 0 or not np.isfinite(weight_sum):
        raise SCREMetricError("Ensemble weights must have a positive finite sum.")
    table["ensemble_weight"] = table["ensemble_weight"] / weight_sum
    return table.sort_values("ensemble_weight", ascending=False).reset_index(drop=True)


def build_scre_pareto_pool(
    metric_table: pd.DataFrame,
    higher_is_better: list[str] | None = None,
    lower_is_better: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> pd.DataFrame:
    """Return validation non-dominated SCRE candidate models.

    Dominance is evaluated only from validation-derived model metrics. Diagnostic
    models such as a majority baseline can be excluded explicitly before the
    Pareto calculation.
    """

    higher = higher_is_better or PARETO_HIGHER_IS_BETTER
    lower = lower_is_better or PARETO_LOWER_IS_BETTER
    required = ["model", *higher, *lower]
    missing = [column for column in required if column not in metric_table.columns]
    if missing:
        raise SCREMetricError(f"Missing Pareto metric columns: {missing}")

    table = metric_table.copy()
    if exclude_models:
        table = table.loc[~table["model"].isin(exclude_models)].copy()
    if table.empty:
        raise SCREMetricError("No models remain after Pareto exclusion filters.")

    for column in [*higher, *lower]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if table[column].isna().any():
            bad_models = table.loc[table[column].isna(), "model"].tolist()
            raise SCREMetricError(f"Pareto metric {column!r} is missing for models: {bad_models}")

    objective = pd.DataFrame(index=table.index)
    for column in higher:
        objective[column] = table[column].astype(float)
    for column in lower:
        objective[column] = -table[column].astype(float)

    values = objective.to_numpy(dtype=float)
    is_efficient = np.ones(len(table), dtype=bool)
    for i in range(len(table)):
        if not is_efficient[i]:
            continue
        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            is_efficient[i] = False

    result = table.loc[is_efficient].copy()
    result["pareto_selected"] = True
    result["pareto_objectives_higher_is_better"] = ", ".join(higher)
    result["pareto_objectives_lower_is_better"] = ", ".join(lower)
    result["selection_split"] = "validation"
    result["test_set_used_for_selection"] = False
    return result.sort_values(["expected_cost", "pr_auc"], ascending=[True, False]).reset_index(drop=True)


def _minimum_cost_threshold_fast(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> dict[str, float | int]:
    """Vectorized minimum expected-cost threshold search."""

    y_array = np.asarray(y_true).astype(int)
    p_array = np.asarray(y_proba, dtype=float)
    predictions = p_array[:, None] >= thresholds[None, :]
    positives = y_array[:, None] == 1
    negatives = ~positives
    fp = np.sum(predictions & negatives, axis=0)
    fn = np.sum((~predictions) & positives, axis=0)
    cost = fn_cost * fn + fp_cost * fp
    best_index = int(np.lexsort((thresholds, cost))[0])
    return {
        "threshold": float(thresholds[best_index]),
        "expected_cost": float(cost[best_index]),
        "fp": int(fp[best_index]),
        "fn": int(fn[best_index]),
    }


def optimize_validation_ensemble_weights(
    probability_frame: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    metric_table: pd.DataFrame,
    initial_weight_table: pd.DataFrame | None = None,
    thresholds: np.ndarray | None = None,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    random_seed: int = 42,
    n_random_candidates: int = 3000,
    n_top_candidates_for_calibration: int = 75,
    final_calibration_method: str = "isotonic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize non-negative ensemble weights on validation probabilities only.

    The search is deterministic: it evaluates equal, single-model, optional
    static SCRE weights, and Dirichlet random portfolios. The final selected
    weight vector is chosen after validation-only final calibration and
    threshold selection.
    """

    probabilities = validate_probability_frame(probability_frame)
    y_array = np.asarray(y_true).astype(int)
    if len(probabilities) != len(y_array):
        raise ValueError("probability_frame and y_true must have the same number of rows.")
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    model_names = list(probabilities.columns)
    n_models = len(model_names)
    if n_models == 0:
        raise ValueError("At least one model is required for optimized weights.")
    matrix = probabilities.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)

    candidate_weights: list[np.ndarray] = []
    candidate_sources: list[str] = []
    candidate_weights.append(np.full(n_models, 1.0 / n_models))
    candidate_sources.append("equal_weight")
    for index, model_name in enumerate(model_names):
        one_hot = np.zeros(n_models)
        one_hot[index] = 1.0
        candidate_weights.append(one_hot)
        candidate_sources.append(f"single_model:{model_name}")

    if initial_weight_table is not None and not initial_weight_table.empty:
        validated_initial = validate_weight_table(initial_weight_table).set_index("model")
        if set(model_names).issubset(validated_initial.index):
            static_weights = validated_initial.loc[model_names, "ensemble_weight"].to_numpy(dtype=float)
            candidate_weights.append(static_weights / static_weights.sum())
            candidate_sources.append("initial_scre_weight")

    random_weights = rng.dirichlet(np.ones(n_models), size=max(0, int(n_random_candidates)))
    for weight in random_weights:
        candidate_weights.append(weight)
        candidate_sources.append("dirichlet_random")

    raw_rows: list[dict[str, float | int | str]] = []
    for candidate_id, (weight, source) in enumerate(zip(candidate_weights, candidate_sources), start=1):
        raw_probability = np.clip(matrix @ weight, 0.0, 1.0)
        cost_summary = _minimum_cost_threshold_fast(
            y_array,
            raw_probability,
            thresholds=thresholds,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        raw_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_source": source,
                "raw_threshold": cost_summary["threshold"],
                "raw_expected_cost": cost_summary["expected_cost"],
                "raw_pr_auc": float(average_precision_score(y_array, raw_probability)),
                "raw_roc_auc": float(roc_auc_score(y_array, raw_probability)),
                "raw_brier_score": float(brier_score_loss(y_array, raw_probability)),
                "weights_json": dict(zip(model_names, [float(value) for value in weight])).__repr__(),
            }
        )

    raw_table = pd.DataFrame(raw_rows).sort_values(
        ["raw_expected_cost", "raw_pr_auc", "raw_roc_auc"],
        ascending=[True, False, False],
    )
    top_candidate_ids = raw_table.head(max(1, int(n_top_candidates_for_calibration)))["candidate_id"].tolist()

    calibrated_rows: list[dict[str, float | int | str]] = []
    for candidate_id in top_candidate_ids:
        weight = candidate_weights[candidate_id - 1]
        raw_probability = np.clip(matrix @ weight, 0.0, 1.0)
        calibrator = fit_final_calibrator(raw_probability, y_array, method=final_calibration_method)
        calibrated_probability = apply_final_calibrator(calibrator, raw_probability)
        threshold_summary = threshold_cost_table(
            y_array,
            calibrated_probability,
            thresholds=thresholds,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        ).sort_values(["expected_cost", "threshold"]).iloc[0]
        calibrated_rows.append(
            {
                "candidate_id": int(candidate_id),
                "calibrated_threshold": float(threshold_summary["threshold"]),
                "calibrated_expected_cost": float(threshold_summary["expected_cost"]),
                "calibrated_pr_auc": float(average_precision_score(y_array, calibrated_probability)),
                "calibrated_roc_auc": float(roc_auc_score(y_array, calibrated_probability)),
                "calibrated_brier_score": float(brier_score_loss(y_array, calibrated_probability)),
            }
        )

    audit = raw_table.merge(pd.DataFrame(calibrated_rows), on="candidate_id", how="left")
    best = audit.dropna(subset=["calibrated_expected_cost"]).sort_values(
        ["calibrated_expected_cost", "raw_expected_cost", "calibrated_pr_auc", "calibrated_roc_auc"],
        ascending=[True, True, False, False],
    ).iloc[0]
    best_weights = candidate_weights[int(best["candidate_id"]) - 1]

    metrics = metric_table.set_index("model").loc[model_names].reset_index()
    optimized = metrics.copy()
    optimized["ensemble_weight"] = best_weights / best_weights.sum()
    optimized["weight_source"] = "validation_optimized"
    optimized["optimization_objective"] = "minimize_validation_expected_cost_after_final_calibration"
    optimized["optimization_split"] = "validation"
    optimized["test_set_used_for_weight_optimization"] = False
    optimized["n_random_candidates"] = int(n_random_candidates)
    optimized["n_top_candidates_for_calibration"] = int(n_top_candidates_for_calibration)
    optimized["selected_candidate_id"] = int(best["candidate_id"])
    optimized["selected_candidate_source"] = str(best["candidate_source"])
    return validate_weight_table(optimized), audit.reset_index(drop=True)


def validate_probability_frame(probability_frame: pd.DataFrame) -> pd.DataFrame:
    """Validate that all supplied base probabilities are numeric and in [0, 1]."""

    probabilities = probability_frame.copy()
    if probabilities.empty:
        raise ValueError("probability_frame must contain at least one model column.")
    for column in probabilities.columns:
        probabilities[column] = pd.to_numeric(probabilities[column], errors="coerce")
    if probabilities.isna().any().any():
        bad_columns = probabilities.columns[probabilities.isna().any()].tolist()
        raise ValueError(f"Probability columns contain missing or non-numeric values: {bad_columns}")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any().any():
        bad_columns = probabilities.columns[((probabilities < 0.0) | (probabilities > 1.0)).any()].tolist()
        raise ValueError(f"Probability columns outside [0, 1]: {bad_columns}")
    return probabilities


def weighted_probability_sum(probability_frame: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    """Return P_final = sum_m w_m * P_m before final calibration."""

    ordered_weights = weights.loc[probability_frame.columns].to_numpy(dtype=float)
    probability = probability_frame.to_numpy(dtype=float) @ ordered_weights
    return np.clip(probability, 0.0, 1.0)


def fit_final_calibrator(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
):
    """Fit the final one-dimensional probability calibrator."""

    y_proba = np.asarray(y_proba, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    if method == "none":
        return None
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(y_proba, y_true)
        return calibrator
    if method == "sigmoid":
        clipped = np.clip(y_proba, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(logits, y_true)
        return calibrator
    raise ValueError("final_calibration_method must be 'isotonic', 'sigmoid', or 'none'.")


def apply_final_calibrator(calibrator, y_proba: np.ndarray) -> np.ndarray:
    """Apply a one-dimensional final calibrator and clip to [0, 1]."""

    y_proba = np.asarray(y_proba, dtype=float)
    if calibrator is None:
        return np.clip(y_proba, 0.0, 1.0)
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(y_proba), 0.0, 1.0)

    clipped = np.clip(y_proba, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return np.clip(calibrator.predict_proba(logits)[:, 1], 0.0, 1.0)


def threshold_cost_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate expected cost over thresholds from 0.01 to 0.99 by default."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        expected_cost = fn_cost * fn + fp_cost * fp
        rows.append(
            {
                "threshold": float(threshold),
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "tp": tp,
                "fn_cost": float(fn_cost),
                "fp_cost": float(fp_cost),
                "expected_cost": float(expected_cost),
            }
        )
    return pd.DataFrame(rows)
