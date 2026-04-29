"""Rebuild the final statistical-test table for SCRE revision comparisons."""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import expected_calibration_error
from heloc_preprocessing import HELOC_TARGET
from run_calibration_analysis import _positive_class_probability
from src.config.paths import (
    EXPERIMENT_LOGS_DIR,
    HELOC_MODEL_READY,
    MODELS_DIR,
    TABLES_DIR,
    TAIWAN_MODEL_READY,
)
from src.utils.logging_utils import create_experiment_logger


N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
FN_COST = 5.0
FP_COST = 1.0
PRIMARY_COST_SCENARIO = "B_FN5_FP1"

METRIC_DIRECTIONS = {
    "ROC-AUC": "higher_is_better",
    "PR-AUC": "higher_is_better",
    "F1": "higher_is_better",
    "Recall": "higher_is_better",
    "Brier": "lower_is_better",
    "ECE": "lower_is_better",
    "Expected cost": "lower_is_better",
}

DATASET_CONFIGS = {
    "taiwan": {"path": TAIWAN_MODEL_READY, "target": TARGET_COLUMN},
    "heloc": {"path": HELOC_MODEL_READY, "target": HELOC_TARGET},
}

MODEL_SOURCE_CANDIDATES = {
    "Best CatBoost": ["catboost", "optuna_best_catboost"],
    "Best XGBoost": ["xgboost", "optuna_best_xgboost", "smotenc_xgboost"],
    "Best Monotonic model": ["monotonic_xgboost", "monotonic_lightgbm"],
    "Best Scorecard": ["woe_scorecard_logistic_regression"],
}

COMPARISONS = {
    "taiwan": [
        ("Best CatBoost", "Old SCRE-Credit"),
        ("Best CatBoost", "SCRE-Pareto"),
        ("Best CatBoost", "SCRE-Optimized"),
        ("Best XGBoost", "SCRE-Pareto"),
        ("Best Scorecard", "SCRE-Pareto"),
        ("Best CatBoost", "Best Scorecard"),
        ("Best CatBoost", "Best Monotonic model"),
    ],
    "heloc": [
        ("Best CatBoost", "Old SCRE-Credit"),
        ("Best CatBoost", "SCRE-Pareto"),
        ("Best CatBoost", "SCRE-Optimized"),
        ("Best Scorecard", "SCRE-Pareto"),
        ("Best Scorecard", "SCRE-Optimized"),
    ],
}


@dataclass(frozen=True)
class PredictionBundle:
    """Predicted probabilities and the validation-selected threshold for one model."""

    display_model: str
    source_model: str
    y_proba: np.ndarray
    threshold: float
    calibration_type: str


def stable_seed(*parts: str, base_seed: int = RANDOM_SEED) -> int:
    """Create a deterministic 32-bit seed from labels and the central seed."""

    payload = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def load_test_split(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the reserved test split for a dataset."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_test = split.test.drop(columns=[config["target"]]).reset_index(drop=True)
    y_test = split.test[config["target"]].astype(int).reset_index(drop=True)
    return X_test, y_test


def select_source_model(dataset: str, display_model: str, weights: pd.DataFrame) -> str:
    """Select the source model for a requested display family using validation cost."""

    if display_model not in MODEL_SOURCE_CANDIDATES:
        raise ValueError(f"No source candidates configured for {display_model!r}.")
    candidates = MODEL_SOURCE_CANDIDATES[display_model]
    dataset_weights = weights.loc[(weights["dataset"] == dataset) & (weights["model"].isin(candidates))].copy()
    if dataset_weights.empty:
        raise ValueError(f"No calibrated source model rows for {dataset}/{display_model}.")
    dataset_weights = dataset_weights.sort_values(
        ["expected_cost", "pr_auc", "roc_auc", "brier_score", "ece"],
        ascending=[True, False, False, True, True],
    )
    return str(dataset_weights.iloc[0]["model"])


def load_base_probabilities(
    dataset: str,
    X_test: pd.DataFrame,
    model_names: list[str],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Load calibrated base probabilities, with Old SCRE available as a component."""

    dataset_weights = weights.loc[weights["dataset"] == dataset].set_index("model")
    probabilities: dict[str, np.ndarray] = {}
    non_scre = [model for model in model_names if model != "SCRE-Credit"]
    for model in non_scre:
        if model not in dataset_weights.index:
            raise KeyError(f"Missing calibrated artifact metadata for {dataset}/{model}.")
        model_path = Path(dataset_weights.loc[model, "model_path"])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        fitted_model = joblib.load(model_path)
        probabilities[model] = _positive_class_probability(fitted_model, X_test)

    if "SCRE-Credit" in model_names:
        scre_path = MODELS_DIR / "scre_credit_model.joblib"
        if not scre_path.exists():
            raise FileNotFoundError(scre_path)
        bundle = joblib.load(scre_path)
        classifier = bundle[dataset]["classifier"]
        base = load_base_probabilities(dataset, X_test, classifier.model_names_, weights)
        probabilities["SCRE-Credit"] = classifier.predict_proba_from_base(base[classifier.model_names_])[:, 1]

    return pd.DataFrame({model: probabilities[model] for model in model_names}, index=X_test.index)


def load_old_scre_prediction(dataset: str, X_test: pd.DataFrame, weights: pd.DataFrame) -> PredictionBundle:
    """Load Old SCRE-Credit probabilities from its saved ensemble bundle."""

    scre_path = MODELS_DIR / "scre_credit_model.joblib"
    if not scre_path.exists():
        raise FileNotFoundError(scre_path)
    bundle = joblib.load(scre_path)
    classifier = bundle[dataset]["classifier"]
    base = load_base_probabilities(dataset, X_test, classifier.model_names_, weights)
    y_proba = classifier.predict_proba_from_base(base[classifier.model_names_])[:, 1]
    return PredictionBundle(
        display_model="Old SCRE-Credit",
        source_model="SCRE-Credit",
        y_proba=y_proba,
        threshold=float(classifier.threshold_),
        calibration_type="weighted_sum_plus_isotonic",
    )


def load_revision_prediction(
    dataset: str,
    variant: str,
    X_test: pd.DataFrame,
    weights: pd.DataFrame,
) -> PredictionBundle:
    """Load SCRE-Pareto or SCRE-Optimized probabilities from saved weights."""

    if variant == "SCRE-Pareto":
        path = MODELS_DIR / f"scre_pareto_{dataset}.joblib"
        artifact = joblib.load(path)
        model_names = artifact["model_names"]
        ensemble_weights = pd.Series(artifact["ensemble_weights"], dtype=float)
        threshold = float(artifact["threshold"])
        calibration_type = str(artifact.get("calibration_type", "weighted_sum"))
    elif variant == "SCRE-Optimized":
        path = MODELS_DIR / f"scre_optimized_{dataset}.joblib"
        artifact = joblib.load(path)
        model_names = artifact["model_names"]
        ensemble_weights = pd.Series(artifact["selected_weights"], dtype=float)
        threshold = float(artifact["selected_threshold"])
        calibration_type = f"validation_optimized_weighted_sum_{artifact['selected_setting']}"
    else:
        raise ValueError(f"Unknown SCRE revision variant: {variant}")

    if not path.exists():
        raise FileNotFoundError(path)
    base = load_base_probabilities(dataset, X_test, model_names, weights)
    ordered_weights = ensemble_weights.loc[model_names].to_numpy(dtype=float)
    ordered_weights = ordered_weights / ordered_weights.sum()
    y_proba = np.clip(base[model_names].to_numpy(dtype=float) @ ordered_weights, 0.0, 1.0)
    return PredictionBundle(
        display_model=variant,
        source_model=variant,
        y_proba=y_proba,
        threshold=threshold,
        calibration_type=calibration_type,
    )


def load_single_model_prediction(
    dataset: str,
    display_model: str,
    X_test: pd.DataFrame,
    weights: pd.DataFrame,
) -> PredictionBundle:
    """Load one validation-selected calibrated single model."""

    source_model = select_source_model(dataset, display_model, weights)
    row = weights.loc[(weights["dataset"] == dataset) & (weights["model"] == source_model)].iloc[0]
    model_path = Path(row["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    fitted_model = joblib.load(model_path)
    return PredictionBundle(
        display_model=display_model,
        source_model=source_model,
        y_proba=_positive_class_probability(fitted_model, X_test),
        threshold=float(row["base_selected_threshold"]),
        calibration_type=str(row["selected_calibration_method"]),
    )


def build_predictions(dataset: str) -> tuple[np.ndarray, dict[str, PredictionBundle]]:
    """Build all prediction bundles needed for a dataset's requested comparisons."""

    X_test, y_test = load_test_split(dataset)
    weights = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
    required_models = sorted({model for pair in COMPARISONS[dataset] for model in pair})
    predictions: dict[str, PredictionBundle] = {}
    for display_model in required_models:
        if display_model == "Old SCRE-Credit":
            predictions[display_model] = load_old_scre_prediction(dataset, X_test, weights)
        elif display_model in {"SCRE-Pareto", "SCRE-Optimized"}:
            predictions[display_model] = load_revision_prediction(dataset, display_model, X_test, weights)
        else:
            predictions[display_model] = load_single_model_prediction(dataset, display_model, X_test, weights)
    return y_test.to_numpy(dtype=int), predictions


def metric_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    metric: str,
) -> float:
    """Compute one requested metric."""

    y_pred = (y_proba >= threshold).astype(int)
    if metric == "ROC-AUC":
        return float(roc_auc_score(y_true, y_proba))
    if metric == "PR-AUC":
        return float(average_precision_score(y_true, y_proba))
    if metric == "F1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "Recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "Brier":
        return float(brier_score_loss(y_true, y_proba))
    if metric == "ECE":
        return float(expected_calibration_error(y_true, y_proba))
    if metric == "Expected cost":
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return float(FN_COST * fn + FP_COST * fp)
    raise ValueError(f"Unsupported metric: {metric}")


def paired_bootstrap_difference(
    y_true: np.ndarray,
    baseline: PredictionBundle,
    candidate: PredictionBundle,
    metric: str,
    random_seed: int,
) -> tuple[float, float, float, float]:
    """Return mean, lower CI, upper CI, and two-sided p-value for candidate-baseline."""

    rng = np.random.default_rng(random_seed)
    n_rows = len(y_true)
    differences: list[float] = []
    attempts = 0
    max_attempts = N_BOOTSTRAP * 5
    while len(differences) < N_BOOTSTRAP and attempts < max_attempts:
        indices = rng.integers(0, n_rows, size=n_rows)
        y_sample = y_true[indices]
        if metric in {"ROC-AUC", "PR-AUC"} and len(np.unique(y_sample)) < 2:
            attempts += 1
            continue
        baseline_value = metric_value(
            y_sample,
            baseline.y_proba[indices],
            baseline.threshold,
            metric,
        )
        candidate_value = metric_value(
            y_sample,
            candidate.y_proba[indices],
            candidate.threshold,
            metric,
        )
        if np.isfinite(baseline_value) and np.isfinite(candidate_value):
            differences.append(float(candidate_value - baseline_value))
        attempts += 1

    if not differences:
        return np.nan, np.nan, np.nan, np.nan
    values = np.asarray(differences, dtype=float)
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    p_value = float(2.0 * min(np.mean(values <= 0.0), np.mean(values >= 0.0)))
    return float(values.mean()), lower, upper, min(p_value, 1.0)


def directional_winner(
    baseline_model: str,
    candidate_model: str,
    difference: float,
    better_direction: str,
) -> str:
    """Choose the directional winner from candidate-minus-baseline difference."""

    if not np.isfinite(difference) or np.isclose(difference, 0.0):
        return "tie"
    if better_direction == "higher_is_better":
        return candidate_model if difference > 0 else baseline_model
    if better_direction == "lower_is_better":
        return candidate_model if difference < 0 else baseline_model
    return "undetermined"


def interpretation_text(
    metric: str,
    baseline_model: str,
    candidate_model: str,
    winner: str,
    p_value: float,
    significant: bool,
    difference: float,
) -> str:
    """Create a clear interpretation for one statistical-test row."""

    if pd.isna(p_value):
        return (
            f"test not available for {metric}; values are reported but no p-value could be computed. "
            f"Baseline={baseline_model}, candidate={candidate_model}."
        )
    if winner == "tie":
        direction_sentence = "Observed values are tied within numerical tolerance"
    else:
        direction_sentence = f"{winner} has the favorable observed direction"
    significance_sentence = "statistically significant at alpha=0.05" if significant else "not statistically significant at alpha=0.05"
    return (
        f"{direction_sentence} for {metric} "
        f"(candidate minus baseline={difference:.6g}); paired bootstrap p={p_value:.4g}, {significance_sentence}. "
        "Significance indicates a directional difference, not automatic SCRE superiority."
    )


def comparison_rows_for_dataset(dataset: str, logger) -> list[dict[str, object]]:
    """Compute all requested cleaned statistical-test rows for one dataset."""

    y_true, predictions = build_predictions(dataset)
    rows: list[dict[str, object]] = []
    for baseline_name, candidate_name in COMPARISONS[dataset]:
        baseline = predictions[baseline_name]
        candidate = predictions[candidate_name]
        logger.info("Testing %s: %s vs %s", dataset, baseline_name, candidate_name)
        for metric, better_direction in METRIC_DIRECTIONS.items():
            baseline_value = metric_value(y_true, baseline.y_proba, baseline.threshold, metric)
            candidate_value = metric_value(y_true, candidate.y_proba, candidate.threshold, metric)
            difference = float(candidate_value - baseline_value)
            _, ci_lower, ci_upper, p_value = paired_bootstrap_difference(
                y_true=y_true,
                baseline=baseline,
                candidate=candidate,
                metric=metric,
                random_seed=stable_seed(dataset, baseline_name, candidate_name, metric),
            )
            significant = bool(pd.notna(p_value) and p_value < 0.05)
            winner = directional_winner(baseline_name, candidate_name, difference, better_direction)
            rows.append(
                {
                    "dataset": dataset,
                    "test_name": "paired_bootstrap",
                    "metric": metric,
                    "baseline_model": baseline_name,
                    "candidate_model": candidate_name,
                    "baseline_value": baseline_value,
                    "candidate_value": candidate_value,
                    "difference_candidate_minus_baseline": difference,
                    "p_value": p_value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "significant_0_05": significant,
                    "better_direction": better_direction,
                    "winner": winner,
                    "interpretation": interpretation_text(
                        metric=metric,
                        baseline_model=baseline_name,
                        candidate_model=candidate_name,
                        winner=winner,
                        p_value=p_value,
                        significant=significant,
                        difference=difference,
                    ),
                    "baseline_source_model": baseline.source_model,
                    "candidate_source_model": candidate.source_model,
                    "baseline_threshold": baseline.threshold,
                    "candidate_threshold": candidate.threshold,
                    "baseline_calibration_type": baseline.calibration_type,
                    "candidate_calibration_type": candidate.calibration_type,
                    "cost_scenario": PRIMARY_COST_SCENARIO,
                    "split": "test",
                    "n_bootstrap": N_BOOTSTRAP,
                    "confidence_level": CONFIDENCE_LEVEL,
                }
            )
    return rows


def build_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Summarize comparison-level statistical outcomes."""

    rows = []
    for (dataset, baseline, candidate), group in cleaned.groupby(
        ["dataset", "baseline_model", "candidate_model"], sort=False
    ):
        significant = group.loc[group["significant_0_05"].astype(bool)]
        expected_cost_row = group.loc[group["metric"] == "Expected cost"].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "baseline_model": baseline,
                "candidate_model": candidate,
                "n_metrics": int(len(group)),
                "n_significant_0_05": int(len(significant)),
                "baseline_directional_wins": int((group["winner"] == baseline).sum()),
                "candidate_directional_wins": int((group["winner"] == candidate).sum()),
                "significant_baseline_wins": int((significant["winner"] == baseline).sum()),
                "significant_candidate_wins": int((significant["winner"] == candidate).sum()),
                "expected_cost_winner": expected_cost_row["winner"],
                "expected_cost_difference_candidate_minus_baseline": float(
                    expected_cost_row["difference_candidate_minus_baseline"]
                ),
                "expected_cost_p_value": float(expected_cost_row["p_value"])
                if pd.notna(expected_cost_row["p_value"])
                else np.nan,
                "recommendation": (
                    f"Under FN=5, FP=1, {expected_cost_row['winner']} has lower observed expected cost. "
                    "Use metric-level rows for significance and tradeoff details."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Create cleaned statistical-test and summary tables."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger(
        "statistical_tests_cleaned",
        EXPERIMENT_LOGS_DIR / "25_statistical_tests_cleaned.log",
    )
    rows = []
    for dataset in ["taiwan", "heloc"]:
        rows.extend(comparison_rows_for_dataset(dataset, logger))

    required_columns = [
        "dataset",
        "test_name",
        "metric",
        "baseline_model",
        "candidate_model",
        "baseline_value",
        "candidate_value",
        "difference_candidate_minus_baseline",
        "p_value",
        "ci_lower",
        "ci_upper",
        "significant_0_05",
        "better_direction",
        "winner",
        "interpretation",
    ]
    cleaned = pd.DataFrame(rows)
    if cleaned[["baseline_model", "candidate_model"]].isna().any().any():
        raise ValueError("baseline_model/candidate_model cannot be empty.")
    cleaned = cleaned[
        required_columns
        + [
            "baseline_source_model",
            "candidate_source_model",
            "baseline_threshold",
            "candidate_threshold",
            "baseline_calibration_type",
            "candidate_calibration_type",
            "cost_scenario",
            "split",
            "n_bootstrap",
            "confidence_level",
        ]
    ]
    summary = build_summary(cleaned)

    cleaned_path = TABLES_DIR / "statistical_tests_cleaned.csv"
    summary_path = TABLES_DIR / "statistical_tests_summary.csv"
    cleaned.to_csv(cleaned_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Saved %s", cleaned_path)
    logger.info("Saved %s", summary_path)
    print(cleaned[required_columns].head(20).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
