"""Train SCRE-Optimized with validation-only constrained ensemble weights."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics, expected_calibration_error
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


PRIMARY_COST_SCENARIO = "B_FN5_FP1"
FN_COST = 5.0
FP_COST = 1.0
MAX_MODEL_WEIGHT = 0.65
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)

OBJECTIVE_SETTINGS = {
    "A_cost_heavy": {
        "alpha": 100.0,
        "beta": 100.0,
        "gamma": 100.0,
        "delta": 25.0,
        "description": "cost-heavy",
    },
    "B_pr_auc_heavy": {
        "alpha": 100.0,
        "beta": 100.0,
        "gamma": 1200.0,
        "delta": 50.0,
        "description": "PR-AUC-heavy",
    },
    "C_calibration_heavy": {
        "alpha": 1000.0,
        "beta": 1000.0,
        "gamma": 100.0,
        "delta": 25.0,
        "description": "calibration-heavy",
    },
    "D_balanced": {
        "alpha": 200.0,
        "beta": 200.0,
        "gamma": 200.0,
        "delta": 50.0,
        "description": "balanced; uses the prompt's starting coefficients",
    },
}

DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "pareto_front": TABLES_DIR / "pareto_front_taiwan.csv",
        "weights_output": TABLES_DIR / "scre_optimized_weights_taiwan.csv",
        "results_output": TABLES_DIR / "scre_optimized_results_taiwan.csv",
        "model_output": MODELS_DIR / "scre_optimized_taiwan.joblib",
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "pareto_front": TABLES_DIR / "pareto_front_heloc.csv",
        "weights_output": TABLES_DIR / "scre_optimized_weights_heloc.csv",
        "results_output": TABLES_DIR / "scre_optimized_results_heloc.csv",
        "model_output": MODELS_DIR / "scre_optimized_heloc.joblib",
    },
}


@dataclass(frozen=True)
class OptimizedSettingResult:
    """Validation result for one SCRE-Optimized objective setting."""

    setting_id: str
    weights: np.ndarray
    metrics: dict[str, float | int | str | bool]
    n_candidates_evaluated: int
    best_candidate_source: str


def load_validation_test(dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load deterministic validation and test splits for a dataset."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_validation = split.validation.drop(columns=[config["target"]]).reset_index(drop=True)
    y_validation = split.validation[config["target"]].astype(int).reset_index(drop=True)
    X_test = split.test.drop(columns=[config["target"]]).reset_index(drop=True)
    y_test = split.test[config["target"]].astype(int).reset_index(drop=True)
    return X_validation, y_validation, X_test, y_test


def load_pareto_front_models(dataset: str) -> list[str]:
    """Return Pareto-front model names selected on validation metrics."""

    front = pd.read_csv(DATASET_CONFIGS[dataset]["pareto_front"])
    if "is_pareto_front" in front.columns:
        front = front.loc[front["is_pareto_front"].astype(bool)].copy()
    if front.empty:
        raise ValueError(f"No Pareto-front models found for {dataset}.")
    if front["model"].duplicated().any():
        duplicates = front.loc[front["model"].duplicated(), "model"].tolist()
        raise ValueError(f"Duplicate Pareto-front model names for {dataset}: {duplicates}")
    return front["model"].astype(str).tolist()


def load_base_model_probabilities(dataset: str, X: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Load calibrated base-model probabilities, including Old SCRE-Credit."""

    metadata = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
    dataset_metadata = metadata.loc[metadata["dataset"] == dataset].set_index("model")
    probabilities: dict[str, np.ndarray] = {}

    non_scre_models = [model for model in model_names if model != "SCRE-Credit"]
    for model in non_scre_models:
        if model not in dataset_metadata.index:
            raise KeyError(f"Missing calibrated model metadata for {dataset}/{model}.")
        model_path = Path(dataset_metadata.loc[model, "model_path"])
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        fitted_model = joblib.load(model_path)
        probabilities[model] = _positive_class_probability(fitted_model, X)

    if "SCRE-Credit" in model_names:
        bundle_path = MODELS_DIR / "scre_credit_model.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(bundle_path)
        bundle = joblib.load(bundle_path)
        classifier = bundle[dataset]["classifier"]
        base_probabilities = load_base_model_probabilities(dataset, X, classifier.model_names_)
        probabilities["SCRE-Credit"] = classifier.predict_proba_from_base(
            base_probabilities[classifier.model_names_]
        )[:, 1]

    return pd.DataFrame({model: probabilities[model] for model in model_names}, index=X.index)


def project_to_capped_simplex(values: np.ndarray, max_weight: float = MAX_MODEL_WEIGHT) -> np.ndarray:
    """Project a vector onto the simplex with an upper bound per coordinate."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    n_values = len(vector)
    if n_values == 0:
        raise ValueError("At least one weight is required.")
    if max_weight * n_values < 1.0 - 1e-12:
        raise ValueError("max_weight is too small to allow weights to sum to one.")

    lower = float(np.min(vector) - max_weight)
    upper = float(np.max(vector))
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        projected = np.clip(vector - midpoint, 0.0, max_weight)
        if projected.sum() > 1.0:
            lower = midpoint
        else:
            upper = midpoint

    projected = np.clip(vector - upper, 0.0, max_weight)
    if projected.sum() <= 0.0:
        projected = np.full(n_values, 1.0 / n_values)
    else:
        projected = projected / projected.sum()
    if projected.max() > max_weight + 1e-8:
        projected = np.minimum(projected, max_weight)
        projected = project_to_capped_simplex(projected, max_weight=max_weight)
    return projected


def make_anchor_weight(n_models: int, primary_index: int, max_weight: float) -> np.ndarray:
    """Create a capped single-model anchor portfolio."""

    if n_models == 1:
        return np.ones(1)
    weights = np.full(n_models, (1.0 - max_weight) / (n_models - 1))
    weights[primary_index] = max_weight
    return weights


def make_candidate_weights(
    n_models: int,
    rng: np.random.Generator,
    initial_weights: np.ndarray | None = None,
    n_random_candidates: int = 5000,
    max_weight: float = MAX_MODEL_WEIGHT,
) -> tuple[list[np.ndarray], list[str]]:
    """Create deterministic and random capped-simplex starting candidates."""

    candidates: list[np.ndarray] = []
    sources: list[str] = []

    def add_candidate(weight: np.ndarray, source: str) -> None:
        projected = project_to_capped_simplex(weight, max_weight=max_weight)
        key = tuple(np.round(projected, 10))
        if key in seen:
            return
        seen.add(key)
        candidates.append(projected)
        sources.append(source)

    seen: set[tuple[float, ...]] = set()
    add_candidate(np.full(n_models, 1.0 / n_models), "equal_weight")
    if initial_weights is not None:
        add_candidate(initial_weights, "scre_pareto_weight_start")
    for index in range(n_models):
        add_candidate(make_anchor_weight(n_models, index, max_weight), f"capped_anchor_model_{index}")
    for left in range(n_models):
        for right in range(left + 1, n_models):
            pair = np.zeros(n_models)
            pair[left] = 0.5
            pair[right] = 0.5
            add_candidate(pair, f"pair_anchor_{left}_{right}")

    alpha_values = [0.35, 0.70, 1.00, 2.00, 5.00]
    per_alpha = int(np.ceil(n_random_candidates / len(alpha_values)))
    for alpha in alpha_values:
        draws = rng.dirichlet(np.full(n_models, alpha), size=per_alpha)
        for draw in draws:
            add_candidate(draw, f"dirichlet_alpha_{alpha:g}")
            if len(candidates) >= n_random_candidates + n_models + 60:
                break
        if len(candidates) >= n_random_candidates + n_models + 60:
            break

    return candidates, sources


def weighted_probability(probability_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return clipped weighted probability."""

    return np.clip(probability_matrix @ weights, 0.0, 1.0)


def minimum_cost_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict[str, float | int]:
    """Return minimum-cost threshold and confusion counts."""

    y_array = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_proba, dtype=float)
    predictions = probabilities[:, None] >= thresholds[None, :]
    positives = y_array[:, None] == 1
    negatives = ~positives
    tp = np.sum(predictions & positives, axis=0)
    tn = np.sum((~predictions) & negatives, axis=0)
    fp = np.sum(predictions & negatives, axis=0)
    fn = np.sum((~predictions) & positives, axis=0)
    costs = fn_cost * fn + fp_cost * fp
    best_index = int(np.lexsort((thresholds, costs))[0])
    return {
        "threshold": float(thresholds[best_index]),
        "expected_cost": float(costs[best_index]),
        "tn": int(tn[best_index]),
        "fp": int(fp[best_index]),
        "fn": int(fn[best_index]),
        "tp": int(tp[best_index]),
    }


def candidate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    setting: dict[str, float | str],
) -> dict[str, float | int]:
    """Compute validation objective terms for one candidate probability."""

    threshold_summary = minimum_cost_threshold(y_true, y_proba)
    threshold = float(threshold_summary["threshold"])
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))
    ece = float(expected_calibration_error(y_true, y_proba))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    tn = int(threshold_summary["tn"])
    fp = int(threshold_summary["fp"])
    fn = int(threshold_summary["fn"])
    tp = int(threshold_summary["tp"])
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    objective_value = (
        float(threshold_summary["expected_cost"])
        + float(setting["alpha"]) * ece
        + float(setting["beta"]) * brier
        - float(setting["gamma"]) * pr_auc
        - float(setting["delta"]) * recall
    )
    return {
        "objective_value": float(objective_value),
        "threshold": threshold,
        "expected_cost": float(threshold_summary["expected_cost"]),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece": ece,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def optimize_one_setting(
    setting_id: str,
    setting: dict[str, float | str],
    probability_frame: pd.DataFrame,
    y_true: pd.Series,
    initial_weights: np.ndarray | None,
    random_seed: int,
) -> OptimizedSettingResult:
    """Optimize capped non-negative ensemble weights for one objective setting."""

    model_names = probability_frame.columns.tolist()
    matrix = probability_frame.to_numpy(dtype=float)
    y_array = y_true.to_numpy(dtype=int)
    rng = np.random.default_rng(random_seed)
    candidates, sources = make_candidate_weights(
        n_models=len(model_names),
        rng=rng,
        initial_weights=initial_weights,
        n_random_candidates=5000,
        max_weight=MAX_MODEL_WEIGHT,
    )

    cache: dict[tuple[float, ...], dict[str, float | int]] = {}

    def evaluate_weights(weights: np.ndarray) -> dict[str, float | int]:
        projected = project_to_capped_simplex(weights, max_weight=MAX_MODEL_WEIGHT)
        key = tuple(np.round(projected, 10))
        if key not in cache:
            probability = weighted_probability(matrix, projected)
            cache[key] = candidate_metrics(y_array, probability, setting)
        return cache[key]

    scored_rows = []
    for candidate_id, (weights, source) in enumerate(zip(candidates, sources), start=1):
        metrics = evaluate_weights(weights)
        scored_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_source": source,
                "objective_value": metrics["objective_value"],
                "expected_cost": metrics["expected_cost"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "recall": metrics["recall"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
                "weights": weights,
            }
        )
    scored = pd.DataFrame(scored_rows).sort_values(
        ["objective_value", "expected_cost", "pr_auc", "ece"],
        ascending=[True, True, False, True],
    )

    bounds = [(0.0, MAX_MODEL_WEIGHT) for _ in model_names]
    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]

    def objective(weights: np.ndarray) -> float:
        if np.any(weights < -1e-8) or np.any(weights > MAX_MODEL_WEIGHT + 1e-8):
            return 1e12
        if abs(float(np.sum(weights)) - 1.0) > 1e-5:
            return 1e12 + 1e9 * abs(float(np.sum(weights)) - 1.0)
        return float(evaluate_weights(weights)["objective_value"])

    best_weights = np.asarray(scored.iloc[0]["weights"], dtype=float)
    best_metrics = evaluate_weights(best_weights)
    best_source = str(scored.iloc[0]["candidate_source"])
    n_local_starts = min(10, len(scored))
    for local_id, row in scored.head(n_local_starts).iterrows():
        start = np.asarray(row["weights"], dtype=float)
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 120, "ftol": 1e-7, "disp": False},
        )
        candidate = project_to_capped_simplex(result.x if result.success else start, max_weight=MAX_MODEL_WEIGHT)
        metrics = evaluate_weights(candidate)
        if (
            float(metrics["objective_value"]),
            float(metrics["expected_cost"]),
            -float(metrics["pr_auc"]),
            float(metrics["ece"]),
        ) < (
            float(best_metrics["objective_value"]),
            float(best_metrics["expected_cost"]),
            -float(best_metrics["pr_auc"]),
            float(best_metrics["ece"]),
        ):
            best_weights = candidate
            best_metrics = metrics
            best_source = f"slsqp_local_start_{int(row['candidate_id'])}"

    if not np.isclose(best_weights.sum(), 1.0):
        raise RuntimeError(f"{setting_id} weights do not sum to one.")
    if np.any(best_weights < -1e-8) or np.any(best_weights > MAX_MODEL_WEIGHT + 1e-8):
        raise RuntimeError(f"{setting_id} violates capped non-negative weight constraints.")

    metrics_with_setting = {
        **best_metrics,
        "setting_id": setting_id,
        "objective_description": str(setting["description"]),
        "alpha": float(setting["alpha"]),
        "beta": float(setting["beta"]),
        "gamma": float(setting["gamma"]),
        "delta": float(setting["delta"]),
        "n_models": len(model_names),
        "max_model_weight": MAX_MODEL_WEIGHT,
        "weights_json": json.dumps(dict(zip(model_names, best_weights.astype(float))), sort_keys=True),
    }
    return OptimizedSettingResult(
        setting_id=setting_id,
        weights=best_weights,
        metrics=metrics_with_setting,
        n_candidates_evaluated=len(cache),
        best_candidate_source=best_source,
    )


def select_final_setting(validation_objectives: pd.DataFrame) -> str:
    """Select the final objective setting using validation-only operational ranking."""

    ranked = validation_objectives.sort_values(
        ["expected_cost", "pr_auc", "ece", "brier_score", "recall", "objective_value"],
        ascending=[True, False, True, True, False, True],
    )
    return str(ranked.iloc[0]["setting_id"])


def evaluate_split(
    dataset: str,
    split: str,
    setting_id: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    selected_final_setting: bool,
) -> dict[str, object]:
    """Build one SCRE-Optimized result row."""

    metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
    return {
        "dataset": dataset,
        "model": "SCRE-Optimized",
        "split": split,
        "setting_id": setting_id,
        "selected_final_setting": selected_final_setting,
        "calibration_type": "weighted_sum_of_calibrated_pareto_front_probabilities_no_final_calibration",
        "threshold": float(threshold),
        "threshold_source": "validation_cost_min_FN5_FP1_for_setting",
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "expected_cost": float(FN_COST * metrics["fn"] + FP_COST * metrics["fp"]),
        "weight_learning_split": "validation",
        "setting_selection_split": "validation",
        "test_set_used_for_weight_optimization": False,
        "test_set_used_for_setting_selection": False,
        "test_set_used_for_threshold_selection": False,
        "valid_for_model_selection": split == "validation",
        **metrics,
    }


def run_dataset(dataset: str, logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train and evaluate SCRE-Optimized for one dataset."""

    logger.info("Training SCRE-Optimized for %s", dataset)
    model_names = load_pareto_front_models(dataset)
    X_validation, y_validation, X_test, y_test = load_validation_test(dataset)
    validation_probabilities = load_base_model_probabilities(dataset, X_validation, model_names)
    test_probabilities = load_base_model_probabilities(dataset, X_test, model_names)

    pareto_weight_path = TABLES_DIR / f"scre_pareto_weights_{dataset}.csv"
    initial_weights = None
    if pareto_weight_path.exists():
        pareto_weights = pd.read_csv(pareto_weight_path).set_index("model").loc[model_names, "ensemble_weight"]
        initial_weights = pareto_weights.to_numpy(dtype=float)

    setting_results: list[OptimizedSettingResult] = []
    for index, (setting_id, setting) in enumerate(OBJECTIVE_SETTINGS.items(), start=1):
        result = optimize_one_setting(
            setting_id=setting_id,
            setting=setting,
            probability_frame=validation_probabilities,
            y_true=y_validation,
            initial_weights=initial_weights,
            random_seed=RANDOM_SEED + 1000 * index + (0 if dataset == "taiwan" else 100),
        )
        setting_results.append(result)
        logger.info(
            "%s/%s validation objective=%0.4f cost=%0.1f PR-AUC=%0.4f threshold=%0.2f",
            dataset,
            setting_id,
            result.metrics["objective_value"],
            result.metrics["expected_cost"],
            result.metrics["pr_auc"],
            result.metrics["threshold"],
        )

    objective_rows = []
    for result in setting_results:
        objective_rows.append(
            {
                "dataset": dataset,
                "model": "SCRE-Optimized",
                "split": "validation",
                "cost_scenario": PRIMARY_COST_SCENARIO,
                "optimization_split": "validation",
                "test_set_used_for_weight_optimization": False,
                "test_set_used_for_setting_selection": False,
                "test_set_used_for_threshold_selection": False,
                "best_candidate_source": result.best_candidate_source,
                "n_candidates_evaluated": result.n_candidates_evaluated,
                **result.metrics,
            }
        )
    objectives = pd.DataFrame(objective_rows)
    selected_setting = select_final_setting(objectives)
    objectives["selected_final_setting"] = objectives["setting_id"].eq(selected_setting)
    objectives["setting_selection_rule"] = (
        "validation expected_cost asc, PR-AUC desc, ECE asc, Brier asc, recall desc"
    )

    weights_rows = []
    for result in setting_results:
        for model, weight in zip(model_names, result.weights):
            weights_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "scre_variant": "SCRE-Optimized",
                    "setting_id": result.setting_id,
                    "selected_final_setting": result.setting_id == selected_setting,
                    "ensemble_weight": float(weight),
                    "max_model_weight": MAX_MODEL_WEIGHT,
                    "weight_learning_split": "validation",
                    "test_set_used_for_weight_optimization": False,
                    "pareto_front_only": True,
                    "cost_scenario": PRIMARY_COST_SCENARIO,
                    "objective_formula": (
                        "expected_cost_FN5_FP1 + alpha*ECE + beta*Brier "
                        "- gamma*PR_AUC - delta*Recall"
                    ),
                }
            )
    weights_table = pd.DataFrame(weights_rows)

    validation_result_rows = []
    selected_result = next(result for result in setting_results if result.setting_id == selected_setting)
    for result in setting_results:
        probability = weighted_probability(validation_probabilities.to_numpy(dtype=float), result.weights)
        validation_result_rows.append(
            evaluate_split(
                dataset=dataset,
                split="validation",
                setting_id=result.setting_id,
                y_true=y_validation,
                y_proba=probability,
                threshold=float(result.metrics["threshold"]),
                selected_final_setting=result.setting_id == selected_setting,
            )
        )

    selected_test_probability = weighted_probability(test_probabilities.to_numpy(dtype=float), selected_result.weights)
    validation_result_rows.append(
        evaluate_split(
            dataset=dataset,
            split="test",
            setting_id=selected_setting,
            y_true=y_test,
            y_proba=selected_test_probability,
            threshold=float(selected_result.metrics["threshold"]),
            selected_final_setting=True,
        )
    )
    results = pd.DataFrame(validation_result_rows)

    artifact = {
        "dataset": dataset,
        "model_name": "SCRE-Optimized",
        "selected_setting": selected_setting,
        "objective_settings": OBJECTIVE_SETTINGS,
        "model_names": model_names,
        "selected_weights": dict(zip(model_names, selected_result.weights.astype(float))),
        "selected_threshold": float(selected_result.metrics["threshold"]),
        "threshold_source": "validation_cost_min_FN5_FP1_for_setting",
        "max_model_weight": MAX_MODEL_WEIGHT,
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "weight_learning_split": "validation",
        "setting_selection_split": "validation",
        "test_set_used_for_weight_optimization": False,
        "test_set_used_for_setting_selection": False,
        "test_set_used_for_threshold_selection": False,
        "validation_objectives": objectives,
        "weights_table": weights_table,
        "results": results,
    }

    config = DATASET_CONFIGS[dataset]
    config["weights_output"].parent.mkdir(parents=True, exist_ok=True)
    config["model_output"].parent.mkdir(parents=True, exist_ok=True)
    weights_table.to_csv(config["weights_output"], index=False)
    results.to_csv(config["results_output"], index=False)
    joblib.dump(artifact, config["model_output"])
    logger.info("Saved %s", config["weights_output"])
    logger.info("Saved %s", config["results_output"])
    logger.info("Saved %s", config["model_output"])
    return objectives, weights_table, results, artifact


def main() -> None:
    """Run SCRE-Optimized for Taiwan and HELOC."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger("train_scre_optimized", EXPERIMENT_LOGS_DIR / "22_train_scre_optimized.log")

    all_objectives = []
    all_results = []
    for dataset in ["taiwan", "heloc"]:
        objectives, _, results, _ = run_dataset(dataset, logger)
        all_objectives.append(objectives)
        all_results.append(results)
        print(
            results[
                [
                    "dataset",
                    "split",
                    "setting_id",
                    "selected_final_setting",
                    "threshold",
                    "expected_cost",
                    "pr_auc",
                    "roc_auc",
                    "recall",
                    "f1",
                    "brier_score",
                    "ece",
                ]
            ].to_string(index=False)
        )

    objective_table = pd.concat(all_objectives, ignore_index=True)
    objective_output = TABLES_DIR / "scre_optimized_validation_objectives.csv"
    objective_table.to_csv(objective_output, index=False)
    logger.info("Saved %s", objective_output)
    logger.info("SCRE-Optimized validation objectives:\n%s", objective_table.to_string(index=False))


if __name__ == "__main__":
    main()
