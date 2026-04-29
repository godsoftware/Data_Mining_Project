"""Train SCRE-Pareto from validation-selected Pareto-front models only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from heloc_preprocessing import HELOC_TARGET
from run_calibration_analysis import _positive_class_probability
from src.config.paths import (
    EXPERIMENT_LOGS_DIR,
    HELOC_MODEL_READY,
    MODELS_DIR,
    TABLES_DIR,
    TAIWAN_MODEL_READY,
)
from src.models.scre_credit import threshold_cost_table, weighted_probability_sum
from src.utils.logging_utils import create_experiment_logger


PRIMARY_COST_SCENARIO = "B_FN5_FP1"
FN_COST = 5.0
FP_COST = 1.0

COMPONENT_WEIGHTS = {
    "pr_auc_norm": 0.25,
    "roc_auc_norm": 0.15,
    "recall_norm": 0.15,
    "cost_norm": 0.15,
    "calibration_norm": 0.10,
    "stability_norm": 0.10,
    "faithfulness_norm": 0.10,
}

DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "pareto_front": TABLES_DIR / "pareto_front_taiwan.csv",
        "weights_output": TABLES_DIR / "scre_pareto_weights_taiwan.csv",
        "results_output": TABLES_DIR / "scre_pareto_results_taiwan.csv",
        "model_output": MODELS_DIR / "scre_pareto_taiwan.joblib",
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "pareto_front": TABLES_DIR / "pareto_front_heloc.csv",
        "weights_output": TABLES_DIR / "scre_pareto_weights_heloc.csv",
        "results_output": TABLES_DIR / "scre_pareto_results_heloc.csv",
        "model_output": MODELS_DIR / "scre_pareto_heloc.joblib",
    },
}


def normalize_higher(values: pd.Series) -> pd.Series:
    """Min-max normalize a higher-is-better metric."""

    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return result
    value_range = float(valid.max() - valid.min())
    if value_range == 0.0 or not np.isfinite(value_range):
        result.loc[valid.index] = 1.0
    else:
        result.loc[valid.index] = (valid - valid.min()) / value_range
    return result


def normalize_lower(values: pd.Series) -> pd.Series:
    """Min-max normalize a lower-is-better metric."""

    series = pd.to_numeric(values, errors="coerce")
    valid = series.dropna()
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if valid.empty:
        return result
    value_range = float(valid.max() - valid.min())
    if value_range == 0.0 or not np.isfinite(value_range):
        result.loc[valid.index] = 1.0
    else:
        result.loc[valid.index] = 1.0 - ((valid - valid.min()) / value_range)
    return result


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


def model_level_stability(dataset: str, model: str) -> float:
    """Return available model-level SHAP stability, otherwise NaN."""

    if model != "lightgbm":
        return np.nan
    path = TABLES_DIR / "shap_kendalls_w.csv"
    if not path.exists():
        return np.nan
    table = pd.read_csv(path)
    match = table.loc[(table["dataset"] == dataset) & (table["model_family"] == "lightgbm")]
    if match.empty:
        return np.nan
    return float(match["kendalls_w"].iloc[0])


def model_level_faithfulness(dataset: str, model: str) -> float:
    """Return available model-level faithfulness proxy, otherwise NaN."""

    if dataset == "taiwan" and model == "lightgbm":
        path = TABLES_DIR / "faithfulness_results.csv"
        if not path.exists():
            return np.nan
        table = pd.read_csv(path)
        match = table.loc[
            (table["dataset"] == dataset)
            & (table["model"] == model)
            & (table["test_type"] == "topk_deletion")
            & (table["step"] == table.loc[table["test_type"] == "topk_deletion", "step"].max())
        ]
        if match.empty:
            return np.nan
        return float(match["pr_auc_drop"].iloc[0])

    if dataset == "heloc" and model == "xgboost":
        path = TABLES_DIR / "external_validation_heloc_faithfulness_results.csv"
        if not path.exists():
            return np.nan
        table = pd.read_csv(path)
        match = table.loc[
            (table["dataset"] == dataset)
            & (table["test_type"] == "topk_deletion")
            & (table["step"] == table.loc[table["test_type"] == "topk_deletion", "step"].max())
        ]
        if match.empty:
            return np.nan
        return float(match["pr_auc_drop"].iloc[0])
    return np.nan


def build_weight_table(dataset: str) -> pd.DataFrame:
    """Build SCRE-Pareto weights from validation Pareto-front metrics only."""

    config = DATASET_CONFIGS[dataset]
    front = pd.read_csv(config["pareto_front"])
    front = front.loc[front["is_pareto_front"].astype(bool)].copy().reset_index(drop=True)
    if front.empty:
        raise ValueError(f"No Pareto-front models found for {dataset}.")

    table = front.copy()
    table["stability_score"] = [model_level_stability(dataset, model) for model in table["model"]]
    table["faithfulness_score"] = [model_level_faithfulness(dataset, model) for model in table["model"]]
    table["pr_auc_norm"] = normalize_higher(table["pr_auc"])
    table["roc_auc_norm"] = normalize_higher(table["roc_auc"])
    table["recall_norm"] = normalize_higher(table["recall"])
    table["cost_norm"] = normalize_lower(table["expected_cost"])
    table["calibration_metric_used"] = "brier"
    table["calibration_norm"] = normalize_lower(table["brier"])
    table["stability_norm"] = normalize_higher(table["stability_score"])
    table["faithfulness_norm"] = normalize_higher(table["faithfulness_score"])

    scores = []
    effective_weight_json = []
    active_components_text = []
    missing_components_text = []
    for row in table.itertuples(index=False):
        active = {}
        missing = []
        for component, base_weight in COMPONENT_WEIGHTS.items():
            value = getattr(row, component)
            if pd.isna(value):
                missing.append(component)
            else:
                active[component] = float(base_weight)
        active_sum = sum(active.values())
        if active_sum <= 0:
            raise ValueError(f"No active SCRE-Pareto components for {dataset}/{row.model}.")
        effective = {component: weight / active_sum for component, weight in active.items()}
        score = sum(effective[component] * float(getattr(row, component)) for component in effective)
        scores.append(score)
        effective_weight_json.append(json.dumps(effective, sort_keys=True))
        active_components_text.append(", ".join(effective))
        missing_components_text.append(", ".join(missing))

    table["reliability_score"] = scores
    total_score = float(table["reliability_score"].sum())
    if total_score <= 0 or not np.isfinite(total_score):
        table["ensemble_weight"] = 1.0 / len(table)
    else:
        table["ensemble_weight"] = table["reliability_score"] / total_score
    table["base_component_weights_json"] = json.dumps(COMPONENT_WEIGHTS, sort_keys=True)
    table["effective_component_weights_json"] = effective_weight_json
    table["active_components"] = active_components_text
    table["missing_components_renormalized"] = missing_components_text
    table["weight_learning_split"] = "validation"
    table["test_set_used_for_weight_learning"] = False
    table["cost_scenario"] = PRIMARY_COST_SCENARIO
    table["formula"] = (
        "0.25*PR_AUC_norm + 0.15*ROC_AUC_norm + 0.15*Recall_norm "
        "+ 0.15*Cost_norm + 0.10*Calibration_norm + 0.10*Stability_norm "
        "+ 0.10*Faithfulness_norm; missing stability/faithfulness renormalized per model"
    )
    table["notes"] = np.where(
        table["missing_components_renormalized"].eq(""),
        "all requested SCRE-Pareto components available",
        "missing components were excluded from this model score and remaining component weights were renormalized",
    )
    return table.sort_values("ensemble_weight", ascending=False).reset_index(drop=True)


def load_base_model_probabilities(dataset: str, X: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Load calibrated model probabilities, including Old SCRE-Credit if requested."""

    weights = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
    dataset_weights = weights.loc[weights["dataset"] == dataset].set_index("model")
    probabilities: dict[str, np.ndarray] = {}

    non_scre_models = [model for model in model_names if model != "SCRE-Credit"]
    for model in non_scre_models:
        if model not in dataset_weights.index:
            raise KeyError(f"Missing calibrated model metadata for {dataset}/{model}.")
        model_path = Path(dataset_weights.loc[model, "model_path"])
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
        all_base = load_base_model_probabilities(dataset, X, classifier.model_names_)
        probabilities["SCRE-Credit"] = classifier.predict_proba_from_base(all_base[classifier.model_names_])[:, 1]

    return pd.DataFrame({model: probabilities[model] for model in model_names}, index=X.index)


def evaluate_probability(
    dataset: str,
    split: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Build a metrics row for SCRE-Pareto."""

    metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
    return {
        "dataset": dataset,
        "model": "SCRE-Pareto",
        "split": split,
        "calibration_type": "weighted_sum_of_calibrated_pareto_front_probabilities",
        "threshold": float(threshold),
        "threshold_source": "validation_cost_min_FN5_FP1",
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "expected_cost": float(FN_COST * metrics["fn"] + FP_COST * metrics["fp"]),
        "weight_learning_split": "validation",
        "test_set_used_for_weight_learning": False,
        "test_set_used_for_threshold_selection": False,
        "valid_for_model_selection": split == "validation",
        **metrics,
    }


def run_dataset(dataset: str, logger) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train and evaluate SCRE-Pareto for one dataset."""

    config = DATASET_CONFIGS[dataset]
    logger.info("Building SCRE-Pareto for %s", dataset)
    weight_table = build_weight_table(dataset)
    model_names = weight_table["model"].tolist()
    weights = weight_table.set_index("model")["ensemble_weight"]

    X_validation, y_validation, X_test, y_test = load_validation_test(dataset)
    validation_probs = load_base_model_probabilities(dataset, X_validation, model_names)
    test_probs = load_base_model_probabilities(dataset, X_test, model_names)

    validation_probability = weighted_probability_sum(validation_probs[model_names], weights)
    test_probability = weighted_probability_sum(test_probs[model_names], weights)
    threshold_table = threshold_cost_table(
        y_validation,
        validation_probability,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    best_threshold = float(threshold_table.sort_values(["expected_cost", "threshold"]).iloc[0]["threshold"])

    results = pd.DataFrame(
        [
            evaluate_probability(dataset, "validation", y_validation, validation_probability, best_threshold),
            evaluate_probability(dataset, "test", y_test, test_probability, best_threshold),
        ]
    )

    model_artifact = {
        "dataset": dataset,
        "model_name": "SCRE-Pareto",
        "model_names": model_names,
        "ensemble_weights": weights.to_dict(),
        "threshold": best_threshold,
        "threshold_source": "validation_cost_min_FN5_FP1",
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "calibration_type": "weighted_sum_of_calibrated_pareto_front_probabilities",
        "test_set_used_for_weight_learning": False,
        "test_set_used_for_threshold_selection": False,
        "component_weights": COMPONENT_WEIGHTS,
        "weight_table": weight_table,
        "validation_threshold_table": threshold_table,
    }

    config["weights_output"].parent.mkdir(parents=True, exist_ok=True)
    config["model_output"].parent.mkdir(parents=True, exist_ok=True)
    weight_table.to_csv(config["weights_output"], index=False)
    results.to_csv(config["results_output"], index=False)
    joblib.dump(model_artifact, config["model_output"])
    logger.info("Saved %s", config["weights_output"])
    logger.info("Saved %s", config["results_output"])
    logger.info("Saved %s", config["model_output"])
    return weight_table, results, model_artifact


def main() -> None:
    """Run SCRE-Pareto for Taiwan and HELOC."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger("train_scre_pareto", EXPERIMENT_LOGS_DIR / "21_train_scre_pareto.log")
    all_results = []
    for dataset in ["taiwan", "heloc"]:
        _, results, _ = run_dataset(dataset, logger)
        all_results.append(results)
        print(
            results[
                [
                    "dataset",
                    "split",
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
    combined = pd.concat(all_results, ignore_index=True)
    logger.info("Combined SCRE-Pareto results:\n%s", combined.to_string(index=False))


if __name__ == "__main__":
    main()
