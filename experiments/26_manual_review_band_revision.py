"""Revise manual-review bands for final candidate credit-risk models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from heloc_preprocessing import HELOC_TARGET
from run_calibration_analysis import _positive_class_probability
from src.config.paths import (
    EXPERIMENT_LOGS_DIR,
    FIGURES_DIR,
    HELOC_MODEL_READY,
    MODELS_DIR,
    TABLES_DIR,
    TAIWAN_MODEL_READY,
)
from src.utils.logging_utils import create_experiment_logger


FN_COST = 5.0
FP_COST = 1.0
PRIMARY_COST_SCENARIO = "B_FN5_FP1"
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

DATASET_CONFIGS = {
    "taiwan": {"path": TAIWAN_MODEL_READY, "target": TARGET_COLUMN},
    "heloc": {"path": HELOC_MODEL_READY, "target": HELOC_TARGET},
}

FINAL_CANDIDATE_MODELS = [
    "Best CatBoost",
    "Best Scorecard",
    "Old SCRE-Credit",
    "SCRE-Pareto",
    "SCRE-Optimized",
]

MODEL_SOURCE_CANDIDATES = {
    "Best CatBoost": ["catboost", "optuna_best_catboost"],
    "Best Scorecard": ["woe_scorecard_logistic_regression"],
}


@dataclass(frozen=True)
class ModelProbabilityBundle:
    """Validation/test probabilities and metadata for one final candidate model."""

    display_model: str
    source_model: str
    validation_proba: np.ndarray
    test_proba: np.ndarray
    threshold: float
    calibration_type: str


def load_splits(dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load deterministic validation and test splits."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_validation = split.validation.drop(columns=[config["target"]]).reset_index(drop=True)
    y_validation = split.validation[config["target"]].astype(int).reset_index(drop=True)
    X_test = split.test.drop(columns=[config["target"]]).reset_index(drop=True)
    y_test = split.test[config["target"]].astype(int).reset_index(drop=True)
    return X_validation, y_validation, X_test, y_test


def select_source_model(dataset: str, display_model: str, weights: pd.DataFrame) -> str:
    """Select a source model by validation expected cost within a display family."""

    candidates = MODEL_SOURCE_CANDIDATES[display_model]
    table = weights.loc[(weights["dataset"] == dataset) & (weights["model"].isin(candidates))].copy()
    if table.empty:
        raise ValueError(f"No source model available for {dataset}/{display_model}.")
    table = table.sort_values(
        ["expected_cost", "pr_auc", "roc_auc", "brier_score", "ece"],
        ascending=[True, False, False, True, True],
    )
    return str(table.iloc[0]["model"])


def positive_probability_from_artifact(model_path: Path, X: pd.DataFrame) -> np.ndarray:
    """Load a fitted probability model and return P(y=1)."""

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    return _positive_class_probability(model, X)


def base_probabilities(
    dataset: str,
    X: pd.DataFrame,
    model_names: list[str],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Load base calibrated probabilities, including Old SCRE-Credit if requested."""

    dataset_weights = weights.loc[weights["dataset"] == dataset].set_index("model")
    probabilities: dict[str, np.ndarray] = {}
    non_scre_models = [model for model in model_names if model != "SCRE-Credit"]
    for model in non_scre_models:
        if model not in dataset_weights.index:
            raise KeyError(f"Missing calibrated model metadata for {dataset}/{model}.")
        probabilities[model] = positive_probability_from_artifact(Path(dataset_weights.loc[model, "model_path"]), X)

    if "SCRE-Credit" in model_names:
        scre_path = MODELS_DIR / "scre_credit_model.joblib"
        if not scre_path.exists():
            raise FileNotFoundError(scre_path)
        bundle = joblib.load(scre_path)
        classifier = bundle[dataset]["classifier"]
        base = base_probabilities(dataset, X, classifier.model_names_, weights)
        probabilities["SCRE-Credit"] = classifier.predict_proba_from_base(base[classifier.model_names_])[:, 1]

    return pd.DataFrame({model: probabilities[model] for model in model_names}, index=X.index)


def load_single_candidate(
    dataset: str,
    display_model: str,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    weights: pd.DataFrame,
) -> ModelProbabilityBundle:
    """Load a validation-selected calibrated single-model candidate."""

    source_model = select_source_model(dataset, display_model, weights)
    row = weights.loc[(weights["dataset"] == dataset) & (weights["model"] == source_model)].iloc[0]
    model_path = Path(row["model_path"])
    return ModelProbabilityBundle(
        display_model=display_model,
        source_model=source_model,
        validation_proba=positive_probability_from_artifact(model_path, X_validation),
        test_proba=positive_probability_from_artifact(model_path, X_test),
        threshold=float(row["base_selected_threshold"]),
        calibration_type=str(row["selected_calibration_method"]),
    )


def load_old_scre_candidate(
    dataset: str,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    weights: pd.DataFrame,
) -> ModelProbabilityBundle:
    """Load Old SCRE-Credit validation/test probabilities."""

    scre_path = MODELS_DIR / "scre_credit_model.joblib"
    if not scre_path.exists():
        raise FileNotFoundError(scre_path)
    bundle = joblib.load(scre_path)
    classifier = bundle[dataset]["classifier"]
    validation_base = base_probabilities(dataset, X_validation, classifier.model_names_, weights)
    test_base = base_probabilities(dataset, X_test, classifier.model_names_, weights)
    return ModelProbabilityBundle(
        display_model="Old SCRE-Credit",
        source_model="SCRE-Credit",
        validation_proba=classifier.predict_proba_from_base(validation_base[classifier.model_names_])[:, 1],
        test_proba=classifier.predict_proba_from_base(test_base[classifier.model_names_])[:, 1],
        threshold=float(classifier.threshold_),
        calibration_type="weighted_sum_plus_isotonic",
    )


def load_revision_candidate(
    dataset: str,
    variant: str,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    weights: pd.DataFrame,
) -> ModelProbabilityBundle:
    """Load SCRE-Pareto or SCRE-Optimized validation/test probabilities."""

    if variant == "SCRE-Pareto":
        path = MODELS_DIR / f"scre_pareto_{dataset}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        artifact = joblib.load(path)
        model_names = artifact["model_names"]
        ensemble_weights = pd.Series(artifact["ensemble_weights"], dtype=float)
        threshold = float(artifact["threshold"])
        calibration_type = str(artifact.get("calibration_type", "weighted_sum"))
    elif variant == "SCRE-Optimized":
        path = MODELS_DIR / f"scre_optimized_{dataset}.joblib"
        if not path.exists():
            raise FileNotFoundError(path)
        artifact = joblib.load(path)
        model_names = artifact["model_names"]
        ensemble_weights = pd.Series(artifact["selected_weights"], dtype=float)
        threshold = float(artifact["selected_threshold"])
        calibration_type = f"validation_optimized_weighted_sum_{artifact['selected_setting']}"
    else:
        raise ValueError(f"Unknown revision variant: {variant}")

    ordered_weights = ensemble_weights.loc[model_names].to_numpy(dtype=float)
    ordered_weights = ordered_weights / ordered_weights.sum()
    validation_base = base_probabilities(dataset, X_validation, model_names, weights)
    test_base = base_probabilities(dataset, X_test, model_names, weights)
    return ModelProbabilityBundle(
        display_model=variant,
        source_model=variant,
        validation_proba=np.clip(validation_base[model_names].to_numpy(dtype=float) @ ordered_weights, 0.0, 1.0),
        test_proba=np.clip(test_base[model_names].to_numpy(dtype=float) @ ordered_weights, 0.0, 1.0),
        threshold=threshold,
        calibration_type=calibration_type,
    )


def load_candidates(dataset: str) -> tuple[pd.Series, pd.Series, list[ModelProbabilityBundle]]:
    """Load validation/test labels and all final candidate probabilities."""

    X_validation, y_validation, X_test, y_test = load_splits(dataset)
    weights = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
    candidates = [
        load_single_candidate(dataset, "Best CatBoost", X_validation, X_test, weights),
        load_single_candidate(dataset, "Best Scorecard", X_validation, X_test, weights),
        load_old_scre_candidate(dataset, X_validation, X_test, weights),
        load_revision_candidate(dataset, "SCRE-Pareto", X_validation, X_test, weights),
        load_revision_candidate(dataset, "SCRE-Optimized", X_validation, X_test, weights),
    ]
    return y_validation, y_test, candidates


def threshold_only_cost(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    """Compute binary threshold-only expected cost."""

    y_pred = (y_proba >= threshold).astype(int)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return float(FN_COST * fn + FP_COST * fp)


def evaluate_band(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    selected_threshold: float,
) -> dict[str, float | int]:
    """Evaluate a low/manual/high-risk band."""

    low = y_proba < low_threshold
    high = y_proba >= high_threshold
    manual = ~(low | high)
    default = y_true == 1
    non_default = ~default
    n_rows = len(y_true)
    total_defaults = int(default.sum())
    no_review_cost = float(FN_COST * total_defaults)
    threshold_cost = threshold_only_cost(y_true, y_proba, selected_threshold)

    low_count = int(low.sum())
    manual_count = int(manual.sum())
    high_count = int(high.sum())
    low_defaults = int(np.sum(low & default))
    manual_defaults = int(np.sum(manual & default))
    high_defaults = int(np.sum(high & default))
    high_non_defaults = int(np.sum(high & non_default))
    remaining_cost = float(FN_COST * low_defaults + FP_COST * high_non_defaults)

    return {
        "low_risk_count": low_count,
        "manual_review_count": manual_count,
        "high_risk_count": high_count,
        "auto_decision_rate": float((low_count + high_count) / n_rows),
        "manual_review_rate": float(manual_count / n_rows),
        "default_rate_low_risk": float(low_defaults / low_count) if low_count else np.nan,
        "default_rate_manual_review": float(manual_defaults / manual_count) if manual_count else np.nan,
        "default_rate_high_risk": float(high_defaults / high_count) if high_count else np.nan,
        "captured_defaults_high_risk": high_defaults,
        "captured_defaults_high_risk_rate": float(high_defaults / total_defaults) if total_defaults else np.nan,
        "remaining_expected_cost": remaining_cost,
        "no_review_expected_cost": no_review_cost,
        "threshold_only_expected_cost": threshold_cost,
        "cost_reduction_vs_no_review": float(no_review_cost - remaining_cost),
        "cost_reduction_vs_threshold_only": float(threshold_cost - remaining_cost),
        "cost_reduction_pct_vs_no_review": float((no_review_cost - remaining_cost) / no_review_cost)
        if no_review_cost
        else np.nan,
        "cost_reduction_pct_vs_threshold_only": float((threshold_cost - remaining_cost) / threshold_cost)
        if threshold_cost
        else np.nan,
        "low_risk_default_count": low_defaults,
        "manual_review_default_count": manual_defaults,
        "high_risk_default_count": high_defaults,
        "high_risk_non_default_count": high_non_defaults,
        "overall_default_rate": float(total_defaults / n_rows),
    }


def band_selection_score(metrics: dict[str, float | int]) -> float:
    """Score validation bands with cost, review-width, and risk-separation penalties."""

    threshold_cost = float(metrics["threshold_only_expected_cost"])
    overall_default_rate = float(metrics["overall_default_rate"])
    manual_rate = float(metrics["manual_review_rate"])
    high_recall = float(metrics["captured_defaults_high_risk_rate"])
    low_default_rate = float(metrics["default_rate_low_risk"]) if pd.notna(metrics["default_rate_low_risk"]) else 1.0
    auto_rate = float(metrics["auto_decision_rate"])

    penalty = 0.0
    penalty += max(0.0, manual_rate - 0.30) * threshold_cost * 1.25
    penalty += max(0.0, 0.65 - auto_rate) * threshold_cost * 0.75
    penalty += max(0.0, 0.55 - high_recall) * threshold_cost * 0.75
    penalty += max(0.0, low_default_rate - overall_default_rate * 0.75) * threshold_cost * 1.00
    review_penalty = 0.50 * int(metrics["manual_review_count"])
    return float(metrics["remaining_expected_cost"] + review_penalty + penalty)


def select_validation_band(y_true: pd.Series, bundle: ModelProbabilityBundle) -> tuple[float, float, pd.DataFrame]:
    """Search validation low/high thresholds and return the selected band."""

    y_array = y_true.to_numpy(dtype=int)
    rows = []
    for low_threshold in THRESHOLD_GRID:
        for high_threshold in THRESHOLD_GRID:
            if low_threshold >= high_threshold:
                continue
            metrics = evaluate_band(
                y_true=y_array,
                y_proba=bundle.validation_proba,
                low_threshold=float(low_threshold),
                high_threshold=float(high_threshold),
                selected_threshold=bundle.threshold,
            )
            objective = band_selection_score(metrics)
            rows.append(
                {
                    "low_risk_threshold": float(low_threshold),
                    "high_risk_threshold": float(high_threshold),
                    "selection_objective": objective,
                    "meets_review_width_guideline": bool(metrics["manual_review_rate"] <= 0.30),
                    "meets_auto_coverage_guideline": bool(metrics["auto_decision_rate"] >= 0.65),
                    "meets_high_risk_recall_guideline": bool(metrics["captured_defaults_high_risk_rate"] >= 0.55),
                    "meets_low_risk_leakage_guideline": bool(
                        metrics["default_rate_low_risk"] <= metrics["overall_default_rate"] * 0.75
                        if pd.notna(metrics["default_rate_low_risk"])
                        else False
                    ),
                    **metrics,
                }
            )

    table = pd.DataFrame(rows)
    guideline_cols = [
        "meets_review_width_guideline",
        "meets_auto_coverage_guideline",
        "meets_high_risk_recall_guideline",
        "meets_low_risk_leakage_guideline",
    ]
    table["guidelines_met_count"] = table[guideline_cols].sum(axis=1)
    feasible = table.loc[table[guideline_cols].all(axis=1)].copy()
    if feasible.empty:
        feasible = table.sort_values(
            [
                "guidelines_met_count",
                "selection_objective",
                "remaining_expected_cost",
                "auto_decision_rate",
                "captured_defaults_high_risk_rate",
            ],
            ascending=[False, True, True, False, False],
        ).head(200)
    selected = feasible.sort_values(
        [
            "selection_objective",
            "remaining_expected_cost",
            "manual_review_rate",
            "auto_decision_rate",
            "captured_defaults_high_risk_rate",
        ],
        ascending=[True, True, True, False, False],
    ).iloc[0]
    return float(selected["low_risk_threshold"]), float(selected["high_risk_threshold"]), table


def result_row(
    dataset: str,
    split: str,
    bundle: ModelProbabilityBundle,
    y_true: pd.Series,
    y_proba: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    selection_table: pd.DataFrame,
) -> dict[str, object]:
    """Build one output row for validation or test."""

    metrics = evaluate_band(
        y_true=y_true.to_numpy(dtype=int),
        y_proba=y_proba,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        selected_threshold=bundle.threshold,
    )
    selection_match = selection_table.loc[
        (np.isclose(selection_table["low_risk_threshold"], low_threshold))
        & (np.isclose(selection_table["high_risk_threshold"], high_threshold))
    ].iloc[0]
    return {
        "dataset": dataset,
        "model": bundle.display_model,
        "source_model": bundle.source_model,
        "split": split,
        "selection_split": "validation",
        "calibration_type": bundle.calibration_type,
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "threshold_only_threshold": bundle.threshold,
        "low_risk_threshold": low_threshold,
        "high_risk_threshold": high_threshold,
        "test_set_used_for_band_selection": False,
        "valid_for_band_selection": split == "validation",
        "selection_objective_validation": float(selection_match["selection_objective"]),
        "guidelines_met_count_validation": int(selection_match["guidelines_met_count"]),
        "selection_policy": (
            "validation-only search over low/high thresholds; minimize remaining auto-decision cost "
            "with manual-review-width, auto-coverage, high-risk-recall, and low-risk-leakage penalties"
        ),
        **metrics,
    }


def build_dataset_revision(dataset: str, logger) -> pd.DataFrame:
    """Build manual-review-band revision rows for one dataset."""

    y_validation, y_test, bundles = load_candidates(dataset)
    rows = []
    for bundle in bundles:
        logger.info("Selecting manual-review band for %s/%s", dataset, bundle.display_model)
        low_threshold, high_threshold, selection_table = select_validation_band(y_validation, bundle)
        rows.append(
            result_row(
                dataset,
                "validation",
                bundle,
                y_validation,
                bundle.validation_proba,
                low_threshold,
                high_threshold,
                selection_table,
            )
        )
        rows.append(
            result_row(
                dataset,
                "test",
                bundle,
                y_test,
                bundle.test_proba,
                low_threshold,
                high_threshold,
                selection_table,
            )
        )
    result = pd.DataFrame(rows)
    result["band_rank_within_split"] = result.groupby(["dataset", "split"])["remaining_expected_cost"].rank(
        method="min",
        ascending=True,
    ).astype(int)
    return result.sort_values(["split", "band_rank_within_split", "model"]).reset_index(drop=True)


def plot_band_distribution(dataset: str, table: pd.DataFrame, output_path: Path) -> None:
    """Save stacked low/manual/high count bars for test-set band distributions."""

    test = table.loc[table["split"] == "test"].sort_values("band_rank_within_split")
    labels = test["model"].tolist()
    low_counts = test["low_risk_count"].to_numpy(dtype=float)
    manual_counts = test["manual_review_count"].to_numpy(dtype=float)
    high_counts = test["high_risk_count"].to_numpy(dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, low_counts, label="Low risk", color="#5b8e7d")
    ax.bar(x, manual_counts, bottom=low_counts, label="Manual review", color="#d9a441")
    ax.bar(x, high_counts, bottom=low_counts + manual_counts, label="High risk", color="#b14d4d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test-set count")
    ax.set_title(f"{dataset.upper()} Manual Review Band Distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    for idx, row in enumerate(test.itertuples(index=False)):
        ax.text(
            idx,
            row.low_risk_count + row.manual_review_count + row.high_risk_count + 20,
            f"review {row.manual_review_rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run manual-review-band revision for Taiwan and HELOC."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger(
        "manual_review_band_revision",
        EXPERIMENT_LOGS_DIR / "26_manual_review_band_revision.log",
    )

    for dataset in ["taiwan", "heloc"]:
        table = build_dataset_revision(dataset, logger)
        table_path = TABLES_DIR / f"manual_review_band_revision_{dataset}.csv"
        figure_path = FIGURES_DIR / f"manual_review_band_distribution_{dataset}.png"
        table.to_csv(table_path, index=False)
        plot_band_distribution(dataset, table, figure_path)
        logger.info("Saved %s", table_path)
        logger.info("Saved %s", figure_path)
        print(
            table.loc[table["split"] == "test"][
                [
                    "dataset",
                    "model",
                    "low_risk_threshold",
                    "high_risk_threshold",
                    "auto_decision_rate",
                    "manual_review_rate",
                    "default_rate_low_risk",
                    "default_rate_high_risk",
                    "captured_defaults_high_risk_rate",
                    "remaining_expected_cost",
                    "cost_reduction_vs_threshold_only",
                    "band_rank_within_split",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
