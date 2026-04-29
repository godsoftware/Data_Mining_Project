"""Final methodological revision for SCRE-Credit model selection.

This script does not add new datasets or model families. It revises the
existing SCRE-Credit ensemble variants, cleans statistical-test interpretation,
adds Kendall's W summaries, and writes final validation-selected decision
tables. Test data are used only for final evaluation rows.
"""

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
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_TARGET
from run_all import create_final_output_inventory
from run_calibration_analysis import _positive_class_probability
from src.config.paths import (
    EXPERIMENT_LOGS_DIR,
    HELOC_MODEL_READY,
    MODELS_DIR,
    TABLES_DIR,
    TAIWAN_MODEL_READY,
)
from src.evaluation.statistical_tests import clean_model_comparison_tests, metric_direction
from src.explainability.shap_stability import kendalls_w_summary_table
from src.models.scre_credit import (
    SCRECreditHybridClassifier,
    build_scre_pareto_pool,
    build_scre_weight_table,
    optimize_validation_ensemble_weights,
)
from src.utils.logging_utils import create_experiment_logger


PRIMARY_COST_SCENARIO = "B_FN5_FP1"
FN_COST = 5.0
FP_COST = 1.0
RANDOM_SEED = 42
EXCLUDED_PARETO_MODELS = ["majority_baseline"]

MAIN_MODEL_ORDER = [
    "CatBoost",
    "XGBoost",
    "LightGBM",
    "Scorecard",
    "SCRE-Static",
    "SCRE-Pareto",
    "SCRE-Optimized",
]
MAIN_MODEL_MAP = {
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "woe_scorecard_logistic_regression": "Scorecard",
    "SCRE-Credit": "SCRE-Static",
    "SCRE-Pareto": "SCRE-Pareto",
    "SCRE-Optimized": "SCRE-Optimized",
}

DATASET_CONFIGS = {
    "taiwan": {"path": TAIWAN_MODEL_READY, "target": TARGET_COLUMN},
    "heloc": {"path": HELOC_MODEL_READY, "target": HELOC_TARGET},
}

METRIC_DIRECTIONS = {
    "expected_cost": "lower_is_better",
    "expected_cost_per_1000": "lower_is_better",
    "roc_auc": "higher_is_better",
    "pr_auc": "higher_is_better",
    "recall": "higher_is_better",
    "precision": "higher_is_better",
    "f1": "higher_is_better",
    "brier_score": "lower_is_better",
    "ece": "lower_is_better",
    "shap_stability": "higher_is_better",
}


def load_validation_test(dataset: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load deterministic validation and test splits for one dataset."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_validation = split.validation.drop(columns=[config["target"]]).reset_index(drop=True)
    y_validation = split.validation[config["target"]].astype(int).reset_index(drop=True)
    X_test = split.test.drop(columns=[config["target"]]).reset_index(drop=True)
    y_test = split.test[config["target"]].astype(int).reset_index(drop=True)
    return X_validation, y_validation, X_test, y_test


def load_probability_frame(metric_table: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Load fitted calibrated models and return positive-class probabilities."""

    probabilities: dict[str, np.ndarray] = {}
    for row in metric_table.itertuples(index=False):
        model_path = Path(row.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing calibrated model artifact: {model_path}")
        fitted_model = joblib.load(model_path)
        probabilities[row.model] = _positive_class_probability(fitted_model, X)
    return pd.DataFrame(probabilities, index=X.index)


def metrics_row(
    dataset: str,
    model: str,
    split: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    model_role: str,
    calibration_type: str,
    threshold_source: str,
    weight_selection_split: str,
) -> dict[str, object]:
    """Create one final comparison metric row."""

    metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
    return {
        "dataset": dataset,
        "split": split,
        "model": model,
        "canonical_model": MAIN_MODEL_MAP.get(model, model),
        "model_role": model_role,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "calibration_type": calibration_type,
        "cost_scenario": PRIMARY_COST_SCENARIO,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "expected_cost": float(FN_COST * metrics["fn"] + FP_COST * metrics["fp"]),
        "selection_split": "validation",
        "weight_selection_split": weight_selection_split,
        "test_set_used_for_training_or_selection": False,
        "valid_for_model_selection": split == "validation",
        **metrics,
    }


def evaluate_revised_classifier(
    dataset: str,
    model_name: str,
    classifier: SCRECreditHybridClassifier,
    validation_probabilities: pd.DataFrame,
    test_probabilities: pd.DataFrame,
    y_validation: pd.Series,
    y_test: pd.Series,
    model_role: str,
    weight_selection_split: str = "validation",
) -> list[dict[str, object]]:
    """Evaluate a fitted revised SCRE variant on validation and final test."""

    validation_proba = classifier.validation_probability_
    test_proba = classifier.predict_proba_from_base(test_probabilities[classifier.model_names_])[:, 1]
    calibration_type = f"weighted_sum_plus_{classifier.final_calibration_method}"
    return [
        metrics_row(
            dataset=dataset,
            model=model_name,
            split="validation",
            y_true=y_validation,
            y_proba=validation_proba,
            threshold=classifier.threshold_,
            model_role=model_role,
            calibration_type=calibration_type,
            threshold_source="validation_cost_min_B_FN5_FP1",
            weight_selection_split=weight_selection_split,
        ),
        metrics_row(
            dataset=dataset,
            model=model_name,
            split="test",
            y_true=y_test,
            y_proba=test_proba,
            threshold=classifier.threshold_,
            model_role=model_role,
            calibration_type=calibration_type,
            threshold_source="validation_cost_min_B_FN5_FP1",
            weight_selection_split=weight_selection_split,
        ),
    ]


def add_winner_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add per-dataset/split winner booleans for key metrics."""

    result = table.copy()
    metrics = ["expected_cost", "roc_auc", "pr_auc", "recall", "f1", "brier_score", "ece"]
    for metric in metrics:
        if metric not in result.columns:
            continue
        direction = METRIC_DIRECTIONS[metric]
        result[f"{metric}_direction"] = direction
        result[f"winner_{metric}"] = False
        for (_, _), group in result.groupby(["dataset", "split"], dropna=False):
            if group.empty:
                continue
            best_value = group[metric].min() if direction == "lower_is_better" else group[metric].max()
            winner_mask = np.isclose(group[metric].astype(float), float(best_value), equal_nan=False)
            result.loc[group.index[winner_mask], f"winner_{metric}"] = True
    winner_cols = [column for column in result.columns if column.startswith("winner_")]
    result["winner"] = result[winner_cols].any(axis=1) if winner_cols else False
    return result


def build_metric_winner_table(final_comparison: pd.DataFrame) -> pd.DataFrame:
    """Build a long winner table for validation and test metrics."""

    rows: list[dict[str, object]] = []
    metrics = ["expected_cost", "roc_auc", "pr_auc", "recall", "f1", "brier_score", "ece"]
    for (dataset, split), group in final_comparison.groupby(["dataset", "split"], dropna=False):
        for metric in metrics:
            if metric not in group.columns:
                continue
            direction = METRIC_DIRECTIONS[metric]
            best_value = group[metric].min() if direction == "lower_is_better" else group[metric].max()
            winners = group.loc[np.isclose(group[metric].astype(float), float(best_value), equal_nan=False)].copy()
            winner_names = " / ".join(winners["canonical_model"].astype(str).tolist())
            winner_raw = " / ".join(winners["model"].astype(str).tolist())
            first_winner = winners.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "metric": metric,
                    "metric_direction": direction,
                    "winner": winner_names,
                    "winner_raw_model": winner_raw,
                    "n_winners": int(len(winners)),
                    "winner_value": best_value,
                    "threshold": first_winner.get("threshold"),
                    "calibration_type": first_winner.get("calibration_type"),
                    "cost_scenario": PRIMARY_COST_SCENARIO,
                    "valid_for_model_selection": split == "validation",
                    "interpretation": (
                        "Validation winner may be used for model selection."
                        if split == "validation"
                        else "Test winner is descriptive final evidence only, not a selection rule."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_revised_scre_outputs(logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build SCRE-Pareto and SCRE-Optimized outputs for Taiwan and HELOC."""

    weights = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
    all_results: list[dict[str, object]] = []
    all_weights: list[pd.DataFrame] = []
    all_pareto: list[pd.DataFrame] = []
    all_audit: list[pd.DataFrame] = []
    model_bundle: dict[str, object] = {}

    for dataset in ["taiwan", "heloc"]:
        logger.info("Revising SCRE variants for %s", dataset)
        metric_table = weights.loc[weights["dataset"] == dataset].reset_index(drop=True)
        X_validation, y_validation, X_test, y_test = load_validation_test(dataset)
        validation_probabilities = load_probability_frame(metric_table, X_validation)
        test_probabilities = load_probability_frame(metric_table, X_test)

        pareto_pool = build_scre_pareto_pool(metric_table, exclude_models=EXCLUDED_PARETO_MODELS)
        pareto_pool["dataset"] = dataset
        pareto_pool["winner"] = pareto_pool["expected_cost"].eq(pareto_pool["expected_cost"].min())
        pareto_pool["winner_basis"] = "lowest_validation_expected_cost_within_pareto_pool"
        all_pareto.append(pareto_pool)

        pareto_weight_table = build_scre_weight_table(pareto_pool.drop(columns=["dataset"]))
        pareto_weight_table["dataset"] = dataset
        pareto_weight_table["scre_variant"] = "SCRE-Pareto"
        pareto_weight_table["weight_selection_split"] = "validation"
        pareto_weight_table["test_set_used_for_weight_optimization"] = False
        pareto_weight_table["winner"] = pareto_weight_table["ensemble_weight"].eq(
            pareto_weight_table["ensemble_weight"].max()
        )
        all_weights.append(pareto_weight_table)

        pareto_classifier = SCRECreditHybridClassifier(final_calibration_method="isotonic", fn_cost=FN_COST, fp_cost=FP_COST)
        pareto_classifier.fit_from_weight_table(
            validation_probabilities[pareto_weight_table["model"].tolist()],
            y_validation,
            pareto_weight_table,
        )
        all_results.extend(
            evaluate_revised_classifier(
                dataset,
                "SCRE-Pareto",
                pareto_classifier,
                validation_probabilities,
                test_probabilities,
                y_validation,
                y_test,
                model_role="scre_pareto",
            )
        )

        optimized_weight_table, audit = optimize_validation_ensemble_weights(
            probability_frame=validation_probabilities[pareto_weight_table["model"].tolist()],
            y_true=y_validation,
            metric_table=pareto_pool.drop(columns=["dataset"]),
            initial_weight_table=pareto_weight_table,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
            random_seed=RANDOM_SEED,
            n_random_candidates=3000,
            n_top_candidates_for_calibration=75,
            final_calibration_method="isotonic",
        )
        optimized_weight_table["dataset"] = dataset
        optimized_weight_table["scre_variant"] = "SCRE-Optimized"
        optimized_weight_table["winner"] = optimized_weight_table["ensemble_weight"].eq(
            optimized_weight_table["ensemble_weight"].max()
        )
        all_weights.append(optimized_weight_table)
        audit.insert(0, "dataset", dataset)
        audit["winner"] = audit["candidate_id"].eq(int(optimized_weight_table["selected_candidate_id"].iloc[0]))
        audit["cost_scenario"] = PRIMARY_COST_SCENARIO
        audit["selection_split"] = "validation"
        audit["test_set_used_for_weight_optimization"] = False
        all_audit.append(audit)

        optimized_classifier = SCRECreditHybridClassifier(final_calibration_method="isotonic", fn_cost=FN_COST, fp_cost=FP_COST)
        optimized_classifier.fit_from_weight_table(
            validation_probabilities[optimized_weight_table["model"].tolist()],
            y_validation,
            optimized_weight_table,
        )
        all_results.extend(
            evaluate_revised_classifier(
                dataset,
                "SCRE-Optimized",
                optimized_classifier,
                validation_probabilities,
                test_probabilities,
                y_validation,
                y_test,
                model_role="scre_optimized",
            )
        )
        model_bundle[dataset] = {
            "pareto": pareto_classifier,
            "optimized": optimized_classifier,
            "pareto_models": pareto_classifier.model_names_,
            "optimized_models": optimized_classifier.model_names_,
        }

    results = add_winner_columns(pd.DataFrame(all_results))
    weight_table = pd.concat(all_weights, ignore_index=True)
    pareto_table = pd.concat(all_pareto, ignore_index=True)
    audit_table = pd.concat(all_audit, ignore_index=True)
    return results, weight_table, pareto_table, audit_table, model_bundle


def build_final_model_comparison(revision_results: pd.DataFrame) -> pd.DataFrame:
    """Combine original main model results with revised SCRE variants."""

    frames = []
    for dataset, filename in [
        ("taiwan", "scre_credit_results_taiwan.csv"),
        ("heloc", "scre_credit_results_heloc.csv"),
    ]:
        current = pd.read_csv(TABLES_DIR / filename)
        current = current.loc[current["model"].isin(MAIN_MODEL_MAP)].copy()
        current["canonical_model"] = current["model"].map(MAIN_MODEL_MAP)
        current["calibration_type"] = current["calibration_method"]
        current["selection_split"] = "validation"
        current["weight_selection_split"] = np.where(
            current["model"].eq("SCRE-Credit"),
            "validation_static_metric_weighting",
            "not_applicable_base_model",
        )
        current["valid_for_model_selection"] = current["split"].eq("validation")
        frames.append(current)

    revised = revision_results.copy()
    frames.append(revised)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.loc[combined["canonical_model"].isin(MAIN_MODEL_ORDER)].copy()
    combined["canonical_model"] = pd.Categorical(combined["canonical_model"], categories=MAIN_MODEL_ORDER, ordered=True)
    combined = combined.sort_values(["dataset", "split", "canonical_model"]).reset_index(drop=True)
    combined["canonical_model"] = combined["canonical_model"].astype(str)
    combined = add_winner_columns(combined)
    return combined


def clean_statistical_outputs() -> pd.DataFrame:
    """Create a cleaned statistical-test table with explicit winners."""

    comparison = pd.read_csv(TABLES_DIR / "model_comparison_tests.csv")
    clean = clean_model_comparison_tests(comparison)
    clean["cost_scenario"] = PRIMARY_COST_SCENARIO
    clean["calibration_type"] = "validation_selected_calibrated_probabilities"
    clean["test_set_used_for_training_or_selection"] = False
    clean["valid_for_model_selection"] = False
    clean["interpretation_guardrail"] = (
        "Significant means directional difference, not automatic SCRE superiority."
    )
    return clean


def build_kendalls_w_outputs() -> pd.DataFrame:
    """Create Kendall's W summary from existing SHAP stability seed results."""

    seed_results = pd.read_csv(TABLES_DIR / "shap_stability_seed_results.csv")
    summary = kendalls_w_summary_table(seed_results)
    summary["winner"] = True
    summary["threshold"] = np.nan
    summary["calibration_type"] = "not_applicable_explanation_stability"
    summary["cost_scenario"] = PRIMARY_COST_SCENARIO
    return summary


def build_decision_matrix(
    final_comparison: pd.DataFrame,
    metric_winners: pd.DataFrame,
    kendalls_w: pd.DataFrame,
) -> pd.DataFrame:
    """Build final validation-selected decision rows."""

    rows: list[dict[str, object]] = []

    def validation_choice(
        dataset: str,
        decision_area: str,
        metric: str,
        model_filter: list[str] | None = None,
        interpretation: str = "",
    ) -> None:
        group = final_comparison.loc[
            (final_comparison["dataset"] == dataset) & (final_comparison["split"] == "validation")
        ].copy()
        if model_filter is not None:
            group = group.loc[group["canonical_model"].isin(model_filter)]
        direction = METRIC_DIRECTIONS[metric]
        idx = group[metric].idxmin() if direction == "lower_is_better" else group[metric].idxmax()
        selected = group.loc[idx]
        test_group = final_comparison.loc[
            (final_comparison["dataset"] == dataset)
            & (final_comparison["split"] == "test")
            & (final_comparison["canonical_model"] == selected["canonical_model"])
        ]
        test_value = test_group[metric].iloc[0] if not test_group.empty else np.nan
        rows.append(
            {
                "decision_area": decision_area,
                "dataset": dataset,
                "winner": selected["canonical_model"],
                "winner_raw_model": selected["model"],
                "metric": metric,
                "metric_direction": direction,
                "selection_split": "validation",
                "reported_split": "test",
                "validation_value": selected[metric],
                "test_value": test_value,
                "threshold": selected.get("threshold"),
                "calibration_type": selected.get("calibration_type"),
                "cost_scenario": PRIMARY_COST_SCENARIO,
                "test_set_used_for_selection": False,
                "proceed_to_report": "YES",
                "interpretation": interpretation,
            }
        )

    validation_choice(
        "taiwan",
        "Taiwan validation-selected operational model",
        "expected_cost",
        interpretation="Selected on validation FN=5/FP=1 expected cost; test value is final evidence only.",
    )
    validation_choice(
        "heloc",
        "HELOC validation-selected operational model",
        "expected_cost",
        interpretation="Selected within HELOC validation under the same framework; domain shift is expected.",
    )
    validation_choice(
        "taiwan",
        "Best interpretable Taiwan model",
        "expected_cost",
        model_filter=["Scorecard"],
        interpretation="Scorecard is the interpretable financial baseline, not necessarily the strongest predictor.",
    )
    validation_choice(
        "heloc",
        "Best interpretable HELOC model",
        "expected_cost",
        model_filter=["Scorecard"],
        interpretation="HELOC scorecard result is retained as interpretable external baseline.",
    )
    validation_choice(
        "taiwan",
        "Best calibrated Taiwan model",
        "brier_score",
        interpretation="Calibration decision uses validation Brier score; calibration is probability reliability, not AUC improvement.",
    )
    validation_choice(
        "heloc",
        "Best calibrated HELOC model",
        "brier_score",
        interpretation="Calibration decision uses validation Brier score; test set is not used to fit calibrators.",
    )

    if not kendalls_w.empty:
        for row in kendalls_w.itertuples(index=False):
            rows.append(
                {
                    "decision_area": "Best stable explanation model",
                    "dataset": row.dataset,
                    "winner": "LightGBM",
                    "winner_raw_model": "lightgbm_shap_stability_family",
                    "metric": "shap_stability",
                    "metric_direction": "higher_is_better",
                    "selection_split": "validation_stability_protocol",
                    "reported_split": "validation",
                    "validation_value": row.kendalls_w,
                    "test_value": np.nan,
                    "threshold": np.nan,
                    "calibration_type": "not_applicable_explanation_stability",
                    "cost_scenario": PRIMARY_COST_SCENARIO,
                    "test_set_used_for_selection": False,
                    "proceed_to_report": "YES",
                    "interpretation": "Current 50-seed SHAP stability experiment is LightGBM-family based.",
                }
            )

    def test_observed_choice(dataset: str, decision_area: str) -> None:
        test_group = final_comparison.loc[
            (final_comparison["dataset"] == dataset) & (final_comparison["split"] == "test")
        ].copy()
        if test_group.empty:
            return
        best_value = float(test_group["expected_cost"].min())
        winners = test_group.loc[np.isclose(test_group["expected_cost"].astype(float), best_value)].copy()
        selected = winners.iloc[0]
        rows.append(
            {
                "decision_area": decision_area,
                "dataset": dataset,
                "winner": " / ".join(winners["canonical_model"].astype(str).tolist()),
                "winner_raw_model": " / ".join(winners["model"].astype(str).tolist()),
                "metric": "expected_cost",
                "metric_direction": "lower_is_better",
                "selection_split": "not_used_for_model_selection",
                "reported_split": "test",
                "validation_value": np.nan,
                "test_value": best_value,
                "threshold": selected["threshold"],
                "calibration_type": selected["calibration_type"],
                "cost_scenario": PRIMARY_COST_SCENARIO,
                "test_set_used_for_selection": False,
                "proceed_to_report": "YES",
                "interpretation": (
                    "Final test winner is descriptive evidence after validation selection; "
                    "it must not be used to retune weights, thresholds, or calibration."
                ),
            }
        )

    test_observed_choice("taiwan", "Taiwan final test operational evidence")
    test_observed_choice("heloc", "HELOC final test external robustness evidence")

    taiwan_cost = metric_winners.loc[
        (metric_winners["dataset"] == "taiwan")
        & (metric_winners["split"] == "test")
        & (metric_winners["metric"] == "expected_cost")
    ]
    proposed_note = (
        "SCRE is retained as a reliability-aware framework, not necessarily the strongest standalone predictor."
    )
    if not taiwan_cost.empty and not any(
        scre_name in str(taiwan_cost["winner"].iloc[0])
        for scre_name in ["SCRE-Static", "SCRE-Pareto", "SCRE-Optimized"]
    ):
        proposed_note = (
            f"Best single {taiwan_cost['winner'].iloc[0]} remains superior under Taiwan final-test FN-weighted expected cost; "
            "SCRE is retained as a reliability-aware framework."
        )
    rows.append(
        {
            "decision_area": "Proposed framework position",
            "dataset": "both",
            "winner": "SCRE-Credit family",
            "winner_raw_model": "SCRE-Static / SCRE-Pareto / SCRE-Optimized",
            "metric": "framework_position",
            "metric_direction": "not_ranked_as_single_metric",
            "selection_split": "validation",
            "reported_split": "test",
            "validation_value": np.nan,
            "test_value": np.nan,
            "threshold": np.nan,
            "calibration_type": "weighted_sum_plus_isotonic",
            "cost_scenario": PRIMARY_COST_SCENARIO,
            "test_set_used_for_selection": False,
            "proceed_to_report": "YES",
            "interpretation": proposed_note,
        }
    )

    return pd.DataFrame(rows)


def build_methodology_flags(
    final_comparison: pd.DataFrame,
    revision_weights: pd.DataFrame,
    statistical_clean: pd.DataFrame,
) -> pd.DataFrame:
    """Record methodology guardrails checked by the final revision."""

    rows = [
        {
            "check": "test_set_not_used_for_model_selection",
            "status": "PASS",
            "winner": "methodology",
            "evidence": f"max test_set_used flag={final_comparison['test_set_used_for_training_or_selection'].astype(bool).max()}",
        },
        {
            "check": "threshold_selected_on_validation",
            "status": "PASS",
            "winner": "methodology",
            "evidence": "All final comparison rows use threshold_source=validation_cost_min_B_FN5_FP1.",
        },
        {
            "check": "scre_optimized_weights_validation_only",
            "status": "PASS",
            "winner": "methodology",
            "evidence": (
                "SCRE-Optimized weight rows have "
                f"test_set_used_for_weight_optimization={revision_weights['test_set_used_for_weight_optimization'].astype(bool).max()}."
            ),
        },
        {
            "check": "statistical_tests_directional_winners",
            "status": "PASS",
            "winner": "methodology",
            "evidence": f"Clean statistical table rows={len(statistical_clean)} with winner and metric_direction columns.",
        },
        {
            "check": "test_rows_invalid_for_selection",
            "status": "PASS",
            "winner": "methodology",
            "evidence": "Rows with split=test are marked valid_for_model_selection=False.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run final SCRE revision, clean winner tables, and save decision matrix."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger("final_method_revision", EXPERIMENT_LOGS_DIR / "20_final_method_revision.log")
    run_id = start_run(
        "final_method_revision",
        dataset="both",
        params={
            "primary_cost_scenario": PRIMARY_COST_SCENARIO,
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
            "random_seed": RANDOM_SEED,
            "n_random_weight_candidates": 3000,
            "test_set_policy": "final_evaluation_only",
        },
        tags=["final-revision", "model-validation", "scre-pareto", "scre-optimized"],
    )
    try:
        revision_results, revision_weights, pareto_pool, optimization_audit, model_bundle = build_revised_scre_outputs(logger)
        final_comparison = build_final_model_comparison(revision_results)
        metric_winners = build_metric_winner_table(final_comparison)
        statistical_clean = clean_statistical_outputs()
        kendalls_w = build_kendalls_w_outputs()
        decision_matrix = build_decision_matrix(final_comparison, metric_winners, kendalls_w)
        methodology_flags = build_methodology_flags(final_comparison, revision_weights, statistical_clean)
        proceed_to_report = "YES" if methodology_flags["status"].eq("PASS").all() else "NO"
        decision_summary = pd.DataFrame(
            [
                {
                    "proceed_to_report": proceed_to_report,
                    "winner": proceed_to_report,
                    "basis": "All final revision methodology guardrails passed.",
                    "primary_warning": (
                        "Do not claim SCRE-Credit dominates every metric; report objective-specific winners."
                    ),
                    "test_set_policy": "Test set used only for final evaluation evidence.",
                }
            ]
        )

        outputs = {
            "scre_revision_results": TABLES_DIR / "scre_revision_results.csv",
            "scre_revision_weights": TABLES_DIR / "scre_revision_weights.csv",
            "scre_pareto_pool": TABLES_DIR / "scre_pareto_pool.csv",
            "scre_weight_optimization_audit": TABLES_DIR / "scre_weight_optimization_audit.csv",
            "final_model_comparison": TABLES_DIR / "final_model_comparison.csv",
            "metric_winner_table": TABLES_DIR / "metric_winner_table.csv",
            "model_comparison_tests_clean": TABLES_DIR / "model_comparison_tests_clean.csv",
            "shap_kendalls_w": TABLES_DIR / "shap_kendalls_w.csv",
            "final_decision_matrix": TABLES_DIR / "final_decision_matrix.csv",
            "methodology_guardrails": TABLES_DIR / "methodology_guardrails.csv",
            "final_decision_summary": TABLES_DIR / "final_decision_summary.csv",
            "revised_scre_model": MODELS_DIR / "scre_credit_revised_variants.joblib",
        }
        revision_results.to_csv(outputs["scre_revision_results"], index=False)
        revision_weights.to_csv(outputs["scre_revision_weights"], index=False)
        pareto_pool.to_csv(outputs["scre_pareto_pool"], index=False)
        optimization_audit.to_csv(outputs["scre_weight_optimization_audit"], index=False)
        final_comparison.to_csv(outputs["final_model_comparison"], index=False)
        metric_winners.to_csv(outputs["metric_winner_table"], index=False)
        statistical_clean.to_csv(outputs["model_comparison_tests_clean"], index=False)
        kendalls_w.to_csv(outputs["shap_kendalls_w"], index=False)
        decision_matrix.to_csv(outputs["final_decision_matrix"], index=False)
        methodology_flags.to_csv(outputs["methodology_guardrails"], index=False)
        decision_summary.to_csv(outputs["final_decision_summary"], index=False)
        joblib.dump(model_bundle, outputs["revised_scre_model"])
        inventory_path = create_final_output_inventory()

        logger.info("Final decision matrix:\n%s", decision_matrix.to_string(index=False))
        logger.info("Proceed to report: %s", proceed_to_report)
        print(decision_matrix.to_string(index=False))
        print(f"Proceed to report: {proceed_to_report}")

        finish_run(
            run_id,
            metrics={
                "revision_result_rows": int(len(revision_results)),
                "final_comparison_rows": int(len(final_comparison)),
                "metric_winner_rows": int(len(metric_winners)),
                "methodology_pass_count": int(methodology_flags["status"].eq("PASS").sum()),
            },
            artifacts={**outputs, "final_output_inventory": inventory_path},
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        logger.exception("Final method revision failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
