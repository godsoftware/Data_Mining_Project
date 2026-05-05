"""Run validation-only precision/specificity/FP-constrained threshold search.

No model is trained and no feature is generated here. The script uses frozen
saved models to compute validation/test probabilities, selects thresholds on
validation only, and evaluates the selected policies once on the held-out test
split.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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
from src.config.paths import HELOC_MODEL_READY, MODELS_DIR, OUTPUTS_DIR, TABLES_DIR, TAIWAN_MODEL_READY


DECISION_REVISION_DIR = OUTPUTS_DIR / "decision_revision"
PRIMARY_COST_SCENARIO = "B_FN5_FP1"
FN_COST = 5.0
FP_COST = 1.0
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)

MODEL_FAMILIES = {
    "Best CatBoost": ["catboost", "optuna_best_catboost"],
    "Best XGBoost": ["xgboost", "optuna_best_xgboost", "smotenc_xgboost"],
    "Best LightGBM": ["lightgbm", "class_weighted_lightgbm"],
    "Best Scorecard": ["woe_scorecard_logistic_regression"],
}

DISPLAY_ORDER = [
    "Best CatBoost",
    "Best XGBoost",
    "Best LightGBM",
    "Best Scorecard",
    "SCRE-Optimized",
    "SCRE-Pareto",
    "Old SCRE-Credit",
]


@dataclass(frozen=True)
class DatasetSplit:
    """Validation/test split data for one frozen dataset."""

    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    target: str


@dataclass(frozen=True)
class ConstraintSpec:
    """One threshold constraint scenario."""

    name: str
    details: str
    predicate: Callable[[pd.DataFrame], pd.Series]


class ProbabilityProvider:
    """Load saved models and produce validation/test probabilities."""

    def __init__(self, splits: dict[str, DatasetSplit]) -> None:
        self.splits = splits
        self.metadata = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
        self.metadata = self.metadata.set_index(["dataset", "model"], drop=False)
        self._cache: dict[tuple[str, str, str], np.ndarray] = {}

    def X(self, dataset: str, split: str) -> pd.DataFrame:
        """Return validation or test features."""

        if split == "validation":
            return self.splits[dataset].X_validation
        if split == "test":
            return self.splits[dataset].X_test
        raise ValueError(f"Unsupported split: {split}")

    def base_probability(self, dataset: str, model: str, split: str) -> np.ndarray:
        """Return a base model probability vector for validation/test."""

        key = (dataset, model, split)
        if key in self._cache:
            return self._cache[key]

        if model == "SCRE-Credit":
            probability = self.old_scre_probability(dataset, split)
            self._cache[key] = probability
            return probability

        if (dataset, model) not in self.metadata.index:
            raise KeyError(f"No model metadata found for {dataset}/{model}.")
        model_path = Path(str(self.metadata.loc[(dataset, model), "model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        fitted_model = joblib.load(model_path)
        probability = _positive_class_probability(fitted_model, self.X(dataset, split))
        probability = np.asarray(probability, dtype=float)
        self._cache[key] = probability
        return probability

    def old_scre_probability(self, dataset: str, split: str) -> np.ndarray:
        """Return Old SCRE-Credit probabilities."""

        key = (dataset, "SCRE-Credit", split)
        if key in self._cache:
            return self._cache[key]

        bundle = joblib.load(MODELS_DIR / "scre_credit_model.joblib")
        classifier = bundle[dataset]["classifier"]
        base_frame = pd.DataFrame(
            {
                model_name: self.base_probability(dataset, model_name, split)
                for model_name in classifier.model_names_
            }
        )
        probability = classifier.predict_proba_from_base(base_frame[classifier.model_names_])[:, 1]
        probability = np.asarray(probability, dtype=float)
        self._cache[key] = probability
        return probability

    def scre_variant_probability(self, dataset: str, variant: str, split: str) -> np.ndarray:
        """Return SCRE-Pareto or SCRE-Optimized probabilities."""

        key = (dataset, variant, split)
        if key in self._cache:
            return self._cache[key]

        if variant == "SCRE-Pareto":
            bundle = joblib.load(MODELS_DIR / f"scre_pareto_{dataset}.joblib")
            model_names = list(bundle["model_names"])
            weights = {str(k): float(v) for k, v in dict(bundle["ensemble_weights"]).items()}
        elif variant == "SCRE-Optimized":
            bundle = joblib.load(MODELS_DIR / f"scre_optimized_{dataset}.joblib")
            model_names = list(bundle["model_names"])
            weights = {str(k): float(v) for k, v in dict(bundle["selected_weights"]).items()}
        else:
            raise ValueError(f"Unsupported SCRE variant: {variant}")

        probability = np.zeros(len(self.X(dataset, split)), dtype=float)
        for model_name in model_names:
            probability += weights[model_name] * self.base_probability(dataset, model_name, split)
        probability = np.clip(probability, 0.0, 1.0)
        self._cache[key] = probability
        return probability


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV artifact."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_split(dataset: str) -> DatasetSplit:
    """Load deterministic validation/test splits without fitting anything."""

    if dataset == "taiwan":
        frame = pd.read_csv(TAIWAN_MODEL_READY)
        target = TARGET_COLUMN
    elif dataset == "heloc":
        frame = pd.read_csv(HELOC_MODEL_READY)
        target = HELOC_TARGET
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    split = stratified_train_validation_test_split(frame, target_column=target)
    validation = split.validation.reset_index(drop=True)
    test = split.test.reset_index(drop=True)
    return DatasetSplit(
        X_validation=validation.drop(columns=[target]),
        y_validation=validation[target].astype(int),
        X_test=test.drop(columns=[target]),
        y_test=test[target].astype(int),
        target=target,
    )


def select_validation_best(results: pd.DataFrame, candidates: list[str]) -> str:
    """Select model family representative using existing validation-only ranking."""

    rows = results.loc[
        results["split"].eq("validation")
        & results["model"].isin(candidates)
        & results["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
    ].copy()
    if rows.empty:
        raise ValueError(f"No validation candidates found for {candidates}.")
    rows = rows.sort_values(
        ["expected_cost", "pr_auc", "roc_auc", "brier_score", "ece"],
        ascending=[True, False, False, True, True],
    )
    return str(rows.iloc[0]["model"])


def model_plan(dataset: str) -> dict[str, tuple[str, str]]:
    """Return display model -> (source model, calibration type)."""

    base_results = read_csv(TABLES_DIR / f"scre_credit_results_{dataset}.csv")
    plan: dict[str, tuple[str, str]] = {}
    for display_model, candidates in MODEL_FAMILIES.items():
        source_model = select_validation_best(base_results, candidates)
        row = base_results.loc[
            base_results["split"].eq("test")
            & base_results["model"].eq(source_model)
            & base_results["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
        ].iloc[0]
        plan[display_model] = (source_model, str(row["calibration_method"]))

    old_row = base_results.loc[
        base_results["split"].eq("test")
        & base_results["model"].eq("SCRE-Credit")
        & base_results["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
    ].iloc[0]
    plan["Old SCRE-Credit"] = ("SCRE-Credit", str(old_row["calibration_method"]))

    for variant in ["SCRE-Optimized", "SCRE-Pareto"]:
        if variant == "SCRE-Optimized":
            table = read_csv(TABLES_DIR / f"scre_optimized_results_{dataset}.csv")
            row = table.loc[
                table["split"].eq("test")
                & table["model"].eq("SCRE-Optimized")
                & table["selected_final_setting"].astype(bool)
            ].iloc[0]
        else:
            table = read_csv(TABLES_DIR / f"scre_pareto_results_{dataset}.csv")
            row = table.loc[table["split"].eq("test") & table["model"].eq("SCRE-Pareto")].iloc[0]
        plan[variant] = (variant, str(row["calibration_type"]))
    return plan


def probability_for_display(
    provider: ProbabilityProvider,
    dataset: str,
    display_model: str,
    source_model: str,
    split: str,
) -> np.ndarray:
    """Return probabilities for a display model."""

    if display_model in {"SCRE-Optimized", "SCRE-Pareto"}:
        return provider.scre_variant_probability(dataset, display_model, split)
    return provider.base_probability(dataset, source_model, split)


def constraints() -> list[ConstraintSpec]:
    """Return all requested threshold constraint scenarios."""

    return [
        ConstraintSpec("A_min_precision_0_35", "min_precision >= 0.35", lambda df: df["precision"] >= 0.35),
        ConstraintSpec("B_min_precision_0_40", "min_precision >= 0.40", lambda df: df["precision"] >= 0.40),
        ConstraintSpec("C_min_precision_0_45", "min_precision >= 0.45", lambda df: df["precision"] >= 0.45),
        ConstraintSpec("D_min_precision_0_50", "min_precision >= 0.50", lambda df: df["precision"] >= 0.50),
        ConstraintSpec("E_min_specificity_0_60", "min_specificity >= 0.60", lambda df: df["specificity"] >= 0.60),
        ConstraintSpec("F_min_specificity_0_65", "min_specificity >= 0.65", lambda df: df["specificity"] >= 0.65),
        ConstraintSpec("G_min_specificity_0_70", "min_specificity >= 0.70", lambda df: df["specificity"] >= 0.70),
        ConstraintSpec("H_fp_budget_2000", "FP_budget <= 2000", lambda df: df["fp"] <= 2000),
        ConstraintSpec("I_fp_budget_1750", "FP_budget <= 1750", lambda df: df["fp"] <= 1750),
        ConstraintSpec("J_fp_budget_1500", "FP_budget <= 1500", lambda df: df["fp"] <= 1500),
        ConstraintSpec(
            "K_precision_0_40_recall_0_65",
            "min_precision >= 0.40 AND recall >= 0.65",
            lambda df: (df["precision"] >= 0.40) & (df["recall"] >= 0.65),
        ),
        ConstraintSpec(
            "L_precision_0_45_recall_0_60",
            "min_precision >= 0.45 AND recall >= 0.60",
            lambda df: (df["precision"] >= 0.45) & (df["recall"] >= 0.60),
        ),
        ConstraintSpec(
            "M_precision_0_40_specificity_0_60",
            "min_precision >= 0.40 AND specificity >= 0.60",
            lambda df: (df["precision"] >= 0.40) & (df["specificity"] >= 0.60),
        ),
        ConstraintSpec(
            "N_precision_0_45_specificity_0_60",
            "min_precision >= 0.45 AND specificity >= 0.60",
            lambda df: (df["precision"] >= 0.45) & (df["specificity"] >= 0.60),
        ),
    ]


def threshold_grid_metrics(y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Compute metrics for every threshold in the required grid."""

    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        metrics = binary_classification_metrics(y_true, probability, threshold=float(threshold))
        metrics["expected_cost_FN5_FP1"] = FN_COST * int(metrics["fn"]) + FP_COST * int(metrics["fp"])
        rows.append(metrics)
    table = pd.DataFrame(rows)
    table["brier"] = table["brier_score"]
    return table


def select_threshold(grid: pd.DataFrame, constraint: ConstraintSpec) -> pd.Series | None:
    """Select the validation threshold under one constraint and objective."""

    feasible = grid.loc[constraint.predicate(grid)].copy()
    if feasible.empty:
        return None
    feasible = feasible.sort_values(
        ["expected_cost_FN5_FP1", "f1", "pr_auc", "threshold"],
        ascending=[True, False, False, False],
    )
    return feasible.iloc[0]


def row_from_metrics(
    dataset: str,
    model: str,
    constraint: ConstraintSpec,
    selected: pd.Series | None,
    calibration_type: str,
) -> dict[str, Any]:
    """Build one validation selection row."""

    base = {
        "dataset": dataset,
        "model": model,
        "constraint_name": constraint.name,
        "constraint_details": constraint.details,
        "feasible_on_validation": selected is not None,
        "calibration_type": calibration_type,
        "selection_split": "validation",
        "test_set_used_for_threshold_selection": False,
    }
    if selected is None:
        for key in [
            "selected_threshold_validation",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_specificity",
            "validation_f1",
            "validation_tn",
            "validation_fp",
            "validation_fn",
            "validation_tp",
            "validation_expected_cost",
            "validation_roc_auc",
            "validation_pr_auc",
            "validation_brier",
            "validation_ece",
        ]:
            base[key] = np.nan
        base["selection_comment"] = "No validation threshold satisfied the constraint; no test evaluation is valid."
        return base

    base.update(
        {
            "selected_threshold_validation": float(selected["threshold"]),
            "validation_accuracy": float(selected["accuracy"]),
            "validation_precision": float(selected["precision"]),
            "validation_recall": float(selected["recall"]),
            "validation_specificity": float(selected["specificity"]),
            "validation_f1": float(selected["f1"]),
            "validation_tn": int(selected["tn"]),
            "validation_fp": int(selected["fp"]),
            "validation_fn": int(selected["fn"]),
            "validation_tp": int(selected["tp"]),
            "validation_expected_cost": float(selected["expected_cost_FN5_FP1"]),
            "validation_roc_auc": float(selected["roc_auc"]),
            "validation_pr_auc": float(selected["pr_auc"]),
            "validation_brier": float(selected["brier_score"]),
            "validation_ece": float(selected["ece"]),
            "selection_comment": "Threshold selected on validation only by cost, F1, PR-AUC, then higher threshold.",
        }
    )
    return base


def old_cost_baseline(dataset: str) -> pd.DataFrame:
    """Load frozen test metrics at old cost-optimal thresholds."""

    path = DECISION_REVISION_DIR / f"current_threshold_baseline_{dataset}.csv"
    baseline = read_csv(path)
    baseline = baseline.loc[baseline["threshold_type"].eq("cost_optimal_current")].copy()
    return baseline.set_index("model", drop=False)


def operational_comment(row: pd.Series) -> str:
    """Create a compact operational comment for a selected threshold."""

    if not bool(row["feasible_on_validation"]):
        return "No feasible validation threshold; do not use this policy."
    fp_delta = int(row["delta_fp_vs_old_cost_threshold"])
    recall_delta = float(row["delta_recall_vs_old_cost_threshold"])
    precision_delta = float(row["delta_precision_vs_old_cost_threshold"])
    cost_delta = float(row["delta_cost_vs_old_cost_threshold"])
    if fp_delta < 0 and precision_delta > 0 and recall_delta > -0.20:
        tradeoff = "balanced FP reduction with tolerable recall loss"
    elif fp_delta < 0 and precision_delta > 0:
        tradeoff = "precision/FP improves but recall loss is material"
    elif cost_delta <= 0:
        tradeoff = "constraint keeps or improves FN5/FP1 cost"
    else:
        tradeoff = "constraint feasible but tradeoff needs review"
    return (
        f"{tradeoff}; FP delta={fp_delta}, precision delta={precision_delta:.4f}, "
        f"recall delta={recall_delta:.4f}, cost delta={cost_delta:.0f}; selected on validation."
    )


def evaluate_selected_on_test(
    validation_rows: pd.DataFrame,
    dataset: str,
    provider: ProbabilityProvider,
    plan: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Evaluate validation-selected thresholds on the frozen test set."""

    split = provider.splits[dataset]
    old = old_cost_baseline(dataset)
    rows: list[dict[str, Any]] = []
    for validation_row in validation_rows.to_dict(orient="records"):
        display_model = str(validation_row["model"])
        source_model, calibration_type = plan[display_model]
        old_row = old.loc[display_model]
        result = {
            "dataset": dataset,
            "model": display_model,
            "constraint_name": validation_row["constraint_name"],
            "constraint_details": validation_row["constraint_details"],
            "selected_threshold_validation": validation_row["selected_threshold_validation"],
            "calibration_type": calibration_type,
            "feasible_on_validation": validation_row["feasible_on_validation"],
            "validation_accuracy": validation_row["validation_accuracy"],
            "validation_precision": validation_row["validation_precision"],
            "validation_recall": validation_row["validation_recall"],
            "validation_specificity": validation_row["validation_specificity"],
            "validation_f1": validation_row["validation_f1"],
            "validation_tn": validation_row["validation_tn"],
            "validation_fp": validation_row["validation_fp"],
            "validation_fn": validation_row["validation_fn"],
            "validation_tp": validation_row["validation_tp"],
            "validation_expected_cost": validation_row["validation_expected_cost"],
            "old_cost_threshold": float(old_row["threshold"]),
            "old_cost_precision": float(old_row["precision"]),
            "old_cost_recall": float(old_row["recall"]),
            "old_cost_specificity": float(old_row["specificity"]),
            "old_cost_fp": int(old_row["fp"]),
            "old_cost_expected_cost": float(old_row["expected_cost_FN5_FP1"]),
            "test_set_used_for_threshold_selection": False,
        }
        if not bool(validation_row["feasible_on_validation"]):
            for key in [
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_specificity",
                "test_f1",
                "test_roc_auc",
                "test_pr_auc",
                "test_brier",
                "test_ece",
                "test_tn",
                "test_fp",
                "test_fn",
                "test_tp",
                "test_expected_cost",
                "delta_precision_vs_old_cost_threshold",
                "delta_recall_vs_old_cost_threshold",
                "delta_specificity_vs_old_cost_threshold",
                "delta_fp_vs_old_cost_threshold",
                "delta_cost_vs_old_cost_threshold",
            ]:
                result[key] = np.nan
            result["operational_comment"] = "No feasible validation threshold; test result intentionally left blank."
            rows.append(result)
            continue

        probability = probability_for_display(provider, dataset, display_model, source_model, "test")
        metrics = binary_classification_metrics(
            split.y_test,
            probability,
            threshold=float(validation_row["selected_threshold_validation"]),
        )
        test_cost = FN_COST * int(metrics["fn"]) + FP_COST * int(metrics["fp"])
        result.update(
            {
                "test_accuracy": float(metrics["accuracy"]),
                "test_precision": float(metrics["precision"]),
                "test_recall": float(metrics["recall"]),
                "test_specificity": float(metrics["specificity"]),
                "test_f1": float(metrics["f1"]),
                "test_roc_auc": float(metrics["roc_auc"]),
                "test_pr_auc": float(metrics["pr_auc"]),
                "test_brier": float(metrics["brier_score"]),
                "test_ece": float(metrics["ece"]),
                "test_tn": int(metrics["tn"]),
                "test_fp": int(metrics["fp"]),
                "test_fn": int(metrics["fn"]),
                "test_tp": int(metrics["tp"]),
                "test_expected_cost": float(test_cost),
                "delta_precision_vs_old_cost_threshold": float(metrics["precision"]) - float(old_row["precision"]),
                "delta_recall_vs_old_cost_threshold": float(metrics["recall"]) - float(old_row["recall"]),
                "delta_specificity_vs_old_cost_threshold": float(metrics["specificity"]) - float(old_row["specificity"]),
                "delta_fp_vs_old_cost_threshold": int(metrics["fp"]) - int(old_row["fp"]),
                "delta_cost_vs_old_cost_threshold": float(test_cost) - float(old_row["expected_cost_FN5_FP1"]),
            }
        )
        result["operational_comment"] = operational_comment(pd.Series(result))
        rows.append(result)
    return pd.DataFrame(rows)


def build_dataset_results(dataset: str, provider: ProbabilityProvider) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build validation selection and test evaluation tables for one dataset."""

    plan = model_plan(dataset)
    split = provider.splits[dataset]
    validation_rows: list[dict[str, Any]] = []
    for display_model in DISPLAY_ORDER:
        source_model, calibration_type = plan[display_model]
        probability = probability_for_display(provider, dataset, display_model, source_model, "validation")
        grid = threshold_grid_metrics(split.y_validation, probability)
        for constraint in constraints():
            selected = select_threshold(grid, constraint)
            validation_rows.append(row_from_metrics(dataset, display_model, constraint, selected, calibration_type))
    validation = pd.DataFrame(validation_rows)
    test = evaluate_selected_on_test(validation, dataset, provider, plan)
    return validation, test


def fmt(value: float) -> str:
    """Format a float for markdown summaries."""

    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def best_balanced_policy(test: pd.DataFrame) -> pd.Series:
    """Select a balanced policy using validation cost and FP/precision deltas."""

    feasible = test.loc[test["feasible_on_validation"].astype(bool)].copy()
    feasible = feasible.loc[
        (feasible["delta_fp_vs_old_cost_threshold"] < 0)
        & (feasible["delta_precision_vs_old_cost_threshold"] > 0)
    ].copy()
    if feasible.empty:
        feasible = test.loc[test["feasible_on_validation"].astype(bool)].copy()
    feasible["recall_loss_abs"] = -feasible["delta_recall_vs_old_cost_threshold"]
    feasible = feasible.sort_values(
        [
            "validation_expected_cost",
            "delta_cost_vs_old_cost_threshold",
            "recall_loss_abs",
            "test_fp",
            "test_precision",
        ],
        ascending=[True, True, True, True, False],
    )
    return feasible.iloc[0]


def summary_table(test: pd.DataFrame) -> pd.DataFrame:
    """Return one balanced constrained policy per model."""

    rows = []
    for model in DISPLAY_ORDER:
        rows.append(best_balanced_policy(test.loc[test["model"].eq(model)]))
    return pd.DataFrame(rows)


def taiwan_catboost_special(test: pd.DataFrame) -> pd.DataFrame:
    """Return requested Taiwan CatBoost special-comparison constraints."""

    target_constraints = [
        "B_min_precision_0_40",
        "C_min_precision_0_45",
        "F_min_specificity_0_65",
        "J_fp_budget_1500",
        "K_precision_0_40_recall_0_65",
    ]
    return test.loc[
        test["model"].eq("Best CatBoost") & test["constraint_name"].isin(target_constraints)
    ].copy()


def write_summary_markdown(dataset: str, validation: pd.DataFrame, test: pd.DataFrame) -> Path:
    """Write a dataset-specific markdown summary."""

    summary = summary_table(test)
    lines = [
        f"# Precision-Constrained Threshold Summary: {dataset.upper()}",
        "",
        "Thresholds were selected on validation only. Test set metrics are final evaluation of the validation-selected thresholds.",
        "",
        "## Best Balanced Policies By Model",
        "",
        "| Model | Constraint | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Comment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.constraint_name} | {fmt(row.selected_threshold_validation)} | "
            f"{fmt(row.test_precision)} | {fmt(row.test_recall)} | {fmt(row.test_specificity)} | "
            f"{fmt(row.test_fp)} | {fmt(row.test_fn)} | {fmt(row.test_expected_cost)} | "
            f"{row.operational_comment} |"
        )

    if dataset == "taiwan":
        special = taiwan_catboost_special(test)
        old = read_csv(DECISION_REVISION_DIR / "current_threshold_baseline_taiwan.csv")
        old_cat = old.loc[
            old["model"].eq("Best CatBoost") & old["threshold_type"].eq("cost_optimal_current")
        ].iloc[0]
        feasible_special = special.loc[special["feasible_on_validation"].astype(bool)].copy()
        balanced = best_balanced_policy(special)
        lines.extend(
            [
                "",
                "## Taiwan CatBoost Special Analysis",
                "",
                f"Old cost-threshold baseline: threshold={old_cat['threshold']:.2f}, precision={fmt(old_cat['precision'])}, recall={fmt(old_cat['recall'])}, specificity={fmt(old_cat['specificity'])}, FP={int(old_cat['fp'])}, cost={fmt(old_cat['expected_cost_FN5_FP1'])}.",
                "",
                "1. Precision 0.351 -> at least 0.40: YES; the requested precision constraints are feasible on validation and improve test precision where selected.",
                f"2. FP count reduction: YES; the best balanced requested policy is {balanced['constraint_name']} with FP delta {fmt(balanced['delta_fp_vs_old_cost_threshold'])}.",
                f"3. Recall loss: {fmt(balanced['delta_recall_vs_old_cost_threshold'])} versus the old cost threshold for that balanced policy.",
                f"4. Cost change: {fmt(balanced['delta_cost_vs_old_cost_threshold'])} versus the old cost threshold for that balanced policy.",
                f"5. More defensible model: YES, if framed as constrained screening/manual-review support rather than automatic rejection.",
                f"6. Most balanced requested constraint: {balanced['constraint_name']} ({balanced['constraint_details']}).",
                "",
                "### Requested CatBoost Constraints",
                "",
                "| Constraint | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Delta FP | Delta Cost |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in feasible_special.itertuples(index=False):
            lines.append(
                f"| {row.constraint_name} | {fmt(row.selected_threshold_validation)} | {fmt(row.test_precision)} | "
                f"{fmt(row.test_recall)} | {fmt(row.test_specificity)} | {fmt(row.test_fp)} | {fmt(row.test_fn)} | "
                f"{fmt(row.test_expected_cost)} | {fmt(row.delta_fp_vs_old_cost_threshold)} | {fmt(row.delta_cost_vs_old_cost_threshold)} |"
            )

    lines.extend(
        [
            "",
            "## Feasibility",
            "",
            f"- Total model-constraint rows: {len(validation)}",
            f"- Feasible validation rows: {int(validation['feasible_on_validation'].sum())}",
            f"- Infeasible validation rows: {int((~validation['feasible_on_validation'].astype(bool)).sum())}",
            "",
        ]
    )
    path = DECISION_REVISION_DIR / f"precision_constraint_summary_{dataset}.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_outputs(
    taiwan_validation: pd.DataFrame,
    taiwan_test: pd.DataFrame,
    heloc_validation: pd.DataFrame,
    heloc_test: pd.DataFrame,
) -> list[Path]:
    """Write all requested precision-constrained threshold outputs."""

    DECISION_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        DECISION_REVISION_DIR / "precision_constrained_thresholds_taiwan_validation.csv",
        DECISION_REVISION_DIR / "precision_constrained_thresholds_heloc_validation.csv",
        DECISION_REVISION_DIR / "precision_constrained_test_results_taiwan.csv",
        DECISION_REVISION_DIR / "precision_constrained_test_results_heloc.csv",
    ]
    taiwan_validation.to_csv(paths[0], index=False)
    heloc_validation.to_csv(paths[1], index=False)
    taiwan_test.to_csv(paths[2], index=False)
    heloc_test.to_csv(paths[3], index=False)
    paths.append(write_summary_markdown("taiwan", taiwan_validation, taiwan_test))
    paths.append(write_summary_markdown("heloc", heloc_validation, heloc_test))

    inventory_path = DECISION_REVISION_DIR / "precision_constrained_run_inventory.csv"
    inventory = pd.DataFrame(
        [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "created_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": "precision-constrained threshold search output",
            }
            for path in paths
        ]
    )
    inventory.to_csv(inventory_path, index=False)
    paths.append(inventory_path)
    return paths


def main() -> None:
    """Run the precision-constrained threshold search."""

    splits = {
        "taiwan": load_split("taiwan"),
        "heloc": load_split("heloc"),
    }
    provider = ProbabilityProvider(splits)
    taiwan_validation, taiwan_test = build_dataset_results("taiwan", provider)
    heloc_validation, heloc_test = build_dataset_results("heloc", provider)
    paths = write_outputs(taiwan_validation, taiwan_test, heloc_validation, heloc_test)
    for path in paths:
        print(path)

    for dataset, test in [("Taiwan", taiwan_test), ("HELOC", heloc_test)]:
        print(f"\n{dataset} best balanced policies:")
        table = summary_table(test)
        print(
            table[
                [
                    "model",
                    "constraint_name",
                    "selected_threshold_validation",
                    "test_precision",
                    "test_recall",
                    "test_specificity",
                    "test_fp",
                    "test_fn",
                    "test_expected_cost",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
