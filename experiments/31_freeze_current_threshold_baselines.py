"""Freeze current threshold baselines for later false-positive reduction work.

This script does not train models, tune thresholds, or modify existing result
tables. It reads frozen final artifacts, loads saved models only for inference
at threshold 0.50, and writes a separate decision-revision baseline package.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
FN_COST = 5.0
FP_COST = 1.0
PRIMARY_COST_SCENARIO = "B_FN5_FP1"

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

OUTPUT_COLUMNS = [
    "dataset",
    "model",
    "threshold_type",
    "threshold",
    "calibration_type",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "tn",
    "fp",
    "fn",
    "tp",
    "expected_cost_FN5_FP1",
    "notes",
]


@dataclass
class DatasetSplit:
    """Frozen split and target metadata for one dataset."""

    X_test: pd.DataFrame
    y_test: pd.Series
    full_y: pd.Series
    target: str


class ProbabilityProvider:
    """Load saved models and produce frozen test-set probabilities."""

    def __init__(self, splits: dict[str, DatasetSplit]) -> None:
        self.splits = splits
        self.metadata = pd.read_csv(TABLES_DIR / "scre_model_weights.csv")
        self.metadata = self.metadata.set_index(["dataset", "model"], drop=False)
        self._probability_cache: dict[tuple[str, str], np.ndarray] = {}

    def base_probability(self, dataset: str, model: str) -> np.ndarray:
        """Return saved-model positive-class probabilities for one base model."""

        key = (dataset, model)
        if key in self._probability_cache:
            return self._probability_cache[key]

        if model == "SCRE-Credit":
            probability = self.old_scre_probability(dataset)
            self._probability_cache[key] = probability
            return probability

        if key not in self.metadata.index:
            raise KeyError(f"No model metadata found for {dataset}/{model}.")
        model_path = Path(str(self.metadata.loc[key, "model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        fitted_model = joblib.load(model_path)
        probability = _positive_class_probability(fitted_model, self.splits[dataset].X_test)
        probability = np.asarray(probability, dtype=float)
        self._probability_cache[key] = probability
        return probability

    def old_scre_probability(self, dataset: str) -> np.ndarray:
        """Return Old SCRE-Credit probabilities from its saved bundle."""

        key = (dataset, "SCRE-Credit")
        if key in self._probability_cache:
            return self._probability_cache[key]

        bundle = joblib.load(MODELS_DIR / "scre_credit_model.joblib")
        classifier = bundle[dataset]["classifier"]
        base_frame = pd.DataFrame(
            {
                model_name: self.base_probability(dataset, model_name)
                for model_name in classifier.model_names_
            }
        )
        probability = classifier.predict_proba_from_base(base_frame[classifier.model_names_])[:, 1]
        probability = np.asarray(probability, dtype=float)
        self._probability_cache[key] = probability
        return probability

    def scre_variant_probability(self, dataset: str, variant: str) -> np.ndarray:
        """Return SCRE-Pareto or SCRE-Optimized frozen probabilities."""

        key = (dataset, variant)
        if key in self._probability_cache:
            return self._probability_cache[key]

        if variant == "SCRE-Pareto":
            bundle_path = MODELS_DIR / f"scre_pareto_{dataset}.joblib"
            model_names_key = "model_names"
            weights_key = "ensemble_weights"
        elif variant == "SCRE-Optimized":
            bundle_path = MODELS_DIR / f"scre_optimized_{dataset}.joblib"
            model_names_key = "model_names"
            weights_key = "selected_weights"
        else:
            raise ValueError(f"Unsupported SCRE variant: {variant}")

        bundle = joblib.load(bundle_path)
        model_names = list(bundle[model_names_key])
        weights = {str(k): float(v) for k, v in dict(bundle[weights_key]).items()}
        missing_weights = sorted(set(model_names) - set(weights))
        if missing_weights:
            raise ValueError(f"Missing SCRE weights for {dataset}/{variant}: {missing_weights}")
        probability = np.zeros(len(self.splits[dataset].y_test), dtype=float)
        for model_name in model_names:
            probability += weights[model_name] * self.base_probability(dataset, model_name)
        probability = np.clip(probability, 0.0, 1.0)
        self._probability_cache[key] = probability
        return probability


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV artifact."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_dataset_split(dataset: str) -> DatasetSplit:
    """Load the frozen deterministic test split for one dataset."""

    if dataset == "taiwan":
        path = TAIWAN_MODEL_READY
        target = TARGET_COLUMN
    elif dataset == "heloc":
        path = HELOC_MODEL_READY
        target = HELOC_TARGET
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    frame = pd.read_csv(path)
    split = stratified_train_validation_test_split(frame, target_column=target)
    test = split.test.reset_index(drop=True)
    return DatasetSplit(
        X_test=test.drop(columns=[target]),
        y_test=test[target].astype(int),
        full_y=frame[target].astype(int),
        target=target,
    )


def select_validation_best(results: pd.DataFrame, candidates: list[str]) -> str:
    """Select a final family representative using validation-only cost ranking."""

    rows = results.loc[
        results["split"].eq("validation")
        & results["model"].isin(candidates)
        & results["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
    ].copy()
    if rows.empty:
        raise ValueError(f"No validation rows found for candidates: {candidates}")
    rows = rows.sort_values(
        ["expected_cost", "pr_auc", "roc_auc", "brier_score", "ece"],
        ascending=[True, False, False, True, True],
    )
    return str(rows.iloc[0]["model"])


def current_base_row(results: pd.DataFrame, source_model: str) -> pd.Series:
    """Return the frozen current cost-optimal test row for a base/Old SCRE model."""

    rows = results.loc[
        results["split"].eq("test")
        & results["model"].eq(source_model)
        & results["cost_scenario"].eq(PRIMARY_COST_SCENARIO)
    ]
    if rows.empty:
        raise ValueError(f"No current test row found for {source_model}.")
    return rows.iloc[0]


def current_scre_variant_row(dataset: str, variant: str) -> pd.Series:
    """Return the frozen current test row for a revised SCRE variant."""

    if variant == "SCRE-Pareto":
        table = read_csv(TABLES_DIR / f"scre_pareto_results_{dataset}.csv")
        rows = table.loc[table["split"].eq("test") & table["model"].eq("SCRE-Pareto")]
    elif variant == "SCRE-Optimized":
        table = read_csv(TABLES_DIR / f"scre_optimized_results_{dataset}.csv")
        rows = table.loc[
            table["split"].eq("test")
            & table["model"].eq("SCRE-Optimized")
            & table["selected_final_setting"].astype(bool)
        ]
    else:
        raise ValueError(f"Unknown SCRE variant: {variant}")
    if rows.empty:
        raise ValueError(f"No current test row found for {dataset}/{variant}.")
    return rows.iloc[0]


def row_from_existing(
    dataset: str,
    model: str,
    threshold_type: str,
    row: pd.Series,
    notes: str,
) -> dict[str, Any]:
    """Convert a frozen result row to the decision-revision schema."""

    calibration_type = row.get("calibration_method", row.get("calibration_type", "unknown"))
    return {
        "dataset": dataset,
        "model": model,
        "threshold_type": threshold_type,
        "threshold": float(row["threshold"]),
        "calibration_type": calibration_type,
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "specificity": float(row["specificity"]),
        "f1": float(row["f1"]),
        "roc_auc": float(row["roc_auc"]),
        "pr_auc": float(row["pr_auc"]),
        "brier": float(row.get("brier", row.get("brier_score"))),
        "ece": float(row["ece"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tp": int(row["tp"]),
        "expected_cost_FN5_FP1": float(row["expected_cost"]),
        "notes": notes,
    }


def row_from_probability(
    dataset: str,
    model: str,
    threshold_type: str,
    threshold: float,
    calibration_type: str,
    y_true: pd.Series,
    probability: np.ndarray,
    notes: str,
) -> dict[str, Any]:
    """Compute a decision-revision row from saved probabilities."""

    metrics = binary_classification_metrics(y_true, probability, threshold=threshold)
    expected_cost = FN_COST * int(metrics["fn"]) + FP_COST * int(metrics["fp"])
    return {
        "dataset": dataset,
        "model": model,
        "threshold_type": threshold_type,
        "threshold": float(threshold),
        "calibration_type": calibration_type,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "f1": float(metrics["f1"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "brier": float(metrics["brier_score"]),
        "ece": float(metrics["ece"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
        "expected_cost_FN5_FP1": float(expected_cost),
        "notes": notes,
    }


def manual_review_threshold(dataset: str, display_model: str) -> tuple[float | None, str]:
    """Return the single-threshold anchor from the manual-review revision if available."""

    path = TABLES_DIR / f"manual_review_band_revision_{dataset}.csv"
    if not path.exists():
        return None, "manual-review revision table not found; operational threshold falls back to current cost-optimal threshold"
    table = pd.read_csv(path)
    rows = table.loc[table["split"].eq("test") & table["model"].eq(display_model)]
    if rows.empty:
        return None, "no manual-review band row for this model; operational threshold falls back to current cost-optimal threshold"
    row = rows.iloc[0]
    note = (
        "threshold-only anchor from manual-review revision; "
        f"manual band also uses low={row['low_risk_threshold']:.2f}, high={row['high_risk_threshold']:.2f}, "
        "selected on validation"
    )
    return float(row["threshold_only_threshold"]), note


def scre_bundle_threshold(dataset: str, model: str) -> tuple[float | None, str]:
    """Return a saved SCRE threshold if the model is a SCRE variant."""

    if model == "Old SCRE-Credit":
        bundle = joblib.load(MODELS_DIR / "scre_credit_model.joblib")
        return float(bundle[dataset]["threshold"]), "saved Old SCRE-Credit threshold from scre_credit_model.joblib"
    if model == "SCRE-Pareto":
        bundle = joblib.load(MODELS_DIR / f"scre_pareto_{dataset}.joblib")
        return float(bundle["threshold"]), "saved SCRE-Pareto threshold from scre_pareto joblib bundle"
    if model == "SCRE-Optimized":
        bundle = joblib.load(MODELS_DIR / f"scre_optimized_{dataset}.joblib")
        return float(bundle["selected_threshold"]), "saved SCRE-Optimized threshold from scre_optimized joblib bundle"
    return None, "not a SCRE-family model"


def build_dataset_rows(dataset: str, provider: ProbabilityProvider) -> pd.DataFrame:
    """Build threshold-baseline rows for one dataset."""

    split = provider.splits[dataset]
    base_results = read_csv(TABLES_DIR / f"scre_credit_results_{dataset}.csv")
    rows: list[dict[str, Any]] = []

    selected_sources: dict[str, str] = {}
    for display_model, candidates in MODEL_FAMILIES.items():
        selected_sources[display_model] = select_validation_best(base_results, candidates)
    selected_sources["Old SCRE-Credit"] = "SCRE-Credit"

    for display_model in DISPLAY_ORDER:
        if display_model in selected_sources:
            source_model = selected_sources[display_model]
            current = current_base_row(base_results, source_model)
            probability = provider.base_probability(dataset, source_model)
            calibration_type = str(current["calibration_method"])
            source_note = f"source_model={source_model}; selected on validation within family"
        else:
            source_model = display_model
            current = current_scre_variant_row(dataset, display_model)
            probability = provider.scre_variant_probability(dataset, display_model)
            calibration_type = str(current["calibration_type"])
            source_note = "SCRE weights/objective/threshold selected on validation only"

        rows.append(
            row_from_probability(
                dataset=dataset,
                model=display_model,
                threshold_type="fixed_0_50",
                threshold=0.50,
                calibration_type=calibration_type,
                y_true=split.y_test,
                probability=probability,
                notes=f"inference-only recomputation from saved model probabilities; {source_note}",
            )
        )
        rows.append(
            row_from_existing(
                dataset=dataset,
                model=display_model,
                threshold_type="cost_optimal_current",
                row=current,
                notes=f"frozen final test row read from existing result table; {source_note}",
            )
        )

        op_threshold, op_note = manual_review_threshold(dataset, display_model)
        if op_threshold is None:
            op_threshold = float(current["threshold"])
            op_note = f"{op_note}; same numeric threshold as cost_optimal_current"
        rows.append(
            row_from_probability(
                dataset=dataset,
                model=display_model,
                threshold_type="operational_current",
                threshold=op_threshold,
                calibration_type=calibration_type,
                y_true=split.y_test,
                probability=probability,
                notes=f"{op_note}; {source_note}",
            )
        )

        scre_threshold, scre_note = scre_bundle_threshold(dataset, display_model)
        if scre_threshold is not None:
            rows.append(
                row_from_probability(
                    dataset=dataset,
                    model=display_model,
                    threshold_type="scre_saved_threshold",
                    threshold=scre_threshold,
                    calibration_type=calibration_type,
                    y_true=split.y_test,
                    probability=probability,
                    notes=f"{scre_note}; {source_note}",
                )
            )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result["model"] = pd.Categorical(result["model"], categories=DISPLAY_ORDER, ordered=True)
    result["threshold_type"] = pd.Categorical(
        result["threshold_type"],
        categories=["fixed_0_50", "cost_optimal_current", "operational_current", "scre_saved_threshold"],
        ordered=True,
    )
    result = result.sort_values(["model", "threshold_type"]).reset_index(drop=True)
    result["model"] = result["model"].astype(str)
    result["threshold_type"] = result["threshold_type"].astype(str)
    return result


def majority_accuracy(split: DatasetSplit) -> tuple[int, float]:
    """Return majority class label and full-dataset majority accuracy."""

    counts = split.full_y.value_counts(normalize=True).sort_values(ascending=False)
    majority_label = int(counts.index[0])
    return majority_label, float(counts.iloc[0])


def fmt(value: float) -> str:
    """Format numbers for markdown."""

    return f"{value:.4f}"


def build_problem_summary(taiwan: pd.DataFrame, heloc: pd.DataFrame, splits: dict[str, DatasetSplit]) -> str:
    """Build the markdown problem summary requested for decision revision."""

    tw_cat_cost = taiwan.loc[
        taiwan["model"].eq("Best CatBoost") & taiwan["threshold_type"].eq("cost_optimal_current")
    ].iloc[0]
    tw_cat_050 = taiwan.loc[
        taiwan["model"].eq("Best CatBoost") & taiwan["threshold_type"].eq("fixed_0_50")
    ].iloc[0]
    tw_worst_fp = taiwan.loc[taiwan["threshold_type"].eq("cost_optimal_current")].sort_values("fp", ascending=False).iloc[0]
    he_scorecard_cost = heloc.loc[
        heloc["model"].eq("Best Scorecard") & heloc["threshold_type"].eq("cost_optimal_current")
    ].iloc[0]
    he_cat_cost = heloc.loc[
        heloc["model"].eq("Best CatBoost") & heloc["threshold_type"].eq("cost_optimal_current")
    ].iloc[0]
    tw_majority_label, tw_majority_acc = majority_accuracy(splits["taiwan"])
    he_majority_label, he_majority_acc = majority_accuracy(splits["heloc"])

    return f"""# Current Decision Problem Summary

## 1. Current Taiwan problem

The frozen Taiwan operational winner is Best CatBoost at threshold {tw_cat_cost['threshold']:.2f}. It has high recall ({fmt(tw_cat_cost['recall'])}) but low precision ({fmt(tw_cat_cost['precision'])}) and only moderate specificity ({fmt(tw_cat_cost['specificity'])}). The confusion matrix is TN={int(tw_cat_cost['tn'])}, FP={int(tw_cat_cost['fp'])}, FN={int(tw_cat_cost['fn'])}, TP={int(tw_cat_cost['tp'])}. This is a screening-style configuration: it catches many likely defaults but sends many non-default cases into the positive/high-risk bucket.

At threshold 0.50, the same CatBoost probability model has much higher precision ({fmt(tw_cat_050['precision'])}) and specificity ({fmt(tw_cat_050['specificity'])}), but recall falls to {fmt(tw_cat_050['recall'])}. That contrast is the core reason for the next false-positive reduction experiments.

The most FP-heavy frozen Taiwan cost-optimal row among the tracked final candidates is {tw_worst_fp['model']} with FP={int(tw_worst_fp['fp'])}, precision={fmt(tw_worst_fp['precision'])}, specificity={fmt(tw_worst_fp['specificity'])}, and recall={fmt(tw_worst_fp['recall'])}.

## 2. Current HELOC problem

The frozen HELOC expected-cost/interpretable winner is Best Scorecard at threshold {he_scorecard_cost['threshold']:.2f}. It has very high recall ({fmt(he_scorecard_cost['recall'])}) but low specificity ({fmt(he_scorecard_cost['specificity'])}) and FP={int(he_scorecard_cost['fp'])}. Best CatBoost is close in cost and calibration, with threshold {he_cat_cost['threshold']:.2f}, recall={fmt(he_cat_cost['recall'])}, precision={fmt(he_cat_cost['precision'])}, specificity={fmt(he_cat_cost['specificity'])}, and FP={int(he_cat_cost['fp'])}.

HELOC has a different class balance and domain structure than Taiwan, so the same threshold intuition should not be transferred blindly.

## 3. Why accuracy is misleading

Taiwan's majority-class accuracy is {fmt(tw_majority_acc)} by predicting class {tw_majority_label} for everyone, because most customers are non-default. HELOC's majority-class accuracy is {fmt(he_majority_acc)} by predicting class {he_majority_label}. A model can look accurate by avoiding positive/default predictions, while missing costly defaults. Conversely, a low-threshold screening policy can lower accuracy by increasing false positives while still reducing FN-weighted cost.

Accuracy therefore hides the precision-recall-specificity tradeoff that matters for credit-risk decisioning.

## 4. Why precision is low

The cost-sensitive thresholds are low because FN is weighted five times FP. Lowering the threshold moves more customers into the predicted-default/high-risk bucket. This raises recall, but many marginal non-default customers also cross the threshold, so precision drops.

For Taiwan CatBoost, threshold {tw_cat_cost['threshold']:.2f} is intentionally aggressive: recall={fmt(tw_cat_cost['recall'])}, precision={fmt(tw_cat_cost['precision'])}, FP={int(tw_cat_cost['fp'])}. This should be interpreted as a high-sensitivity screening policy, not as a final automatic rejection rule.

## 5. Why false positives matter

False positives are non-default customers flagged as high risk. In a real credit workflow they can create unnecessary manual review, customer friction, opportunity cost, and potential unfair treatment concerns. The original FN=5, FP=1 cost setup made sense for default capture, but the next revision should explicitly constrain FP volume, precision, specificity, and review capacity.

## 6. Why this model should not be presented as automatic rejection

The current low-threshold models are not precise enough for automatic rejection. A Taiwan CatBoost precision of {fmt(tw_cat_cost['precision'])} means many predicted high-risk cases are actually non-default. The safer framing is: use the model as a screening/risk-prioritization system with manual review, not as a standalone automatic deny system.

## 7. What the next experiments will try to fix

The next experiments should search for constrained operating points that reduce FP, improve precision and specificity, preserve a minimum recall floor, and keep expected cost within an acceptable tolerance. The natural next step is constrained threshold search and manual-review band optimization with explicit constraints on precision, specificity, recall, FP count, and review coverage.
"""


def build_inventory(
    taiwan: pd.DataFrame,
    heloc: pd.DataFrame,
    splits: dict[str, DatasetSplit],
    generated_paths: list[Path],
) -> pd.DataFrame:
    """Build a frozen inventory of source and generated files."""

    source_files = [
        TABLES_DIR / "scre_credit_results_taiwan.csv",
        TABLES_DIR / "scre_credit_results_heloc.csv",
        TABLES_DIR / "scre_pareto_results_taiwan.csv",
        TABLES_DIR / "scre_pareto_results_heloc.csv",
        TABLES_DIR / "scre_optimized_results_taiwan.csv",
        TABLES_DIR / "scre_optimized_results_heloc.csv",
        TABLES_DIR / "scre_revision_comparison_taiwan.csv",
        TABLES_DIR / "scre_revision_comparison_heloc.csv",
        TABLES_DIR / "manual_review_band_revision_taiwan.csv",
        TABLES_DIR / "manual_review_band_revision_heloc.csv",
        TABLES_DIR / "scre_model_weights.csv",
        MODELS_DIR / "scre_credit_model.joblib",
        MODELS_DIR / "scre_pareto_taiwan.joblib",
        MODELS_DIR / "scre_pareto_heloc.joblib",
        MODELS_DIR / "scre_optimized_taiwan.joblib",
        MODELS_DIR / "scre_optimized_heloc.joblib",
        TAIWAN_MODEL_READY,
        HELOC_MODEL_READY,
    ]

    rows: list[dict[str, Any]] = []
    for path in source_files:
        rows.append(
            {
                "artifact_type": "source",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "notes": "read-only frozen input for decision revision baseline",
            }
        )
    for path in generated_paths:
        rows.append(
            {
                "artifact_type": "generated",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "notes": "new output written under outputs/decision_revision",
            }
        )

    for dataset, table in [("taiwan", taiwan), ("heloc", heloc)]:
        majority_label, majority_acc = majority_accuracy(splits[dataset])
        rows.append(
            {
                "artifact_type": "majority_baseline",
                "path": dataset,
                "exists": True,
                "bytes": 0,
                "notes": json.dumps(
                    {
                        "majority_label": majority_label,
                        "majority_accuracy_full_dataset": majority_acc,
                        "rows_in_threshold_table": int(len(table)),
                    },
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame, splits: dict[str, DatasetSplit]) -> list[Path]:
    """Write all decision-revision outputs."""

    DECISION_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    taiwan_path = DECISION_REVISION_DIR / "current_threshold_baseline_taiwan.csv"
    heloc_path = DECISION_REVISION_DIR / "current_threshold_baseline_heloc.csv"
    summary_path = DECISION_REVISION_DIR / "current_problem_summary.md"
    inventory_path = DECISION_REVISION_DIR / "frozen_baseline_inventory.csv"

    taiwan.to_csv(taiwan_path, index=False)
    heloc.to_csv(heloc_path, index=False)
    summary_path.write_text(build_problem_summary(taiwan, heloc, splits), encoding="utf-8", newline="\n")
    inventory = build_inventory(taiwan, heloc, splits, [taiwan_path, heloc_path, summary_path, inventory_path])
    inventory["created_local"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inventory.to_csv(inventory_path, index=False)
    return [taiwan_path, heloc_path, summary_path, inventory_path]


def main() -> None:
    """Build the frozen threshold baseline package."""

    splits = {
        "taiwan": load_dataset_split("taiwan"),
        "heloc": load_dataset_split("heloc"),
    }
    provider = ProbabilityProvider(splits)
    taiwan = build_dataset_rows("taiwan", provider)
    heloc = build_dataset_rows("heloc", provider)
    paths = write_outputs(taiwan, heloc, splits)
    for path in paths:
        print(path)
    print("\nTaiwan tracked rows:")
    print(taiwan[["model", "threshold_type", "threshold", "precision", "recall", "specificity", "fp", "expected_cost_FN5_FP1"]].to_string(index=False))
    print("\nHELOC tracked rows:")
    print(heloc[["model", "threshold_type", "threshold", "precision", "recall", "specificity", "fp", "expected_cost_FN5_FP1"]].to_string(index=False))


if __name__ == "__main__":
    main()
