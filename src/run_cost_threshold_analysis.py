"""Phase 10 cost-sensitive threshold and manual-review-band analysis."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config.settings import MANUAL_REVIEW_COST
from data.split_leakage import stratified_train_validation_test_split
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from run_calibration_analysis import (
    DATASET_CONFIGS as CALIBRATION_DATASET_CONFIGS,
    _fit_probability_model,
    _positive_class_probability,
    build_calibration_model_specs,
)
from src.config.paths import FIGURES_DIR, TABLES_DIR


warnings.filterwarnings("ignore", category=UserWarning)


SCENARIOS = {
    "A_FN2_FP1": {"fn_cost": 2.0, "fp_cost": 1.0},
    "B_FN5_FP1": {"fn_cost": 5.0, "fp_cost": 1.0},
    "C_FN10_FP1": {"fn_cost": 10.0, "fp_cost": 1.0},
}


THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


@dataclass
class ProbabilityRecord:
    """Fitted validation probabilities for one model/calibration variant."""

    model: str
    source_phase: str
    model_family: str
    calibration_method: str
    y_proba: np.ndarray
    roc_auc: float
    pr_auc: float
    brier_score: float


@dataclass
class DatasetDecisionResults:
    """Phase 10 outputs for one dataset."""

    threshold_table: pd.DataFrame
    manual_review_table: pd.DataFrame
    primary_record: ProbabilityRecord


def _confusion_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute threshold-dependent binary decision metrics without refitting anything."""

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    g_mean = float(np.sqrt(recall * specificity))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "g_mean": g_mean,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _threshold_rows_for_record(
    dataset: str,
    record: ProbabilityRecord,
    y_true: np.ndarray,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> list[dict]:
    """Build threshold-analysis rows for one fitted probability record."""

    base_rows = [_confusion_at_threshold(y_true, record.y_proba, threshold) for threshold in THRESHOLD_GRID]
    rows: list[dict] = []
    for scenario, costs in SCENARIOS.items():
        scenario_rows = []
        for metrics in base_rows:
            expected_cost = costs["fn_cost"] * metrics["fn"] + costs["fp_cost"] * metrics["fp"]
            row = {
                "dataset": dataset,
                "selection_split": "validation",
                "test_set_used": False,
                "train_rows": int(train_rows),
                "validation_rows": int(validation_rows),
                "test_rows_reserved": int(test_rows),
                "model": record.model,
                "source_phase": record.source_phase,
                "model_family": record.model_family,
                "calibration_method": record.calibration_method,
                "scenario": scenario,
                "fn_cost": costs["fn_cost"],
                "fp_cost": costs["fp_cost"],
                "roc_auc": record.roc_auc,
                "pr_auc": record.pr_auc,
                "brier_score": record.brier_score,
                "fn_cost_component": costs["fn_cost"] * metrics["fn"],
                "fp_cost_component": costs["fp_cost"] * metrics["fp"],
                "expected_cost": expected_cost,
                "expected_cost_per_1000": expected_cost / validation_rows * 1000.0,
                **metrics,
            }
            scenario_rows.append(row)

        cost_050 = next(row["expected_cost"] for row in scenario_rows if row["threshold"] == 0.5)
        for row in scenario_rows:
            row["cost_at_threshold_0_50"] = cost_050
            row["cost_improvement_vs_0_50"] = cost_050 - row["expected_cost"]
            row["cost_improvement_pct_vs_0_50"] = (
                (cost_050 - row["expected_cost"]) / cost_050 if cost_050 else 0.0
            )
        rows.extend(scenario_rows)
    return rows


def collect_probability_records(dataset: str) -> tuple[list[ProbabilityRecord], pd.Series, int, int, int]:
    """Fit model/calibration variants and return validation probabilities."""

    config = CALIBRATION_DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    target = config["target"]
    split = stratified_train_validation_test_split(df, target_column=target)
    X_train = split.train.drop(columns=[target])
    y_train = split.train[target]
    X_validation = split.validation.drop(columns=[target])
    y_validation = split.validation[target]

    specs = build_calibration_model_specs(
        dataset,
        X_train,
        categorical_columns=config["categorical_columns"],
    )

    records: list[ProbabilityRecord] = []
    for spec in specs:
        for calibration_method in ["uncalibrated", "sigmoid", "isotonic"]:
            print(f"Fitting {dataset} decision probability: {spec.name} [{calibration_method}]")
            fitted = _fit_probability_model(spec.estimator, calibration_method, X_train, y_train)
            y_proba = _positive_class_probability(fitted, X_validation)
            metrics = binary_classification_metrics(y_validation, y_proba, threshold=0.5)
            records.append(
                ProbabilityRecord(
                    model=spec.name,
                    source_phase=spec.source_phase,
                    model_family=spec.model_family,
                    calibration_method=calibration_method,
                    y_proba=y_proba,
                    roc_auc=float(metrics["roc_auc"]),
                    pr_auc=float(metrics["pr_auc"]),
                    brier_score=float(metrics["brier_score"]),
                )
            )

    return records, y_validation, len(split.train), len(split.validation), len(split.test)


def build_threshold_analysis(
    dataset: str,
    records: list[ProbabilityRecord],
    y_validation: pd.Series,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> pd.DataFrame:
    """Evaluate all model/calibration variants over the full threshold grid."""

    y_true = np.asarray(y_validation).astype(int)
    rows: list[dict] = []
    for record in records:
        rows.extend(
            _threshold_rows_for_record(dataset, record, y_true, train_rows, validation_rows, test_rows)
        )

    table = pd.DataFrame(rows)
    table["is_best_threshold_for_model_scenario"] = False
    best_idx = table.groupby(["dataset", "model", "calibration_method", "scenario"])["expected_cost"].idxmin()
    table.loc[best_idx, "is_best_threshold_for_model_scenario"] = True
    return table.sort_values(
        ["dataset", "scenario", "expected_cost", "model", "calibration_method", "threshold"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)


def select_primary_decision_record(records: list[ProbabilityRecord]) -> ProbabilityRecord:
    """Select the primary model for manual-review search and plots."""

    return sorted(records, key=lambda item: (-item.pr_auc, item.brier_score, item.model, item.calibration_method))[0]


def manual_review_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    t_low: float,
    t_high: float,
    fn_cost: float,
    fp_cost: float,
    review_cost: float,
) -> dict[str, float | int]:
    """Evaluate low/manual/high risk bands on validation probabilities."""

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
    manual_review_count = int(review.sum())
    expected_policy_cost = fn_cost * low_fn + fp_cost * high_fp + review_cost * manual_review_count
    auto_decisions = low | high

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "expected_policy_cost": float(expected_policy_cost),
        "expected_policy_cost_per_1000": float(expected_policy_cost / len(y_true) * 1000.0),
        "low_count": int(low.sum()),
        "manual_review_count": manual_review_count,
        "high_count": int(high.sum()),
        "low_rate": float(low.mean()),
        "manual_review_rate": float(review.mean()),
        "high_rate": float(high.mean()),
        "auto_decision_rate": float(auto_decisions.mean()),
        "low_false_negative": low_fn,
        "high_false_positive": high_fp,
        "low_true_negative": low_tn,
        "high_true_positive": high_tp,
        "manual_review_default": review_default,
        "manual_review_non_default": review_non_default,
        "manual_review_default_rate": review_default / manual_review_count if manual_review_count else np.nan,
        "high_risk_precision": high_tp / max(high_tp + high_fp, 1),
        "high_risk_recall": high_tp / positives,
        "review_capture_recall": (high_tp + review_default) / positives,
    }


def build_manual_review_band_results(
    dataset: str,
    record: ProbabilityRecord,
    y_validation: pd.Series,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    max_review_rate: float = 0.45,
    min_review_capture_recall: float = 0.75,
) -> pd.DataFrame:
    """Search t_low/t_high bands for the selected decision record."""

    y_true = np.asarray(y_validation).astype(int)
    rows: list[dict] = []
    for scenario, costs in SCENARIOS.items():
        for t_low in THRESHOLD_GRID:
            for t_high in THRESHOLD_GRID:
                if t_low >= t_high:
                    continue
                metrics = manual_review_metrics(
                    y_true,
                    record.y_proba,
                    t_low=t_low,
                    t_high=t_high,
                    fn_cost=costs["fn_cost"],
                    fp_cost=costs["fp_cost"],
                    review_cost=MANUAL_REVIEW_COST,
                )
                meets_review_rate = metrics["manual_review_rate"] <= max_review_rate
                meets_review_capture = metrics["review_capture_recall"] >= min_review_capture_recall
                rows.append(
                    {
                        "dataset": dataset,
                        "selection_split": "validation",
                        "test_set_used": False,
                        "train_rows": int(train_rows),
                        "validation_rows": int(validation_rows),
                        "test_rows_reserved": int(test_rows),
                        "model": record.model,
                        "source_phase": record.source_phase,
                        "model_family": record.model_family,
                        "calibration_method": record.calibration_method,
                        "scenario": scenario,
                        "fn_cost": costs["fn_cost"],
                        "fp_cost": costs["fp_cost"],
                        "manual_review_cost": MANUAL_REVIEW_COST,
                        "max_review_rate_constraint": max_review_rate,
                        "min_review_capture_recall_constraint": min_review_capture_recall,
                        "meets_review_rate_constraint": bool(meets_review_rate),
                        "meets_review_capture_recall_constraint": bool(meets_review_capture),
                        "meets_constraints": bool(meets_review_rate and meets_review_capture),
                        "candidate_selection_policy": "highest_validation_pr_auc_with_brier_tiebreak",
                        **metrics,
                    }
                )

    table = pd.DataFrame(rows)
    table["is_best_band_for_scenario"] = False
    feasible = table.loc[table["meets_constraints"]].copy()
    if feasible.empty:
        feasible = table.copy()
    best_idx = feasible.groupby(["dataset", "scenario"])["expected_policy_cost"].idxmin()
    table.loc[best_idx, "is_best_band_for_scenario"] = True
    return table.sort_values(
        ["dataset", "scenario", "expected_policy_cost", "manual_review_rate", "t_low", "t_high"]
    ).reset_index(drop=True)


def plot_cost_curve(
    dataset: str,
    threshold_table: pd.DataFrame,
    primary_record: ProbabilityRecord,
    output_path: Path,
) -> None:
    """Plot expected cost over thresholds for the selected primary model."""

    primary = threshold_table.loc[
        (threshold_table["dataset"] == dataset)
        & (threshold_table["model"] == primary_record.model)
        & (threshold_table["calibration_method"] == primary_record.calibration_method)
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario, scenario_df in primary.groupby("scenario"):
        scenario_df = scenario_df.sort_values("threshold")
        best = scenario_df.loc[scenario_df["expected_cost"].idxmin()]
        ax.plot(
            scenario_df["threshold"],
            scenario_df["expected_cost"],
            linewidth=2,
            label=f"{scenario} best t={best['threshold']:.2f}",
        )
        ax.scatter([best["threshold"]], [best["expected_cost"]], s=40)

    ax.set_title(f"{dataset.upper()} Cost Curve - {primary_record.model} ({primary_record.calibration_method})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Cost on Validation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(
    dataset: str,
    threshold_table: pd.DataFrame,
    primary_record: ProbabilityRecord,
    output_path: Path,
) -> None:
    """Plot precision/recall/F1 trade-off for the selected primary model."""

    primary = threshold_table.loc[
        (threshold_table["dataset"] == dataset)
        & (threshold_table["model"] == primary_record.model)
        & (threshold_table["calibration_method"] == primary_record.calibration_method)
        & (threshold_table["scenario"] == "B_FN5_FP1")
    ].sort_values("threshold")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(primary["threshold"], primary["precision"], label="Precision", linewidth=2)
    ax.plot(primary["threshold"], primary["recall"], label="Recall", linewidth=2)
    ax.plot(primary["threshold"], primary["f1"], label="F1", linewidth=2)

    primary_all_scenarios = threshold_table.loc[
        (threshold_table["dataset"] == dataset)
        & (threshold_table["model"] == primary_record.model)
        & (threshold_table["calibration_method"] == primary_record.calibration_method)
    ]
    best_thresholds = [
        frame.loc[frame["expected_cost"].idxmin()]
        for _, frame in primary_all_scenarios.groupby("scenario")
    ]
    for row in best_thresholds:
        ax.axvline(row["threshold"], linestyle="--", alpha=0.45)
        ax.text(row["threshold"], 0.02, row["scenario"], rotation=90, va="bottom", ha="right", fontsize=8)

    ax.set_title(f"{dataset.upper()} Threshold Trade-off - {primary_record.model} ({primary_record.calibration_method})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_cost_curve(threshold_table: pd.DataFrame, output_path: Path) -> None:
    """Plot best validation cost curves for all datasets and scenarios."""

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (dataset, scenario), scenario_df in threshold_table.groupby(["dataset", "scenario"]):
        best = scenario_df.loc[scenario_df["expected_cost"].idxmin()]
        curve = scenario_df.loc[
            (scenario_df["model"] == best["model"])
            & (scenario_df["calibration_method"] == best["calibration_method"])
        ].sort_values("threshold")
        ax.plot(
            curve["threshold"],
            curve["expected_cost"],
            linewidth=1.8,
            label=f"{dataset} {scenario} best t={best['threshold']:.2f}",
        )
        ax.scatter([best["threshold"]], [best["expected_cost"]], s=25)

    ax.set_title("Cost Curve Summary")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Cost on Validation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_threshold_tradeoff(threshold_table: pd.DataFrame, output_path: Path) -> None:
    """Plot scenario-B threshold trade-offs for each dataset's best decision model."""

    fig, ax = plt.subplots(figsize=(9, 5.5))
    linestyles = {"precision": "-", "recall": "--", "f1": ":"}
    for dataset, dataset_df in threshold_table.groupby("dataset"):
        scenario_df = dataset_df.loc[dataset_df["scenario"] == "B_FN5_FP1"]
        best = scenario_df.loc[scenario_df["expected_cost"].idxmin()]
        curve = scenario_df.loc[
            (scenario_df["model"] == best["model"])
            & (scenario_df["calibration_method"] == best["calibration_method"])
        ].sort_values("threshold")
        for metric, linestyle in linestyles.items():
            ax.plot(
                curve["threshold"],
                curve[metric],
                linestyle=linestyle,
                linewidth=1.8,
                label=f"{dataset} {metric}",
            )

    ax.set_title("Threshold Trade-off Summary")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_cost_threshold_analysis_for_dataset(dataset: str) -> DatasetDecisionResults:
    """Run Phase 10 for one dataset and save dataset-specific tables/figures."""

    records, y_validation, train_rows, validation_rows, test_rows = collect_probability_records(dataset)
    threshold_table = build_threshold_analysis(dataset, records, y_validation, train_rows, validation_rows, test_rows)
    primary_record = select_primary_decision_record(records)
    manual_review_table = build_manual_review_band_results(
        dataset,
        primary_record,
        y_validation,
        train_rows,
        validation_rows,
        test_rows,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(TABLES_DIR / f"threshold_analysis_{dataset}.csv", index=False)
    manual_review_table.to_csv(TABLES_DIR / f"manual_review_band_results_{dataset}.csv", index=False)
    plot_cost_curve(dataset, threshold_table, primary_record, FIGURES_DIR / f"cost_curve_{dataset}.png")
    plot_threshold_tradeoff(dataset, threshold_table, primary_record, FIGURES_DIR / f"threshold_tradeoff_{dataset}.png")
    return DatasetDecisionResults(
        threshold_table=threshold_table,
        manual_review_table=manual_review_table,
        primary_record=primary_record,
    )


def write_contract_outputs(results: dict[str, DatasetDecisionResults]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write prompt-contract aliases, combining datasets when both are present."""

    threshold_table = pd.concat([result.threshold_table for result in results.values()], ignore_index=True)
    manual_review_table = pd.concat([result.manual_review_table for result in results.values()], ignore_index=True)
    threshold_table.to_csv(TABLES_DIR / "threshold_analysis.csv", index=False)
    manual_review_table.to_csv(TABLES_DIR / "manual_review_band_results.csv", index=False)

    if len(results) == 1:
        dataset, result = next(iter(results.items()))
        plot_cost_curve(dataset, threshold_table, result.primary_record, FIGURES_DIR / "cost_curve.png")
        plot_threshold_tradeoff(dataset, threshold_table, result.primary_record, FIGURES_DIR / "threshold_tradeoff.png")
    else:
        plot_combined_cost_curve(threshold_table, FIGURES_DIR / "cost_curve.png")
        plot_combined_threshold_tradeoff(threshold_table, FIGURES_DIR / "threshold_tradeoff.png")

    return threshold_table, manual_review_table


def _print_best_summaries(threshold_table: pd.DataFrame, manual_review_table: pd.DataFrame) -> None:
    """Print compact best-threshold and best-band summaries."""

    best_threshold_idx = threshold_table.groupby(["dataset", "scenario"])["expected_cost"].idxmin()
    best_thresholds = threshold_table.loc[best_threshold_idx]
    print(
        best_thresholds[
            [
                "dataset",
                "scenario",
                "model",
                "calibration_method",
                "threshold",
                "expected_cost",
                "cost_improvement_pct_vs_0_50",
                "precision",
                "recall",
                "f1",
            ]
        ]
        .sort_values(["dataset", "scenario"])
        .to_string(index=False)
    )

    best_band_idx = manual_review_table.groupby(["dataset", "scenario"])["expected_policy_cost"].idxmin()
    best_bands = manual_review_table.loc[best_band_idx]
    print(
        best_bands[
            [
                "dataset",
                "scenario",
                "model",
                "calibration_method",
                "t_low",
                "t_high",
                "expected_policy_cost",
                "manual_review_rate",
                "review_capture_recall",
                "high_risk_precision",
            ]
        ]
        .sort_values(["dataset", "scenario"])
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    args = parser.parse_args()

    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    run_id = start_run("cost_threshold_manual_review", dataset=args.dataset, tags=["phase-10", "cost-threshold"])
    try:
        results = {
            dataset: run_cost_threshold_analysis_for_dataset(dataset)
            for dataset in datasets
        }
        threshold_table, manual_review_table = write_contract_outputs(results)
        _print_best_summaries(threshold_table, manual_review_table)

        best_threshold_costs = {
            f"{dataset}_{scenario}": float(value)
            for (dataset, scenario), value in threshold_table.groupby(["dataset", "scenario"])["expected_cost"].min().items()
        }
        best_manual_review_costs = {
            f"{dataset}_{scenario}": float(value)
            for (dataset, scenario), value in manual_review_table.groupby(["dataset", "scenario"])[
                "expected_policy_cost"
            ].min().items()
        }
        artifacts = {
            "threshold_analysis_contract_alias": TABLES_DIR / "threshold_analysis.csv",
            "manual_review_band_results_contract_alias": TABLES_DIR / "manual_review_band_results.csv",
            "cost_curve_contract_alias": FIGURES_DIR / "cost_curve.png",
            "threshold_tradeoff_contract_alias": FIGURES_DIR / "threshold_tradeoff.png",
        }
        for dataset in datasets:
            artifacts[f"{dataset}_threshold_analysis"] = TABLES_DIR / f"threshold_analysis_{dataset}.csv"
            artifacts[f"{dataset}_manual_review_band_results"] = TABLES_DIR / f"manual_review_band_results_{dataset}.csv"
            artifacts[f"{dataset}_cost_curve"] = FIGURES_DIR / f"cost_curve_{dataset}.png"
            artifacts[f"{dataset}_threshold_tradeoff"] = FIGURES_DIR / f"threshold_tradeoff_{dataset}.png"

        finish_run(
            run_id,
            metrics={
                "threshold_rows": int(len(threshold_table)),
                "manual_review_rows": int(len(manual_review_table)),
                "best_threshold_costs": best_threshold_costs,
                "best_manual_review_costs": best_manual_review_costs,
            },
            artifacts=artifacts,
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
