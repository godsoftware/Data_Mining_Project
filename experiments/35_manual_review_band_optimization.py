"""Optimize low/manual/high risk bands on validation-only frozen probabilities.

No model is trained and no feature is generated. The script searches low/high
probability thresholds on validation, applies the selected policies once on the
held-out test split, and writes manual-review decision-support artifacts.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_threshold_tools() -> Any:
    """Load frozen-probability helpers from the threshold experiment."""

    script_path = PROJECT_ROOT / "experiments" / "32_precision_constrained_threshold_search.py"
    spec = importlib.util.spec_from_file_location("precision_threshold_tools", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tools = _load_threshold_tools()

DECISION_REVISION_DIR: Path = tools.DECISION_REVISION_DIR
FN_COST = 5.0
FP_COST = 1.0
MODEL_CANDIDATES = [
    "Best CatBoost",
    "Best XGBoost",
    "Best Scorecard",
    "SCRE-Optimized",
    "SCRE-Pareto",
]
T_LOW_GRID = np.round(np.arange(0.03, 0.4001, 0.01), 2)
T_HIGH_GRID = np.round(np.arange(0.10, 0.8001, 0.01), 2)


@dataclass(frozen=True)
class ConstraintSpec:
    """Manual-review band constraint."""

    name: str
    details: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def constraint_specs() -> list[ConstraintSpec]:
    """Return all requested manual-review constraint scenarios."""

    return [
        ConstraintSpec("A_manual_review_rate_0_20", "manual_review_rate <= 0.20", lambda df: df["manual_review_rate"] <= 0.20),
        ConstraintSpec("B_manual_review_rate_0_25", "manual_review_rate <= 0.25", lambda df: df["manual_review_rate"] <= 0.25),
        ConstraintSpec("C_manual_review_rate_0_30", "manual_review_rate <= 0.30", lambda df: df["manual_review_rate"] <= 0.30),
        ConstraintSpec("D_high_risk_precision_0_40", "high_risk_precision >= 0.40", lambda df: df["high_risk_precision"] >= 0.40),
        ConstraintSpec("E_high_risk_precision_0_45", "high_risk_precision >= 0.45", lambda df: df["high_risk_precision"] >= 0.45),
        ConstraintSpec("F_high_risk_precision_0_50", "high_risk_precision >= 0.50", lambda df: df["high_risk_precision"] >= 0.50),
        ConstraintSpec("G_low_risk_default_rate_0_10", "low_risk_default_rate <= 0.10", lambda df: df["default_rate_low_risk"] <= 0.10),
        ConstraintSpec("H_low_risk_default_rate_0_08", "low_risk_default_rate <= 0.08", lambda df: df["default_rate_low_risk"] <= 0.08),
        ConstraintSpec("I_high_risk_recall_0_60", "high_risk_recall >= 0.60", lambda df: df["high_risk_recall"] >= 0.60),
        ConstraintSpec("J_high_risk_recall_0_65", "high_risk_recall >= 0.65", lambda df: df["high_risk_recall"] >= 0.65),
        ConstraintSpec(
            "K_review_0_25_precision_0_40",
            "manual_review_rate <= 0.25 AND high_risk_precision >= 0.40",
            lambda df: (df["manual_review_rate"] <= 0.25) & (df["high_risk_precision"] >= 0.40),
        ),
        ConstraintSpec(
            "L_review_0_25_low_default_0_10",
            "manual_review_rate <= 0.25 AND low_risk_default_rate <= 0.10",
            lambda df: (df["manual_review_rate"] <= 0.25) & (df["default_rate_low_risk"] <= 0.10),
        ),
        ConstraintSpec(
            "M_precision_0_45_low_default_0_10",
            "high_risk_precision >= 0.45 AND low_risk_default_rate <= 0.10",
            lambda df: (df["high_risk_precision"] >= 0.45) & (df["default_rate_low_risk"] <= 0.10),
        ),
        ConstraintSpec(
            "N_review_0_30_precision_0_45_low_default_0_10",
            "manual_review_rate <= 0.30 AND high_risk_precision >= 0.45 AND low_risk_default_rate <= 0.10",
            lambda df: (
                (df["manual_review_rate"] <= 0.30)
                & (df["high_risk_precision"] >= 0.45)
                & (df["default_rate_low_risk"] <= 0.10)
            ),
        ),
    ]


def evaluate_band(y_true: np.ndarray, probability: np.ndarray, t_low: float, t_high: float) -> dict[str, Any]:
    """Evaluate one three-way policy."""

    low = probability < t_low
    high = probability >= t_high
    manual = ~(low | high)
    default = y_true == 1
    non_default = ~default
    n_rows = len(y_true)
    total_defaults = int(default.sum())

    low_count = int(low.sum())
    manual_count = int(manual.sum())
    high_count = int(high.sum())
    low_defaults = int(np.sum(low & default))
    manual_defaults = int(np.sum(manual & default))
    high_defaults = int(np.sum(high & default))
    low_non_defaults = int(np.sum(low & non_default))
    manual_non_defaults = int(np.sum(manual & non_default))
    high_non_defaults = int(np.sum(high & non_default))
    remaining_cost = float(FN_COST * low_defaults + FP_COST * high_non_defaults)

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "low_risk_count": low_count,
        "manual_review_count": manual_count,
        "high_risk_count": high_count,
        "auto_decision_rate": float((low_count + high_count) / n_rows),
        "manual_review_rate": float(manual_count / n_rows),
        "default_rate_low_risk": float(low_defaults / low_count) if low_count else np.nan,
        "default_rate_manual_review": float(manual_defaults / manual_count) if manual_count else np.nan,
        "default_rate_high_risk": float(high_defaults / high_count) if high_count else np.nan,
        "high_risk_precision": float(high_defaults / high_count) if high_count else 0.0,
        "high_risk_recall": float(high_defaults / total_defaults) if total_defaults else np.nan,
        "low_risk_false_negative_count": low_defaults,
        "high_risk_false_positive_count": high_non_defaults,
        "low_risk_non_default_count": low_non_defaults,
        "manual_review_default_count": manual_defaults,
        "manual_review_non_default_count": manual_non_defaults,
        "high_risk_default_count": high_defaults,
        "total_defaults": total_defaults,
        "remaining_expected_cost": remaining_cost,
    }


def build_band_grid(y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Evaluate all valid t_low/t_high pairs for one validation split."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    rows: list[dict[str, Any]] = []
    for t_low in T_LOW_GRID:
        for t_high in T_HIGH_GRID:
            if float(t_low) >= float(t_high):
                continue
            rows.append(evaluate_band(y, probability, float(t_low), float(t_high)))
    return pd.DataFrame(rows)


def select_band(grid: pd.DataFrame, constraint: ConstraintSpec) -> pd.Series | None:
    """Select validation band under one constraint and objective."""

    feasible = grid.loc[constraint.predicate(grid)].copy()
    if feasible.empty:
        return None
    feasible = feasible.sort_values(
        [
            "remaining_expected_cost",
            "high_risk_precision",
            "manual_review_rate",
            "high_risk_recall",
            "t_high",
            "t_low",
        ],
        ascending=[True, False, True, False, False, True],
    )
    return feasible.iloc[0]


def load_binary_reference(dataset: str) -> pd.DataFrame:
    """Load frozen current binary cost-threshold baseline rows."""

    path = DECISION_REVISION_DIR / f"current_threshold_baseline_{dataset}.csv"
    baseline = pd.read_csv(path)
    baseline = baseline.loc[baseline["threshold_type"].eq("cost_optimal_current")].copy()
    return baseline.set_index("model", drop=False)


def notes_for_row(row: dict[str, Any]) -> str:
    """Create a compact operational note."""

    if not bool(row["feasible_on_validation"]):
        return "No feasible validation band for this constraint."
    precision_gain = float(row["test_high_risk_precision"]) - float(row["binary_policy_reference_precision"])
    fp_reduction = float(row["fp_reduction_vs_binary_threshold"])
    cost_reduction = float(row["cost_reduction_vs_binary_threshold"])
    review_rate = float(row["test_manual_review_rate"])
    if precision_gain > 0 and fp_reduction > 0 and review_rate <= 0.30:
        return "More defensible screening policy: precision and FP improve with bounded review load."
    if precision_gain > 0 and fp_reduction > 0:
        return "Precision and FP improve, but manual-review workload needs operational review."
    if cost_reduction > 0:
        return "Remaining auto-decision cost improves versus binary threshold."
    return "Feasible band, but tradeoff is weaker than the binary reference."


def result_row(
    dataset: str,
    model: str,
    source_model: str,
    calibration_type: str,
    constraint: ConstraintSpec,
    selected: pd.Series | None,
    validation_grid: pd.DataFrame,
    y_test: pd.Series,
    test_probability: np.ndarray,
    binary_reference: pd.Series,
) -> dict[str, Any]:
    """Build one selected-band test result row."""

    base: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "source_model": source_model,
        "constraint_setting": constraint.name,
        "constraint_details": constraint.details,
        "selection_split": "validation",
        "test_set_used_for_band_selection": False,
        "feasible_on_validation": selected is not None,
        "calibration_type": calibration_type,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "binary_policy_reference_threshold": float(binary_reference["threshold"]),
        "binary_policy_reference_precision": float(binary_reference["precision"]),
        "binary_policy_reference_recall": float(binary_reference["recall"]),
        "binary_policy_reference_fp": int(binary_reference["fp"]),
        "binary_policy_reference_cost": float(binary_reference["expected_cost_FN5_FP1"]),
    }
    if selected is None:
        nullable_columns = [
            "t_low_validation",
            "t_high_validation",
            "validation_remaining_expected_cost",
            "validation_manual_review_rate",
            "validation_high_risk_precision",
            "validation_high_risk_recall",
            "validation_low_risk_default_rate",
            "test_low_risk_count",
            "test_manual_review_count",
            "test_high_risk_count",
            "test_auto_decision_rate",
            "test_manual_review_rate",
            "test_default_rate_low_risk",
            "test_default_rate_manual_review",
            "test_default_rate_high_risk",
            "test_high_risk_precision",
            "test_high_risk_recall",
            "test_low_risk_false_negative_count",
            "test_high_risk_false_positive_count",
            "test_remaining_expected_cost",
            "cost_reduction_vs_binary_threshold",
            "fp_reduction_vs_binary_threshold",
        ]
        for column in nullable_columns:
            base[column] = np.nan
        base["notes"] = notes_for_row(base)
        return base

    selected_match = validation_grid.loc[
        np.isclose(validation_grid["t_low"], float(selected["t_low"]))
        & np.isclose(validation_grid["t_high"], float(selected["t_high"]))
    ].iloc[0]
    test_metrics = evaluate_band(
        np.asarray(y_test, dtype=int),
        np.asarray(test_probability, dtype=float),
        float(selected["t_low"]),
        float(selected["t_high"]),
    )
    base.update(
        {
            "t_low_validation": float(selected["t_low"]),
            "t_high_validation": float(selected["t_high"]),
            "validation_remaining_expected_cost": float(selected_match["remaining_expected_cost"]),
            "validation_manual_review_rate": float(selected_match["manual_review_rate"]),
            "validation_high_risk_precision": float(selected_match["high_risk_precision"]),
            "validation_high_risk_recall": float(selected_match["high_risk_recall"]),
            "validation_low_risk_default_rate": float(selected_match["default_rate_low_risk"]),
            "test_low_risk_count": int(test_metrics["low_risk_count"]),
            "test_manual_review_count": int(test_metrics["manual_review_count"]),
            "test_high_risk_count": int(test_metrics["high_risk_count"]),
            "test_auto_decision_rate": float(test_metrics["auto_decision_rate"]),
            "test_manual_review_rate": float(test_metrics["manual_review_rate"]),
            "test_default_rate_low_risk": float(test_metrics["default_rate_low_risk"]),
            "test_default_rate_manual_review": float(test_metrics["default_rate_manual_review"]),
            "test_default_rate_high_risk": float(test_metrics["default_rate_high_risk"]),
            "test_high_risk_precision": float(test_metrics["high_risk_precision"]),
            "test_high_risk_recall": float(test_metrics["high_risk_recall"]),
            "test_low_risk_false_negative_count": int(test_metrics["low_risk_false_negative_count"]),
            "test_high_risk_false_positive_count": int(test_metrics["high_risk_false_positive_count"]),
            "test_remaining_expected_cost": float(test_metrics["remaining_expected_cost"]),
        }
    )
    base["cost_reduction_vs_binary_threshold"] = (
        float(base["binary_policy_reference_cost"]) - float(base["test_remaining_expected_cost"])
    )
    base["fp_reduction_vs_binary_threshold"] = (
        int(base["binary_policy_reference_fp"]) - int(base["test_high_risk_false_positive_count"])
    )
    base["notes"] = notes_for_row(base)
    return base


def build_dataset_results(dataset: str, provider: Any) -> pd.DataFrame:
    """Build selected manual-review policies for one dataset."""

    plan = tools.model_plan(dataset)
    split = provider.splits[dataset]
    binary_reference = load_binary_reference(dataset)
    rows: list[dict[str, Any]] = []

    for model in MODEL_CANDIDATES:
        source_model, calibration_type = plan[model]
        validation_probability = tools.probability_for_display(
            provider,
            dataset,
            model,
            source_model,
            "validation",
        )
        test_probability = tools.probability_for_display(
            provider,
            dataset,
            model,
            source_model,
            "test",
        )
        validation_grid = build_band_grid(split.y_validation, validation_probability)
        for constraint in constraint_specs():
            selected = select_band(validation_grid, constraint)
            rows.append(
                result_row(
                    dataset,
                    model,
                    source_model,
                    calibration_type,
                    constraint,
                    selected,
                    validation_grid,
                    split.y_test,
                    test_probability,
                    binary_reference.loc[model],
                )
            )

    return pd.DataFrame(rows)


def best_policy_per_model(table: pd.DataFrame) -> pd.DataFrame:
    """Pick one practical validation-selected policy per model."""

    feasible = table.loc[table["feasible_on_validation"].astype(bool)].copy()
    if feasible.empty:
        return feasible
    feasible["bounded_workload"] = feasible["validation_manual_review_rate"] <= 0.30
    feasible["precision_gain_validation"] = (
        feasible["validation_high_risk_precision"] - feasible["binary_policy_reference_precision"]
    )
    feasible = feasible.sort_values(
        [
            "model",
            "bounded_workload",
            "validation_remaining_expected_cost",
            "validation_high_risk_precision",
            "validation_manual_review_rate",
        ],
        ascending=[True, False, True, False, True],
    )
    return feasible.groupby("model", as_index=False).head(1).reset_index(drop=True)


def validation_winner(table: pd.DataFrame) -> pd.Series:
    """Select the overall best practical policy by validation criteria."""

    best = best_policy_per_model(table)
    best = best.sort_values(
        [
            "validation_remaining_expected_cost",
            "validation_high_risk_precision",
            "validation_manual_review_rate",
        ],
        ascending=[True, False, True],
    )
    return best.iloc[0]


def fmt(value: float) -> str:
    """Format numeric values for markdown."""

    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def write_summary(dataset: str, table: pd.DataFrame) -> Path:
    """Write dataset-specific manual-review summary."""

    best = best_policy_per_model(table)
    winner = validation_winner(table)
    lines = [
        f"# Manual Review Band Optimization Summary: {dataset.upper()}",
        "",
        "Low/high thresholds were selected on validation only. Test metrics report the held-out performance of those selected policies.",
        "",
        "Remaining expected cost uses FN=5 for low-risk defaults plus FP=1 for high-risk non-defaults. Manual-review cases are assumed deferred to human review rather than automatic accept/reject.",
        "",
        "## Best Policy Per Model",
        "",
        "| Model | Constraint | t_low | t_high | Manual Review Rate | High-Risk Precision | High-Risk Recall | Low-Risk Default Rate | Cost | Comment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.constraint_setting} | {fmt(row.t_low_validation)} | {fmt(row.t_high_validation)} | "
            f"{fmt(row.test_manual_review_rate)} | {fmt(row.test_high_risk_precision)} | "
            f"{fmt(row.test_high_risk_recall)} | {fmt(row.test_default_rate_low_risk)} | "
            f"{fmt(row.test_remaining_expected_cost)} | {row.notes} |"
        )

    lines.extend(
        [
            "",
            "## Overall Validation-Selected Recommendation",
            "",
            f"- Model: {winner['model']}",
            f"- Constraint: {winner['constraint_setting']} ({winner['constraint_details']})",
            f"- Validation thresholds: t_low={winner['t_low_validation']:.2f}, t_high={winner['t_high_validation']:.2f}",
            f"- Test manual review rate: {winner['test_manual_review_rate']:.4f}",
            f"- Test high-risk precision: {winner['test_high_risk_precision']:.4f}",
            f"- Test high-risk recall: {winner['test_high_risk_recall']:.4f}",
            f"- Test remaining expected cost: {winner['test_remaining_expected_cost']:.0f}",
            "",
        ]
    )

    if dataset == "taiwan":
        old_precision = 0.351
        catboost = best.loc[best["model"].eq("Best CatBoost")].iloc[0]
        lines.extend(
            [
                "## Taiwan Operational Questions",
                "",
                f"1. High-risk precision above old binary precision 0.351: {'YES' if catboost['test_high_risk_precision'] > old_precision else 'NO'} ({catboost['test_high_risk_precision']:.4f}).",
                f"2. Manual review rate realistic: {'YES' if catboost['test_manual_review_rate'] <= 0.30 else 'NO'} ({catboost['test_manual_review_rate']:.4f}).",
                f"3. Low-risk default rate acceptable under 0.10 target: {'YES' if catboost['test_default_rate_low_risk'] <= 0.10 else 'NO'} ({catboost['test_default_rate_low_risk']:.4f}).",
                f"4. FP reduced vs binary threshold: {'YES' if catboost['fp_reduction_vs_binary_threshold'] > 0 else 'NO'} ({catboost['fp_reduction_vs_binary_threshold']:.0f}).",
                f"5. Cost deterioration vs binary threshold: {'NO' if catboost['cost_reduction_vs_binary_threshold'] >= 0 else 'YES'} (cost reduction {catboost['cost_reduction_vs_binary_threshold']:.0f}).",
                "6. Defensible framing: YES, as screening + manual review; not as automatic rejection.",
                "",
            ]
        )

    path = DECISION_REVISION_DIR / f"manual_review_band_summary_{dataset}.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def plot_distribution(dataset: str, table: pd.DataFrame) -> Path:
    """Plot low/manual/high counts for best policy per model."""

    best = best_policy_per_model(table).sort_values("test_remaining_expected_cost")
    path = DECISION_REVISION_DIR / f"manual_review_band_distribution_{dataset}.png"
    labels = best["model"].tolist()
    x = np.arange(len(labels))
    low = best["test_low_risk_count"].to_numpy(dtype=float)
    manual = best["test_manual_review_count"].to_numpy(dtype=float)
    high = best["test_high_risk_count"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x, low, label="Low risk")
    ax.bar(x, manual, bottom=low, label="Manual review")
    ax.bar(x, high, bottom=low + manual, label="High risk")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test count")
    ax.set_title(f"Manual Review Band Distribution: {dataset.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_cost_tradeoff(dataset: str, table: pd.DataFrame) -> Path:
    """Plot manual-review workload vs remaining cost."""

    feasible = table.loc[table["feasible_on_validation"].astype(bool)].copy()
    path = DECISION_REVISION_DIR / f"manual_review_band_cost_tradeoff_{dataset}.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODEL_CANDIDATES:
        rows = feasible.loc[feasible["model"].eq(model)]
        ax.scatter(
            rows["test_manual_review_rate"],
            rows["test_remaining_expected_cost"],
            label=model,
            alpha=0.75,
            s=35,
        )
    ax.set_xlabel("Manual review rate")
    ax.set_ylabel("Remaining expected cost")
    ax.set_title(f"Manual Review Cost Tradeoff: {dataset.upper()}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> list[Path]:
    """Write CSV, markdown, and figure outputs."""

    DECISION_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        DECISION_REVISION_DIR / "manual_review_band_optimized_taiwan.csv",
        DECISION_REVISION_DIR / "manual_review_band_optimized_heloc.csv",
    ]
    taiwan.to_csv(paths[0], index=False)
    heloc.to_csv(paths[1], index=False)
    paths.extend(
        [
            write_summary("taiwan", taiwan),
            write_summary("heloc", heloc),
            plot_distribution("taiwan", taiwan),
            plot_distribution("heloc", heloc),
            plot_cost_tradeoff("taiwan", taiwan),
            plot_cost_tradeoff("heloc", heloc),
        ]
    )
    return paths


def main() -> None:
    """Run manual-review band optimization."""

    splits = {
        "taiwan": tools.load_split("taiwan"),
        "heloc": tools.load_split("heloc"),
    }
    provider = tools.ProbabilityProvider(splits)
    taiwan = build_dataset_results("taiwan", provider)
    heloc = build_dataset_results("heloc", provider)
    paths = write_outputs(taiwan, heloc)
    for path in paths:
        print(path)

    for dataset, table in [("Taiwan", taiwan), ("HELOC", heloc)]:
        print(f"\n{dataset} best manual-review policies:")
        best = best_policy_per_model(table)
        print(
            best[
                [
                    "model",
                    "t_low_validation",
                    "t_high_validation",
                    "test_manual_review_rate",
                    "test_high_risk_precision",
                    "test_high_risk_recall",
                    "test_default_rate_low_risk",
                    "test_remaining_expected_cost",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
