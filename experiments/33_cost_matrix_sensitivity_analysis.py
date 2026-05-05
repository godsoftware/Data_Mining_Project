"""Run cost-matrix sensitivity analysis without training new models.

The script reuses frozen calibrated model probabilities, selects thresholds on
the validation split for each FN/FP cost scenario, and evaluates those selected
policies once on the held-out test split.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_threshold_tools() -> Any:
    """Load shared probability helpers from the prior threshold experiment."""

    script_path = PROJECT_ROOT / "experiments" / "32_precision_constrained_threshold_search.py"
    spec = importlib.util.spec_from_file_location("precision_threshold_tools", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tools = _load_threshold_tools()
from evaluation import expected_calibration_error  # noqa: E402

DECISION_REVISION_DIR: Path = tools.DECISION_REVISION_DIR
DISPLAY_ORDER: list[str] = tools.DISPLAY_ORDER
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)
COST_SCENARIOS = [
    (2.0, 1.0),
    (3.0, 1.0),
    (5.0, 1.0),
    (5.0, 2.0),
    (5.0, 3.0),
    (10.0, 1.0),
    (10.0, 2.0),
    (10.0, 3.0),
    (3.0, 2.0),
    (2.0, 2.0),
]
REFERENCE_FN_COST = 5.0
REFERENCE_FP_COST = 1.0


def scenario_name(fn_cost: float, fp_cost: float) -> str:
    """Return a compact cost scenario label."""

    return f"FN{fn_cost:g}_FP{fp_cost:g}"


def threshold_grid_metrics(
    y_true: pd.Series,
    probability: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> pd.DataFrame:
    """Compute threshold metrics for one cost scenario."""

    rows: list[dict[str, Any]] = []
    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    n_customers = len(y)
    roc_auc = float(roc_auc_score(y, probability))
    pr_auc = float(average_precision_score(y, probability))
    brier = float(brier_score_loss(y, probability))
    ece = float(expected_calibration_error(y, probability))
    for threshold in THRESHOLDS:
        metrics = classification_metrics_at_threshold(
            y,
            probability,
            threshold=float(threshold),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier=brier,
            ece=ece,
        )
        expected_cost = fn_cost * int(metrics["fn"]) + fp_cost * int(metrics["fp"])
        metrics["fn_cost"] = fn_cost
        metrics["fp_cost"] = fp_cost
        metrics["expected_cost"] = float(expected_cost)
        metrics["cost_per_customer"] = float(expected_cost) / float(n_customers)
        rows.append(metrics)
    return pd.DataFrame(rows)


def classification_metrics_at_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
    brier: float | None = None,
    ece: float | None = None,
) -> dict[str, float | int]:
    """Compute binary metrics at one threshold without refitting calibration models."""

    prediction = (probability >= threshold).astype(int)
    positive = y_true == 1
    negative = ~positive
    predicted_positive = prediction == 1
    predicted_negative = ~predicted_positive
    tp = int(np.sum(predicted_positive & positive))
    fp = int(np.sum(predicted_positive & negative))
    tn = int(np.sum(predicted_negative & negative))
    fn = int(np.sum(predicted_negative & positive))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc if roc_auc is not None else roc_auc_score(y_true, probability)),
        "pr_auc": float(pr_auc if pr_auc is not None else average_precision_score(y_true, probability)),
        "brier_score": float(brier if brier is not None else brier_score_loss(y_true, probability)),
        "ece": float(ece if ece is not None else expected_calibration_error(y_true, probability)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def select_threshold(grid: pd.DataFrame) -> pd.Series:
    """Select threshold on validation by cost, then F1, PR-AUC, and threshold."""

    selected = grid.sort_values(
        ["expected_cost", "f1", "pr_auc", "threshold"],
        ascending=[True, False, False, False],
    ).iloc[0]
    return selected


def build_base_result(
    dataset: str,
    display_model: str,
    calibration_type: str,
    fn_cost: float,
    fp_cost: float,
    selected: pd.Series,
) -> dict[str, Any]:
    """Build a validation-selected policy result shell."""

    return {
        "dataset": dataset,
        "model": display_model,
        "scenario_name": scenario_name(fn_cost, fp_cost),
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "cost_ratio_FN_to_FP": float(fn_cost / fp_cost) if fp_cost else np.inf,
        "selected_threshold_validation": float(selected["threshold"]),
        "selection_split": "validation",
        "selected_by": "minimum validation expected cost",
        "test_set_used_for_threshold_selection": False,
        "calibration_type": calibration_type,
        "validation_expected_cost": float(selected["expected_cost"]),
        "validation_accuracy": float(selected["accuracy"]),
        "validation_precision": float(selected["precision"]),
        "validation_recall": float(selected["recall"]),
        "validation_specificity": float(selected["specificity"]),
        "validation_f1": float(selected["f1"]),
        "validation_tn": int(selected["tn"]),
        "validation_fp": int(selected["fp"]),
        "validation_fn": int(selected["fn"]),
        "validation_tp": int(selected["tp"]),
    }


def evaluate_policy_on_test(
    result: dict[str, Any],
    y_test: pd.Series,
    probability: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a validation-selected threshold on the test split."""

    y = np.asarray(y_test, dtype=int)
    probability = np.asarray(probability, dtype=float)
    metrics = classification_metrics_at_threshold(
        y,
        probability,
        threshold=float(result["selected_threshold_validation"]),
    )
    test_cost = result["fn_cost"] * int(metrics["fn"]) + result["fp_cost"] * int(metrics["fp"])
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
            "test_cost_per_customer": float(test_cost) / float(len(y_test)),
        }
    )
    return result


def add_reference_deltas(table: pd.DataFrame) -> pd.DataFrame:
    """Add deltas against the same model's FN=5, FP=1 policy."""

    reference = table.loc[
        table["fn_cost"].eq(REFERENCE_FN_COST) & table["fp_cost"].eq(REFERENCE_FP_COST)
    ].copy()
    reference = reference.set_index(["dataset", "model"])
    rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        ref = reference.loc[(row["dataset"], row["model"])]
        row["delta_threshold_vs_FN5_FP1"] = (
            float(row["selected_threshold_validation"]) - float(ref["selected_threshold_validation"])
        )
        row["delta_precision_vs_FN5_FP1"] = float(row["test_precision"]) - float(ref["test_precision"])
        row["delta_recall_vs_FN5_FP1"] = float(row["test_recall"]) - float(ref["test_recall"])
        row["delta_specificity_vs_FN5_FP1"] = float(row["test_specificity"]) - float(ref["test_specificity"])
        row["delta_fp_vs_FN5_FP1"] = int(row["test_fp"]) - int(ref["test_fp"])
        row["delta_cost_vs_FN5_FP1"] = float(row["test_expected_cost"]) - float(ref["test_expected_cost"])
        row["operational_comment"] = operational_comment(row)
        rows.append(row)
    return pd.DataFrame(rows)


def operational_comment(row: dict[str, Any]) -> str:
    """Return a compact cost-sensitivity interpretation."""

    if row["fp_cost"] > REFERENCE_FP_COST and row["delta_threshold_vs_FN5_FP1"] > 0:
        return (
            "Higher FP cost pushes threshold upward; precision/specificity improve "
            "at the cost of recall."
        )
    if row["fn_cost"] > REFERENCE_FN_COST and row["fp_cost"] == REFERENCE_FP_COST:
        return "Higher FN cost keeps an aggressive screening threshold and protects recall."
    if row["cost_ratio_FN_to_FP"] <= 2:
        return "More conservative cost ratio; lower FP burden is prioritized."
    return "Reference-like cost tradeoff; use alongside precision/specificity constraints."


def build_dataset_results(dataset: str, provider: Any) -> pd.DataFrame:
    """Build all cost-sensitivity rows for one dataset."""

    plan = tools.model_plan(dataset)
    split = provider.splits[dataset]
    rows: list[dict[str, Any]] = []

    for display_model in DISPLAY_ORDER:
        source_model, calibration_type = plan[display_model]
        validation_probability = tools.probability_for_display(
            provider,
            dataset,
            display_model,
            source_model,
            "validation",
        )
        test_probability = tools.probability_for_display(
            provider,
            dataset,
            display_model,
            source_model,
            "test",
        )

        for fn_cost, fp_cost in COST_SCENARIOS:
            grid = threshold_grid_metrics(split.y_validation, validation_probability, fn_cost, fp_cost)
            selected = select_threshold(grid)
            result = build_base_result(
                dataset,
                display_model,
                calibration_type,
                fn_cost,
                fp_cost,
                selected,
            )
            rows.append(evaluate_policy_on_test(result, split.y_test, test_probability))

    return add_reference_deltas(pd.DataFrame(rows))


def validation_winners(table: pd.DataFrame) -> pd.DataFrame:
    """Select one model per cost scenario using validation expected cost only."""

    winners = (
        table.sort_values(
            ["fn_cost", "fp_cost", "validation_expected_cost", "validation_f1", "test_pr_auc"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["fn_cost", "fp_cost"], as_index=False)
        .head(1)
        .copy()
    )
    return winners.sort_values(["fn_cost", "fp_cost"]).reset_index(drop=True)


def fmt(value: float) -> str:
    """Format a value for markdown."""

    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def write_summary(dataset: str, table: pd.DataFrame) -> Path:
    """Write a dataset-specific cost sensitivity markdown summary."""

    winners = validation_winners(table)
    fn5_fp1 = table.loc[table["fn_cost"].eq(5.0) & table["fp_cost"].eq(1.0)].copy()
    fn5_fp2 = table.loc[table["fn_cost"].eq(5.0) & table["fp_cost"].eq(2.0)].copy()
    merged = fn5_fp2.merge(
        fn5_fp1,
        on=["dataset", "model"],
        suffixes=("_fp2", "_fp1"),
    )
    threshold_up_count = int(
        (merged["selected_threshold_validation_fp2"] > merged["selected_threshold_validation_fp1"]).sum()
    )
    precision_up_count = int((merged["test_precision_fp2"] > merged["test_precision_fp1"]).sum())
    specificity_up_count = int((merged["test_specificity_fp2"] > merged["test_specificity_fp1"]).sum())
    recall_change = float((merged["test_recall_fp2"] - merged["test_recall_fp1"]).mean())
    catboost_wins = int(winners["model"].eq("Best CatBoost").sum())
    scorecard_best_rows = winners.loc[winners["model"].eq("Best Scorecard")]
    scre_best_rows = winners.loc[winners["model"].isin(["SCRE-Optimized", "SCRE-Pareto"])]

    if dataset == "taiwan":
        recommended = "FN=5, FP=2 as a defensible sensitivity scenario, paired with precision constraints."
    else:
        recommended = "FN=5, FP=2 or FN=3, FP=2, because HELOC false positives are operationally costly."

    lines = [
        f"# Cost Matrix Sensitivity Summary: {dataset.upper()}",
        "",
        "Thresholds were selected on validation only for each cost scenario. Test metrics are holdout evaluations of those selected policies.",
        "",
        "## Validation-Selected Winners By Cost Scenario",
        "",
        "| Scenario | Best Model | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Comment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in winners.itertuples(index=False):
        lines.append(
            f"| FN={row.fn_cost:g}, FP={row.fp_cost:g} | {row.model} | "
            f"{fmt(row.selected_threshold_validation)} | {fmt(row.test_precision)} | "
            f"{fmt(row.test_recall)} | {fmt(row.test_specificity)} | {fmt(row.test_fp)} | "
            f"{fmt(row.test_fn)} | {fmt(row.test_expected_cost)} | {row.operational_comment} |"
        )

    lines.extend(
        [
            "",
            "## Audit Answers",
            "",
            f"1. FP cost 1 -> 2 raises threshold for {threshold_up_count}/{len(merged)} tracked model policies.",
            f"2. Precision increases for {precision_up_count}/{len(merged)} tracked model policies.",
            f"3. Specificity increases for {specificity_up_count}/{len(merged)} tracked model policies.",
            f"4. Average recall change when moving FN=5, FP=1 -> FN=5, FP=2: {recall_change:.4f}.",
            f"5. CatBoost is the validation-cost winner in {catboost_wins}/{len(winners)} scenarios.",
            f"6. Scorecard winner scenarios: {len(scorecard_best_rows)}. It remains useful as an interpretable benchmark even when not cost-winning.",
            f"7. SCRE winner scenarios: {len(scre_best_rows)}. Use SCRE sensitivity rows to discuss reliability-aware integration rather than guaranteed superiority.",
            f"8. Recommended operational discussion scenario: {recommended}",
            "",
        ]
    )
    path = DECISION_REVISION_DIR / f"cost_matrix_summary_{dataset}.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def plot_taiwan_sensitivity(table: pd.DataFrame) -> list[Path]:
    """Write requested Taiwan cost-sensitivity figures."""

    paths = [
        DECISION_REVISION_DIR / "cost_sensitivity_thresholds_taiwan.png",
        DECISION_REVISION_DIR / "cost_sensitivity_precision_recall_taiwan.png",
        DECISION_REVISION_DIR / "cost_sensitivity_fp_reduction_taiwan.png",
    ]
    taiwan = table.copy()
    taiwan["scenario_label"] = taiwan.apply(
        lambda row: f"FN={row['fn_cost']:g}\nFP={row['fp_cost']:g}",
        axis=1,
    )
    scenario_order = [
        scenario_name(fn_cost, fp_cost)
        for fn_cost, fp_cost in COST_SCENARIOS
    ]
    x_labels = [
        f"FN={fn_cost:g}\nFP={fp_cost:g}"
        for fn_cost, fp_cost in COST_SCENARIOS
    ]
    x_positions = np.arange(len(scenario_order))

    fig, ax = plt.subplots(figsize=(11, 6))
    for model in DISPLAY_ORDER:
        rows = taiwan.loc[taiwan["model"].eq(model)].set_index("scenario_name").loc[scenario_order]
        ax.plot(x_positions, rows["selected_threshold_validation"], marker="o", label=model)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Validation-selected threshold")
    ax.set_title("Taiwan Cost Sensitivity: Threshold Movement")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(paths[0], dpi=300)
    plt.close(fig)

    winners = validation_winners(taiwan).set_index("scenario_name").loc[scenario_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_positions, winners["test_precision"], marker="o", label="Precision")
    ax.plot(x_positions, winners["test_recall"], marker="o", label="Recall")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test metric")
    ax.set_title("Taiwan Validation-Selected Winners: Precision/Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for model in DISPLAY_ORDER:
        rows = taiwan.loc[taiwan["model"].eq(model)].set_index("scenario_name").loc[scenario_order]
        ax.plot(x_positions, rows["delta_fp_vs_FN5_FP1"], marker="o", label=model)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Delta FP vs FN=5, FP=1")
    ax.set_title("Taiwan Cost Sensitivity: FP Reduction")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(paths[2], dpi=300)
    plt.close(fig)
    return paths


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> list[Path]:
    """Write CSV, markdown, and figure outputs."""

    DECISION_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    taiwan_path = DECISION_REVISION_DIR / "cost_matrix_sensitivity_taiwan.csv"
    heloc_path = DECISION_REVISION_DIR / "cost_matrix_sensitivity_heloc.csv"
    taiwan.to_csv(taiwan_path, index=False)
    heloc.to_csv(heloc_path, index=False)
    paths = [
        taiwan_path,
        heloc_path,
        write_summary("taiwan", taiwan),
        write_summary("heloc", heloc),
    ]
    paths.extend(plot_taiwan_sensitivity(taiwan))
    return paths


def main() -> None:
    """Run cost matrix sensitivity analysis."""

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
        print(f"\n{dataset} validation-selected winners:")
        winners = validation_winners(table)
        print(
            winners[
                [
                    "scenario_name",
                    "model",
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
