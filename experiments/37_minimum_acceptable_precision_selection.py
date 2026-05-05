"""Select operationally eligible model policies from existing result tables.

This script does not train models and does not create features. It consolidates
the current decision-revision result tables, applies minimum acceptable
precision/recall/specificity/AUC criteria, and ranks eligible policies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_REVISION_DIR = PROJECT_ROOT / "outputs" / "decision_revision"
FN_COST = 5.0
FP_COST = 1.0


CRITERIA = {
    "taiwan": {
        "basic": {
            "precision": (">=", 0.40),
            "recall": (">=", 0.60),
            "specificity": (">=", 0.55),
            "pr_auc": (">=", 0.54),
            "roc_auc": (">=", 0.77),
        },
        "strict": {
            "precision": (">=", 0.45),
            "recall": (">=", 0.55),
            "specificity": (">=", 0.60),
            "pr_auc": (">=", 0.54),
            "roc_auc": (">=", 0.77),
        },
    },
    "heloc": {
        "basic": {
            "precision": (">=", 0.55),
            "recall": (">=", 0.90),
            "pr_auc": (">=", 0.78),
            "roc_auc": (">=", 0.78),
            "specificity": (">=", 0.10),
        },
        "strict": {
            "precision": (">=", 0.60),
            "recall": (">=", 0.85),
            "pr_auc": (">=", 0.78),
            "roc_auc": (">=", 0.78),
            "specificity": (">=", 0.20),
        },
    },
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required result table."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def base_metric_map(dataset: str) -> dict[str, dict[str, float]]:
    """Map frozen model names to threshold-independent test metrics."""

    rows: dict[str, dict[str, float]] = {}
    current = read_csv(DECISION_REVISION_DIR / f"current_threshold_baseline_{dataset}.csv")
    for row in current.to_dict(orient="records"):
        rows[str(row["model"])] = {
            "roc_auc": float(row["roc_auc"]),
            "pr_auc": float(row["pr_auc"]),
            "brier": float(row["brier"]),
            "ece": float(row["ece"]),
        }
    return rows


def dataset_totals(dataset: str) -> dict[str, int]:
    """Return test-set positive/negative totals from baseline tables."""

    current = read_csv(DECISION_REVISION_DIR / f"current_threshold_baseline_{dataset}.csv")
    row = current.iloc[0]
    positives = int(row["tp"]) + int(row["fn"])
    negatives = int(row["tn"]) + int(row["fp"])
    return {"positives": positives, "negatives": negatives, "n": positives + negatives}


def add_candidate(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Append one normalized candidate row."""

    rows.append(
        {
            "dataset": row["dataset"],
            "model": row["model"],
            "policy_type": row["policy_type"],
            "threshold_or_band": row["threshold_or_band"],
            "precision": row["precision"],
            "recall": row["recall"],
            "specificity": row["specificity"],
            "f1": row["f1"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "brier": row["brier"],
            "ece": row["ece"],
            "tn": row["tn"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tp": row["tp"],
            "expected_cost": row["expected_cost"],
            "source_file": row["source_file"],
            "source_comment": row.get("source_comment", ""),
        }
    )


def normalize_current_thresholds(dataset: str) -> pd.DataFrame:
    """Normalize frozen fixed/cost/current threshold rows."""

    path = DECISION_REVISION_DIR / f"current_threshold_baseline_{dataset}.csv"
    table = read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        add_candidate(
            rows,
            {
                "dataset": dataset,
                "model": str(row["model"]),
                "policy_type": f"current_threshold:{row['threshold_type']}",
                "threshold_or_band": f"threshold={float(row['threshold']):.3f}",
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "specificity": float(row["specificity"]),
                "f1": float(row["f1"]),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
                "brier": float(row["brier"]),
                "ece": float(row["ece"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tp": int(row["tp"]),
                "expected_cost": float(row["expected_cost_FN5_FP1"]),
                "source_file": path.name,
                "source_comment": str(row.get("notes", "")),
            },
        )
    return pd.DataFrame(rows)


def normalize_precision_constraints(dataset: str) -> pd.DataFrame:
    """Normalize precision-constrained threshold policies."""

    path = DECISION_REVISION_DIR / f"precision_constrained_test_results_{dataset}.csv"
    table = read_csv(path)
    table = table.loc[table["feasible_on_validation"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        add_candidate(
            rows,
            {
                "dataset": dataset,
                "model": str(row["model"]),
                "policy_type": f"precision_constraint:{row['constraint_name']}",
                "threshold_or_band": f"threshold={float(row['selected_threshold_validation']):.3f}",
                "precision": float(row["test_precision"]),
                "recall": float(row["test_recall"]),
                "specificity": float(row["test_specificity"]),
                "f1": float(row["test_f1"]),
                "roc_auc": float(row["test_roc_auc"]),
                "pr_auc": float(row["test_pr_auc"]),
                "brier": float(row["test_brier"]),
                "ece": float(row["test_ece"]),
                "tn": int(row["test_tn"]),
                "fp": int(row["test_fp"]),
                "fn": int(row["test_fn"]),
                "tp": int(row["test_tp"]),
                "expected_cost": float(row["test_expected_cost"]),
                "source_file": path.name,
                "source_comment": str(row.get("operational_comment", "")),
            },
        )
    return pd.DataFrame(rows)


def normalize_cost_sensitivity(dataset: str) -> pd.DataFrame:
    """Normalize cost-matrix threshold sensitivity policies."""

    path = DECISION_REVISION_DIR / f"cost_matrix_sensitivity_{dataset}.csv"
    table = read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        add_candidate(
            rows,
            {
                "dataset": dataset,
                "model": str(row["model"]),
                "policy_type": f"cost_matrix:{row['scenario_name']}",
                "threshold_or_band": f"threshold={float(row['selected_threshold_validation']):.3f}",
                "precision": float(row["test_precision"]),
                "recall": float(row["test_recall"]),
                "specificity": float(row["test_specificity"]),
                "f1": float(row["test_f1"]),
                "roc_auc": float(row["test_roc_auc"]),
                "pr_auc": float(row["test_pr_auc"]),
                "brier": float(row["test_brier"]),
                "ece": float(row["test_ece"]),
                "tn": int(row["test_tn"]),
                "fp": int(row["test_fp"]),
                "fn": int(row["test_fn"]),
                "tp": int(row["test_tp"]),
                "expected_cost": float(row["test_expected_cost"]),
                "source_file": path.name,
                "source_comment": str(row.get("operational_comment", "")),
            },
        )
    return pd.DataFrame(rows)


def normalize_manual_review(dataset: str) -> pd.DataFrame:
    """Normalize manual-review bands as high-risk-vs-not-high-risk policies."""

    path = DECISION_REVISION_DIR / f"manual_review_band_optimized_{dataset}.csv"
    table = read_csv(path)
    table = table.loc[table["feasible_on_validation"].astype(bool)].copy()
    totals = dataset_totals(dataset)
    metrics = base_metric_map(dataset)
    rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        model = str(row["model"])
        high_count = int(row["test_high_risk_count"])
        fp = int(row["test_high_risk_false_positive_count"])
        tp = high_count - fp
        fn = totals["positives"] - tp
        tn = totals["negatives"] - fp
        precision = float(row["test_high_risk_precision"])
        recall = float(row["test_high_risk_recall"])
        specificity = tn / totals["negatives"] if totals["negatives"] else np.nan
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metric = metrics.get(model, {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "ece": np.nan})
        add_candidate(
            rows,
            {
                "dataset": dataset,
                "model": model,
                "policy_type": f"manual_review_band:{row['constraint_setting']}",
                "threshold_or_band": f"t_low={float(row['t_low_validation']):.2f}; t_high={float(row['t_high_validation']):.2f}",
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "roc_auc": metric["roc_auc"],
                "pr_auc": metric["pr_auc"],
                "brier": metric["brier"],
                "ece": metric["ece"],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "expected_cost": float(row["test_remaining_expected_cost"]),
                "source_file": path.name,
                "source_comment": (
                    f"Three-way policy; expected_cost is remaining auto-decision cost. "
                    f"manual_review_rate={float(row['test_manual_review_rate']):.4f}; "
                    f"low_risk_default_rate={float(row['test_default_rate_low_risk']):.4f}. "
                    f"{row.get('notes', '')}"
                ),
            },
        )
    return pd.DataFrame(rows)


def normalize_imbalance_training(dataset: str) -> pd.DataFrame:
    """Normalize imbalance-training revision policies."""

    path = DECISION_REVISION_DIR / f"imbalance_training_variants_{dataset}.csv"
    table = read_csv(path)
    table = table.dropna(subset=["selected_threshold_validation"]).copy()
    rows: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        model = f"{row['model_family']}::{row['variant_name']}"
        add_candidate(
            rows,
            {
                "dataset": dataset,
                "model": model,
                "policy_type": f"imbalance_training:{row['calibration_type']}:{row['threshold_policy']}",
                "threshold_or_band": f"threshold={float(row['selected_threshold_validation']):.3f}",
                "precision": float(row["test_precision"]),
                "recall": float(row["test_recall"]),
                "specificity": float(row["test_specificity"]),
                "f1": float(row["test_f1"]),
                "roc_auc": float(row["test_roc_auc"]),
                "pr_auc": float(row["test_pr_auc"]),
                "brier": float(row["test_brier"]),
                "ece": float(row["test_ece"]),
                "tn": int(row["test_tn"]),
                "fp": int(row["test_fp"]),
                "fn": int(row["test_fn"]),
                "tp": int(row["test_tp"]),
                "expected_cost": float(row["test_expected_cost_FN5_FP1"]),
                "source_file": path.name,
                "source_comment": (
                    f"{row.get('imbalance_strategy', '')}; "
                    f"{row.get('probability_calibration_comment', '')}; "
                    f"{row.get('operational_comment', '')}"
                ),
            },
        )
    return pd.DataFrame(rows)


def all_candidates(dataset: str) -> pd.DataFrame:
    """Collect all normalized candidate policies for one dataset."""

    tables = [
        normalize_current_thresholds(dataset),
        normalize_precision_constraints(dataset),
        normalize_cost_sensitivity(dataset),
        normalize_manual_review(dataset),
        normalize_imbalance_training(dataset),
    ]
    candidates = pd.concat(tables, ignore_index=True)
    candidates = candidates.drop_duplicates(
        subset=["dataset", "model", "policy_type", "threshold_or_band"],
        keep="first",
    ).reset_index(drop=True)
    return candidates


def failed_criteria(row: pd.Series, criterion: dict[str, tuple[str, float]]) -> list[str]:
    """Return failed criterion names for one row."""

    failed: list[str] = []
    for metric, (_, cutoff) in criterion.items():
        value = row.get(metric, np.nan)
        if pd.isna(value) or float(value) < cutoff:
            failed.append(f"{metric}<{cutoff:g}")
    return failed


def apply_eligibility(table: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add eligibility, ranks, winners, and comments."""

    result = table.copy()
    basic = CRITERIA[dataset]["basic"]
    strict = CRITERIA[dataset]["strict"]
    result["failed_criteria_basic"] = result.apply(lambda row: "; ".join(failed_criteria(row, basic)), axis=1)
    result["failed_criteria_strict"] = result.apply(lambda row: "; ".join(failed_criteria(row, strict)), axis=1)
    result["eligible_basic"] = result["failed_criteria_basic"].eq("")
    result["eligible_strict"] = result["failed_criteria_strict"].eq("")
    result["failed_criteria"] = np.where(
        result["eligible_basic"],
        "",
        result["failed_criteria_basic"],
    )
    result["rank_basic"] = np.nan
    result["rank_strict"] = np.nan
    result["winner_basic"] = False
    result["winner_strict"] = False

    for eligibility_col, rank_col, winner_col in [
        ("eligible_basic", "rank_basic", "winner_basic"),
        ("eligible_strict", "rank_strict", "winner_strict"),
    ]:
        eligible = result.loc[result[eligibility_col]].copy()
        if eligible.empty:
            continue
        ranked = eligible.sort_values(
            ["expected_cost", "pr_auc", "ece", "f1", "fp"],
            ascending=[True, False, True, False, True],
        ).copy()
        ranked[rank_col] = np.arange(1, len(ranked) + 1)
        result.loc[ranked.index, rank_col] = ranked[rank_col]
        result.loc[ranked.index[0], winner_col] = True

    result["decision_comment"] = result.apply(decision_comment, axis=1)
    output_cols = [
        "dataset",
        "model",
        "policy_type",
        "threshold_or_band",
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
        "expected_cost",
        "eligible_basic",
        "eligible_strict",
        "failed_criteria",
        "rank_basic",
        "rank_strict",
        "winner_basic",
        "winner_strict",
        "decision_comment",
        "source_file",
        "source_comment",
        "failed_criteria_basic",
        "failed_criteria_strict",
    ]
    return result[output_cols].sort_values(
        ["eligible_basic", "rank_basic", "expected_cost", "model"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def decision_comment(row: pd.Series) -> str:
    """Create a model-selection committee style comment."""

    if bool(row["winner_strict"]):
        return "Strict winner: strongest operationally eligible policy under minimum acceptable criteria."
    if bool(row["winner_basic"]):
        return "Basic winner: lowest-cost eligible policy under minimum acceptable criteria."
    if bool(row["eligible_strict"]):
        return "Strict eligible; usable candidate, but not the lowest-cost ranked winner."
    if bool(row["eligible_basic"]):
        if "manual_review_band" in str(row["policy_type"]):
            return "Basic eligible three-way decision-support policy; not an automatic rejection rule."
        return "Basic eligible binary policy; usable if its FP/review burden is acceptable."
    failed = str(row["failed_criteria_basic"])
    if "precision" in failed:
        return "Not eligible: precision is below the minimum acceptable operational threshold."
    if "recall" in failed:
        return "Not eligible: recall is too low for screening/default capture."
    if "specificity" in failed:
        return "Not eligible: false-positive control remains too weak."
    if "pr_auc" in failed or "roc_auc" in failed:
        return "Not eligible: ranking quality is below dataset minimum."
    return "Not eligible under current committee criteria."


def final_selection(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> pd.DataFrame:
    """Build final revised operational selection table."""

    rows: list[pd.Series] = []
    for dataset, table in [("taiwan", taiwan), ("heloc", heloc)]:
        for scope, winner_col in [("basic", "winner_basic"), ("strict", "winner_strict")]:
            winner = table.loc[table[winner_col]].copy()
            if winner.empty:
                continue
            row = winner.iloc[0].copy()
            row["selection_scope"] = scope
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    """Format a numeric value for markdown."""

    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def top_eligible(table: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Return compact top eligible rows."""

    return table.loc[table["eligible_basic"]].sort_values("rank_basic").head(limit)


def old_catboost_answer(table: pd.DataFrame, dataset: str) -> str:
    """Return old CatBoost cost-threshold eligibility answer."""

    row = table.loc[
        table["model"].eq("Best CatBoost")
        & table["policy_type"].eq("current_threshold:cost_optimal_current")
    ]
    if row.empty:
        return f"{dataset}: old CatBoost cost-threshold row not found."
    row = row.iloc[0]
    return (
        f"{dataset}: {'YES' if row['eligible_basic'] else 'NO'} basic, "
        f"{'YES' if row['eligible_strict'] else 'NO'} strict "
        f"(precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
        f"specificity={row['specificity']:.4f})."
    )


def write_summary(taiwan: pd.DataFrame, heloc: pd.DataFrame, final: pd.DataFrame) -> Path:
    """Write eligibility summary markdown."""

    lines = [
        "# Minimum Acceptable Precision Model Selection",
        "",
        "No model was trained in this step. Existing result tables were consolidated and ranked by eligibility criteria.",
        "",
        "## Old CatBoost Cost Threshold",
        "",
        f"- {old_catboost_answer(taiwan, 'Taiwan')}",
        f"- {old_catboost_answer(heloc, 'HELOC')}",
        "",
        "## Taiwan Top Eligible Policies",
        "",
        "| Rank | Model/Policy | Precision | Recall | Specificity | FP | FN | Cost | Strict? | Comment |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in top_eligible(taiwan).itertuples(index=False):
        lines.append(
            f"| {int(row.rank_basic)} | {row.model} / {row.policy_type} | {fmt(row.precision)} | "
            f"{fmt(row.recall)} | {fmt(row.specificity)} | {fmt(row.fp)} | {fmt(row.fn)} | "
            f"{fmt(row.expected_cost)} | {'YES' if row.eligible_strict else 'NO'} | {row.decision_comment} |"
        )
    lines.extend(
        [
            "",
            "## HELOC Top Eligible Policies",
            "",
            "| Rank | Model/Policy | Precision | Recall | Specificity | FP | FN | Cost | Strict? | Comment |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in top_eligible(heloc).itertuples(index=False):
        lines.append(
            f"| {int(row.rank_basic)} | {row.model} / {row.policy_type} | {fmt(row.precision)} | "
            f"{fmt(row.recall)} | {fmt(row.specificity)} | {fmt(row.fp)} | {fmt(row.fn)} | "
            f"{fmt(row.expected_cost)} | {'YES' if row.eligible_strict else 'NO'} | {row.decision_comment} |"
        )

    precision_winner_changed = (
        "YES; Taiwan old CatBoost cost threshold fails precision eligibility, "
        "so the winner moves to constrained/manual-review policies."
    )
    manual_review_taiwan = int(taiwan["policy_type"].str.startswith("manual_review_band").where(taiwan["eligible_basic"], False).sum())
    manual_review_heloc = int(heloc["policy_type"].str.startswith("manual_review_band").where(heloc["eligible_basic"], False).sum())
    scorecard_eligible = {
        "taiwan": int(taiwan["model"].eq("Best Scorecard").where(taiwan["eligible_basic"], False).sum()),
        "heloc": int(heloc["model"].eq("Best Scorecard").where(heloc["eligible_basic"], False).sum()),
    }
    scre_eligible = {
        "taiwan": int(taiwan["model"].isin(["SCRE-Optimized", "SCRE-Pareto", "Old SCRE-Credit"]).where(taiwan["eligible_basic"], False).sum()),
        "heloc": int(heloc["model"].isin(["SCRE-Optimized", "SCRE-Pareto", "Old SCRE-Credit"]).where(heloc["eligible_basic"], False).sum()),
    }
    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"1. Old CatBoost cost-threshold eligible? {old_catboost_answer(taiwan, 'Taiwan')} {old_catboost_answer(heloc, 'HELOC')}",
            f"2. Precision constraint altında winner değişiyor mu? {precision_winner_changed}",
            f"3. Manual review policy eligible oluyor mu? YES; Taiwan eligible manual-review rows={manual_review_taiwan}, HELOC eligible manual-review rows={manual_review_heloc}.",
            f"4. Scorecard daha dengeli mi? Taiwan eligible Scorecard rows={scorecard_eligible['taiwan']}; HELOC eligible Scorecard rows={scorecard_eligible['heloc']}. HELOC tarafında daha dengeli ve savunulabilir.",
            f"5. SCRE hangi koşullarda mantıklı? SCRE rows passing basic eligibility: Taiwan={scre_eligible['taiwan']}, HELOC={scre_eligible['heloc']}; özellikle manual-review veya stricter FP-control bandlarında anlamlı.",
            "6. En gerçekçi final operational model/policy: Taiwan için manual-review destekli veya precision-constrained policy; HELOC için SCRE/Scorecard/CatBoost arasında eligibility-passing decision-support policy.",
            "",
            "## Final Revised Winners",
            "",
            "| Dataset | Scope | Model | Policy | Precision | Recall | Specificity | FP | FN | Cost |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.selection_scope} | {row.model} | {row.policy_type} | "
            f"{fmt(row.precision)} | {fmt(row.recall)} | {fmt(row.specificity)} | "
            f"{fmt(row.fp)} | {fmt(row.fn)} | {fmt(row.expected_cost)} |"
        )

    path = DECISION_REVISION_DIR / "eligibility_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame, final: pd.DataFrame) -> list[Path]:
    """Write requested outputs."""

    paths = [
        DECISION_REVISION_DIR / "eligible_model_selection_taiwan.csv",
        DECISION_REVISION_DIR / "eligible_model_selection_heloc.csv",
        DECISION_REVISION_DIR / "final_operational_model_selection_revised.csv",
    ]
    taiwan.to_csv(paths[0], index=False)
    heloc.to_csv(paths[1], index=False)
    final.to_csv(paths[2], index=False)
    paths.append(write_summary(taiwan, heloc, final))
    return paths


def main() -> None:
    """Run minimum acceptable precision model selection."""

    taiwan = apply_eligibility(all_candidates("taiwan"), "taiwan")
    heloc = apply_eligibility(all_candidates("heloc"), "heloc")
    final = final_selection(taiwan, heloc)
    paths = write_outputs(taiwan, heloc, final)
    for path in paths:
        print(path)

    for dataset, table in [("Taiwan", taiwan), ("HELOC", heloc)]:
        print(f"\n{dataset} top eligible policies:")
        print(
            top_eligible(table)[
                [
                    "model",
                    "policy_type",
                    "precision",
                    "recall",
                    "specificity",
                    "fp",
                    "fn",
                    "expected_cost",
                    "eligible_strict",
                    "rank_basic",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
