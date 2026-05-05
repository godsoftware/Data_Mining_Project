"""Consolidate final revised comparison from existing decision-revision outputs.

This script does not train models, does not generate features, and does not run
new experiments. It merges the Prompt 1-7 decision-revision artifacts into final
model-selection and validation committee tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECISION_REVISION_DIR = PROJECT_ROOT / "outputs" / "decision_revision"


def read_csv(name: str) -> pd.DataFrame:
    """Read a decision-revision CSV output."""

    path = DECISION_REVISION_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def parse_threshold_or_band(text: str) -> dict[str, float]:
    """Parse threshold=..., t_low=..., or t_high=... text."""

    values: dict[str, float] = {}
    for part in str(text).split(";"):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        values[key.strip()] = float(value.strip())
    return values


def useful_range_text(dataset: str, model: str) -> str:
    """Return net-benefit useful threshold ranges for a display model."""

    ranges = read_csv("net_benefit_model_ranges.csv")
    model_ranges = ranges.loc[
        ranges["dataset"].eq(dataset)
        & ranges["model"].eq(model)
        & ranges["range_type"].eq("better_than_both")
    ].copy()
    if model_ranges.empty:
        return "not available"
    parts = []
    for row in model_ranges.itertuples(index=False):
        if abs(float(row.threshold_start) - float(row.threshold_end)) <= 1e-9:
            parts.append(f"{row.threshold_start:.2f}")
        else:
            parts.append(f"{row.threshold_start:.2f}-{row.threshold_end:.2f}")
    return ", ".join(parts)


def model_family(model: str) -> str:
    """Map display model names to model families."""

    if "CatBoost" in model:
        return "CatBoost"
    if "XGBoost" in model:
        return "XGBoost"
    if "LightGBM" in model:
        return "LightGBM"
    if "Scorecard" in model:
        return "Scorecard"
    if "SCRE" in model:
        return "SCRE-Credit"
    return model.split("::")[0]


def manual_review_lookup(dataset: str) -> pd.DataFrame:
    """Return manual-review rows with normalized model/policy keys."""

    table = read_csv(f"manual_review_band_optimized_{dataset}.csv").copy()
    table["policy_type"] = "manual_review_band:" + table["constraint_setting"].astype(str)
    return table


def cost_matrix_lookup(dataset: str) -> pd.DataFrame:
    """Return cost-matrix rows with normalized model/policy keys."""

    table = read_csv(f"cost_matrix_sensitivity_{dataset}.csv").copy()
    table["policy_type"] = "cost_matrix:" + table["scenario_name"].astype(str)
    return table


def eligible_table(dataset: str) -> pd.DataFrame:
    """Read eligible model selection output for a dataset."""

    return read_csv(f"eligible_model_selection_{dataset}.csv")


def fp_comment(dataset: str, system: str, model: str) -> str:
    """Return FP-profile interpretation when available."""

    table = read_csv(f"false_positive_system_comparison_{dataset}.csv")
    row = table.loc[table["system"].eq(system)]
    if row.empty:
        if model in {"Best CatBoost", "Best Scorecard", "SCRE-Optimized", "SCRE-Pareto", "Old SCRE-Credit"}:
            return "FP profile not directly profiled for this exact policy; related FP analysis suggests risk-like false positives."
        return "FP profile not directly profiled for this system."
    row = row.iloc[0]
    return (
        f"FP closer to {row['fp_closer_to']} than alternative group; "
        f"looks risky={bool(row['fp_looks_risky'])}; "
        f"similarity_score={float(row['overall_similarity_score']):.3f}."
    )


def base_from_eligible(dataset: str, system: str, row: pd.Series, revised_cost: float | None = None) -> dict[str, Any]:
    """Build one final comparison row from an eligibility table row."""

    manual_rate = np.nan
    auto_rate = 1.0
    if str(row["policy_type"]).startswith("manual_review_band"):
        manual = manual_review_lookup(dataset)
        match = manual.loc[
            manual["model"].eq(row["model"]) & manual["policy_type"].eq(row["policy_type"])
        ]
        if not match.empty:
            manual_rate = float(match.iloc[0]["test_manual_review_rate"])
            auto_rate = float(match.iloc[0]["test_auto_decision_rate"])

    binary_cost = float(5 * int(row["fn"]) + int(row["fp"]))
    return {
        "dataset": dataset,
        "system": system,
        "model": str(row["model"]),
        "model_family": model_family(str(row["model"])),
        "policy_type": str(row["policy_type"]),
        "threshold_or_band": str(row["threshold_or_band"]),
        "accuracy": float((int(row["tp"]) + int(row["tn"])) / (int(row["tp"]) + int(row["tn"]) + int(row["fp"]) + int(row["fn"]))),
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
        "expected_cost_FN5_FP1": binary_cost,
        "expected_cost_revised_policy": float(revised_cost if revised_cost is not None else row["expected_cost"]),
        "manual_review_rate": manual_rate,
        "auto_decision_rate": auto_rate,
        "net_benefit_useful_range": useful_range_text(dataset, str(row["model"])),
        "eligible_basic": bool(row["eligible_basic"]),
        "eligible_strict": bool(row["eligible_strict"]),
        "fp_profile_comment": fp_comment(dataset, system, str(row["model"])),
        "operational_rank": np.nan,
        "final_comment": "",
    }


def row_by_policy(dataset: str, model: str, policy_prefix: str, exact_policy: str | None = None) -> pd.Series:
    """Select one row from eligible table by model and policy."""

    table = eligible_table(dataset)
    rows = table.loc[table["model"].eq(model)].copy()
    if exact_policy is not None:
        rows = rows.loc[rows["policy_type"].eq(exact_policy)].copy()
    else:
        rows = rows.loc[rows["policy_type"].str.startswith(policy_prefix)].copy()
    if rows.empty:
        raise ValueError(f"No eligible-table row for {dataset}/{model}/{policy_prefix}/{exact_policy}")
    rows = rows.sort_values(
        ["eligible_basic", "rank_basic", "expected_cost", "precision"],
        ascending=[False, True, True, False],
    )
    return rows.iloc[0]


def current_catboost(dataset: str) -> dict[str, Any]:
    """Old CatBoost cost-threshold row."""

    row = row_by_policy(dataset, "Best CatBoost", "", "current_threshold:cost_optimal_current")
    result = base_from_eligible(dataset, "Old CatBoost cost threshold", row)
    result["final_comment"] = (
        "Old cost-minimizing binary policy; high recall but too many false positives for automatic rejection."
    )
    return result


def catboost_precision(dataset: str) -> dict[str, Any]:
    """Best CatBoost precision-constrained policy."""

    row = row_by_policy(dataset, "Best CatBoost", "precision_constraint")
    result = base_from_eligible(dataset, "CatBoost precision-constrained threshold", row)
    result["final_comment"] = "Precision-constrained binary policy; improves FP control versus old threshold."
    return result


def catboost_cost_matrix(dataset: str) -> dict[str, Any]:
    """CatBoost revised cost matrix policy."""

    row = row_by_policy(dataset, "Best CatBoost", "", "cost_matrix:FN5_FP2")
    cost_table = cost_matrix_lookup(dataset)
    cost_match = cost_table.loc[
        cost_table["model"].eq("Best CatBoost") & cost_table["policy_type"].eq("cost_matrix:FN5_FP2")
    ].iloc[0]
    result = base_from_eligible(
        dataset,
        "CatBoost revised cost matrix policy",
        row,
        revised_cost=float(cost_match["test_expected_cost"]),
    )
    result["final_comment"] = "FN=5, FP=2 sensitivity policy; more conservative than old FN=5, FP=1."
    return result


def catboost_manual(dataset: str) -> dict[str, Any]:
    """CatBoost bounded manual-review policy."""

    row = row_by_policy(dataset, "Best CatBoost", "", "manual_review_band:C_manual_review_rate_0_30")
    result = base_from_eligible(dataset, "CatBoost manual review band", row)
    result["final_comment"] = (
        "Bounded manual-review policy; best practical CatBoost option if review capacity exists."
    )
    return result


def best_imbalance(dataset: str) -> dict[str, Any]:
    """Best eligible training-level imbalance variant."""

    table = eligible_table(dataset)
    rows = table.loc[
        table["policy_type"].str.startswith("imbalance_training") & table["eligible_basic"].astype(bool)
    ].copy()
    if rows.empty:
        rows = table.loc[table["policy_type"].str.startswith("imbalance_training")].copy()
    rows = rows.sort_values(["rank_basic", "expected_cost", "precision"], ascending=[True, True, False])
    row = rows.iloc[0]
    result = base_from_eligible(dataset, "Best training-level imbalance variant", row)
    result["final_comment"] = "Retrained imbalance-strategy candidate; useful binary alternative, but not final winner."
    return result


def best_scorecard(dataset: str) -> dict[str, Any]:
    """Best eligible Scorecard policy."""

    row = row_by_policy(dataset, "Best Scorecard", "")
    result = base_from_eligible(dataset, "Best Scorecard policy", row)
    result["final_comment"] = "Best interpretable benchmark; valuable for validation even when not the operational winner."
    return result


def best_scre(dataset: str) -> dict[str, Any]:
    """Best eligible SCRE policy."""

    table = eligible_table(dataset)
    rows = table.loc[
        table["model"].isin(["SCRE-Optimized", "SCRE-Pareto", "Old SCRE-Credit"])
        & table["eligible_basic"].astype(bool)
    ].copy()
    rows = rows.sort_values(["rank_basic", "expected_cost", "precision"], ascending=[True, True, False])
    row = rows.iloc[0]
    result = base_from_eligible(dataset, "Best SCRE policy", row)
    result["final_comment"] = "Best reliability-aware framework policy; use as research framework, not a blanket automatic-rejection model."
    return result


def best_eligible(dataset: str) -> dict[str, Any]:
    """Best overall eligible model/policy."""

    table = eligible_table(dataset)
    rows = table.loc[table["eligible_basic"].astype(bool)].sort_values("rank_basic")
    row = rows.iloc[0]
    result = base_from_eligible(dataset, "Best eligible model/policy", row)
    result["final_comment"] = "Eligibility winner by committee criteria; check manual-review workload before operational use."
    return result


def heloc_external_policy() -> dict[str, Any]:
    """Best HELOC external validation policy row."""

    table = read_csv("final_operational_model_selection_revised.csv")
    row = table.loc[table["dataset"].eq("heloc") & table["selection_scope"].eq("basic")].iloc[0]
    result = base_from_eligible("heloc", "Best HELOC external validation policy", row)
    result["final_comment"] = "Best external-validation decision-support policy from revised eligibility selection."
    return result


def comparison_table(dataset: str) -> pd.DataFrame:
    """Build final comparison table for one dataset."""

    rows = [
        current_catboost(dataset),
        catboost_precision(dataset),
        catboost_cost_matrix(dataset),
        catboost_manual(dataset),
        best_imbalance(dataset),
        best_scorecard(dataset),
        best_scre(dataset),
        best_eligible(dataset),
    ]
    if dataset == "heloc":
        rows.append(heloc_external_policy())
    table = pd.DataFrame(rows)
    table["manual_review_practical"] = table["manual_review_rate"].isna() | (table["manual_review_rate"] <= 0.30)
    table["automatic_decision_suitable"] = (
        table["manual_review_rate"].isna()
        & table["eligible_basic"].astype(bool)
        & (table["precision"] >= (0.40 if dataset == "taiwan" else 0.55))
        & (table["specificity"] >= (0.55 if dataset == "taiwan" else 0.10))
    )
    table["ranking_bucket"] = np.select(
        [
            table["manual_review_rate"].notna() & table["manual_review_practical"] & table["eligible_strict"],
            table["manual_review_rate"].isna() & table["eligible_strict"],
            table["manual_review_rate"].isna() & table["eligible_basic"],
            table["manual_review_rate"].notna() & ~table["manual_review_practical"] & table["eligible_basic"],
            table["manual_review_rate"].notna() & table["manual_review_practical"],
        ],
        [0, 1, 2, 3, 4],
        default=5,
    )
    ranked = table.sort_values(
        [
            "ranking_bucket",
            "expected_cost_revised_policy",
            "precision",
            "specificity",
            "fp",
        ],
        ascending=[True, True, False, False, True],
    ).copy()
    ranked["operational_rank"] = np.arange(1, len(ranked) + 1)
    table = table.drop(columns=["operational_rank"]).merge(
        ranked[["system", "operational_rank"]],
        on="system",
        how="left",
    )
    output_cols = [
        "dataset",
        "system",
        "model_family",
        "policy_type",
        "threshold_or_band",
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
        "expected_cost_revised_policy",
        "manual_review_rate",
        "auto_decision_rate",
        "net_benefit_useful_range",
        "eligible_basic",
        "eligible_strict",
        "fp_profile_comment",
        "operational_rank",
        "final_comment",
        "manual_review_practical",
        "automatic_decision_suitable",
        "ranking_bucket",
    ]
    return table[output_cols].sort_values("operational_rank").reset_index(drop=True)


def best_net_benefit(dataset: str) -> str:
    """Return model with highest mean net benefit across useful ranges."""

    ranges = read_csv("net_benefit_model_ranges.csv")
    rows = ranges.loc[ranges["dataset"].eq(dataset) & ranges["range_type"].eq("better_than_both")].copy()
    if rows.empty:
        return "not available"
    model_scores = rows.groupby("model", as_index=False)["mean_net_benefit"].mean()
    winner = model_scores.sort_values("mean_net_benefit", ascending=False).iloc[0]
    return str(winner["model"])


def recommendation_rows(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> pd.DataFrame:
    """Build winner-type recommendation table."""

    rows: list[dict[str, Any]] = []
    for dataset, table in [("taiwan", taiwan), ("heloc", heloc)]:
        manual = table.loc[
            table["manual_review_rate"].notna() & table["manual_review_practical"] & table["eligible_strict"]
        ].sort_values(["expected_cost_revised_policy", "precision"], ascending=[True, False])
        binary = table.loc[table["manual_review_rate"].isna() & table["eligible_basic"]].copy()
        interpretable = table.loc[table["system"].eq("Best Scorecard policy")].iloc[0]
        cost_min = table.sort_values("expected_cost_revised_policy").iloc[0]
        precision = table.loc[table["eligible_basic"]].sort_values(["precision", "specificity"], ascending=[False, False]).iloc[0]
        recall = table.loc[table["eligible_basic"]].sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
        manual_winner = manual.iloc[0] if not manual.empty else table.loc[table["manual_review_rate"].notna()].sort_values("expected_cost_revised_policy").iloc[0]
        scre = table.loc[table["system"].eq("Best SCRE policy")].iloc[0]
        final_policy = manual_winner
        if dataset == "taiwan":
            final_comment = (
                "Final operational screening policy should be manual-review based. "
                "Avoid automatic rejection; binary fallback is CatBoost precision-constrained or eligible SCRE/LightGBM."
            )
        else:
            final_comment = (
                "External-validation support favors SCRE-Pareto/manual-review, with Scorecard retained as interpretable benchmark."
            )
        winner_specs = [
            ("Best cost-minimizer", cost_min, "Lowest revised policy cost in final comparison table."),
            ("Best precision-constrained system", precision, "Highest precision among eligible final systems."),
            ("Best recall-preserving system", recall, "Highest recall among eligible final systems."),
            ("Best manual-review policy", manual_winner, "Best practical manual-review policy with review rate <= 0.30 where available."),
            ("Best interpretable system", interpretable, "Scorecard retained as interpretable benchmark."),
            ("Best net-benefit system", best_net_benefit(dataset), "Winner by decision-curve mean net benefit across useful ranges."),
            ("Best external validation system", heloc_external_policy() if dataset == "heloc" else "See HELOC rows", "External validation is assessed on HELOC."),
            ("Final recommended operational policy", final_policy, final_comment),
            ("SCRE role", scre, "SCRE retained as reliability-aware framework rather than guaranteed strongest single predictor."),
        ]
        for winner_type, item, comment in winner_specs:
            if isinstance(item, str):
                rows.append(
                    {
                        "dataset": dataset,
                        "winner_type": winner_type,
                        "system": item,
                        "model_family": "",
                        "policy_type": "",
                        "threshold_or_band": "",
                        "precision": np.nan,
                        "recall": np.nan,
                        "specificity": np.nan,
                        "fp": np.nan,
                        "fn": np.nan,
                        "expected_cost_revised_policy": np.nan,
                        "manual_review_rate": np.nan,
                        "comment": comment,
                    }
                )
            else:
                row = dict(item)
                rows.append(
                    {
                        "dataset": dataset,
                        "winner_type": winner_type,
                        "system": row["system"],
                        "model_family": row["model_family"],
                        "policy_type": row["policy_type"],
                        "threshold_or_band": row["threshold_or_band"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "specificity": row["specificity"],
                        "fp": row["fp"],
                        "fn": row["fn"],
                        "expected_cost_revised_policy": row["expected_cost_revised_policy"],
                        "manual_review_rate": row["manual_review_rate"],
                        "comment": comment,
                    }
                )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    """Format numeric values for markdown."""

    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def write_summary(taiwan: pd.DataFrame, heloc: pd.DataFrame, recommendation: pd.DataFrame) -> Path:
    """Write final revised decision summary markdown."""

    taiwan_final = recommendation.loc[
        recommendation["dataset"].eq("taiwan")
        & recommendation["winner_type"].eq("Final recommended operational policy")
    ].iloc[0]
    heloc_final = recommendation.loc[
        recommendation["dataset"].eq("heloc")
        & recommendation["winner_type"].eq("Final recommended operational policy")
    ].iloc[0]
    old = taiwan.loc[taiwan["system"].eq("Old CatBoost cost threshold")].iloc[0]
    precision = taiwan.loc[taiwan["system"].eq("CatBoost precision-constrained threshold")].iloc[0]
    manual = taiwan.loc[taiwan["system"].eq("CatBoost manual review band")].iloc[0]
    imbalance = taiwan.loc[taiwan["system"].eq("Best training-level imbalance variant")].iloc[0]

    lines = [
        "# Final Revised Decision Summary",
        "",
        "## 1. What was wrong with the old cost-threshold model?",
        "",
        (
            f"Taiwan old CatBoost cost-threshold had precision={old['precision']:.4f}, "
            f"recall={old['recall']:.4f}, specificity={old['specificity']:.4f}, FP={int(old['fp'])}. "
            "It captured defaults, but generated too many false positives for automatic rejection."
        ),
        "",
        "## 2. Did precision-constrained thresholding help?",
        "",
        (
            f"Yes. CatBoost precision-constrained threshold improved precision to {precision['precision']:.4f} "
            f"and reduced FP to {int(precision['fp'])}, with recall={precision['recall']:.4f}."
        ),
        "",
        "## 3. Did cost sensitivity change the winner?",
        "",
        "Yes. Increasing FP cost moves thresholds upward, improves precision/specificity, and changes the winner in several scenarios. FN=5, FP=1 is too aggressive as the only operational assumption.",
        "",
        "## 4. Did manual review band improve operational usability?",
        "",
        (
            f"Yes, conditionally. CatBoost bounded manual-review band reduced high-risk FP to {int(manual['fp'])} "
            f"with precision={manual['precision']:.4f} and manual_review_rate={manual['manual_review_rate']:.4f}. "
            "Manual-review capacity is the key operational constraint."
        ),
        "",
        "## 5. Did class weighting help?",
        "",
        (
            f"Partially. Best Taiwan imbalance-training row reached precision={imbalance['precision']:.4f}, "
            f"recall={imbalance['recall']:.4f}, FP={int(imbalance['fp'])}. It is a useful binary alternative, "
            "but not enough to replace manual-review decision support."
        ),
        "",
        "## 6. Are false positives truly risky-looking customers?",
        "",
        "Mostly yes. FP profiles are closer to TP than TN across the analyzed systems, especially for delay and repayment behavior. This supports review/screening, not automatic rejection.",
        "",
        "## 7. What is the final Taiwan recommendation?",
        "",
        (
            f"Final Taiwan operational screening policy: {taiwan_final['system']} "
            f"({taiwan_final['policy_type']}, {taiwan_final['threshold_or_band']}). "
            "Use CatBoost precision-constrained or eligible binary SCRE/LightGBM only as fallback when manual review is unavailable."
        ),
        "",
        "## 8. What is the final HELOC recommendation?",
        "",
        (
            f"Final HELOC external-validation policy: {heloc_final['system']} "
            f"({heloc_final['policy_type']}, {heloc_final['threshold_or_band']}). "
            "Scorecard remains the interpretable benchmark."
        ),
        "",
        "## 9. Should the model be used for automatic decisioning?",
        "",
        "No. The final evidence does not justify fully automatic rejection. Precision and false-positive behavior require human review for high-risk cases.",
        "",
        "## 10. Should the model be used for screening/manual review?",
        "",
        "Yes. The revised policies are much stronger as screening and manual-review prioritization tools.",
        "",
        "## 11. What should be claimed?",
        "",
        "- Revised decision policies reduce false-positive burden and improve operational usability.",
        "- Manual-review framing is more defensible than automatic rejection.",
        "- SCRE remains useful as a reliability-aware framework.",
        "- Scorecard remains an interpretable benchmark.",
        "",
        "## 12. What should not be claimed?",
        "",
        "- Do not claim the model is suitable for fully automatic rejection.",
        "- Do not claim SCRE universally beats all single models.",
        "- Do not claim false positives are harmless; they are risk-like and need review.",
        "- Do not present FN=5, FP=1 as the only valid operational cost assumption.",
        "",
    ]
    path = DECISION_REVISION_DIR / "final_revised_decision_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame, recommendation: pd.DataFrame) -> list[Path]:
    """Write requested final revised comparison outputs."""

    paths = [
        DECISION_REVISION_DIR / "final_revised_comparison_taiwan.csv",
        DECISION_REVISION_DIR / "final_revised_comparison_heloc.csv",
        DECISION_REVISION_DIR / "final_revised_recommendation.csv",
    ]
    taiwan.to_csv(paths[0], index=False)
    heloc.to_csv(paths[1], index=False)
    recommendation.to_csv(paths[2], index=False)
    paths.append(write_summary(taiwan, heloc, recommendation))
    return paths


def main() -> None:
    """Build final revised comparison artifacts."""

    taiwan = comparison_table("taiwan")
    heloc = comparison_table("heloc")
    recommendation = recommendation_rows(taiwan, heloc)
    paths = write_outputs(taiwan, heloc, recommendation)
    for path in paths:
        print(path)

    for dataset, table in [("Taiwan", taiwan), ("HELOC", heloc)]:
        print(f"\n{dataset} final comparison:")
        print(
            table[
                [
                    "system",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                    "fp",
                    "fn",
                    "expected_cost_revised_policy",
                    "manual_review_rate",
                    "operational_rank",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
