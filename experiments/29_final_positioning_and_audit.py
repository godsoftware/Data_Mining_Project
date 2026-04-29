"""Write the final positioning statement and revision audit."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import EXPERIMENT_LOGS_DIR, OUTPUTS_DIR, TABLES_DIR
from src.utils.logging_utils import create_experiment_logger


LOGGER = create_experiment_logger(
    "29_final_positioning_and_audit",
    EXPERIMENT_LOGS_DIR / "29_final_positioning_and_audit.log",
)


REQUIRED_ARTIFACTS = [
    TABLES_DIR / "result_inventory_audit.csv",
    TABLES_DIR / "final_leaderboard_taiwan.csv",
    TABLES_DIR / "final_leaderboard_heloc.csv",
    TABLES_DIR / "pareto_front_taiwan.csv",
    TABLES_DIR / "pareto_front_heloc.csv",
    TABLES_DIR / "scre_pareto_results_taiwan.csv",
    TABLES_DIR / "scre_optimized_results_taiwan.csv",
    TABLES_DIR / "scre_revision_global_comparison.csv",
    TABLES_DIR / "kendalls_w_stability_taiwan.csv",
    TABLES_DIR / "statistical_tests_cleaned.csv",
    TABLES_DIR / "manual_review_band_revision_taiwan.csv",
    TABLES_DIR / "final_decision_matrix.csv",
    TABLES_DIR / "claim_control_matrix.csv",
    OUTPUTS_DIR / "final_positioning_statement.md",
]


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV artifact."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt(value: float) -> str:
    """Format numeric evidence for final audit text."""

    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def model_row(table: pd.DataFrame, model: str) -> pd.Series:
    """Return one model row from a comparison table."""

    rows = table.loc[table["model"].eq(model)]
    if rows.empty:
        raise ValueError(f"Missing model row: {model}")
    return rows.iloc[0]


def artifact_status(path: Path) -> dict[str, str]:
    """Return existence, size, and shape information for one artifact."""

    exists = path.exists()
    size = path.stat().st_size if exists else 0
    shape = ""
    if exists and path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            shape = f"{df.shape[0]} rows x {df.shape[1]} cols"
        except Exception as exc:  # pragma: no cover - defensive audit detail
            shape = f"unreadable csv: {exc}"
    return {
        "artifact": str(path.relative_to(PROJECT_ROOT)),
        "exists": "YES" if exists else "NO",
        "non_empty": "YES" if size > 0 else "NO",
        "shape": shape,
    }


def write_text(path: Path, content: str) -> Path:
    """Write UTF-8 text with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def build_positioning_statement() -> str:
    """Build the short academic positioning statement."""

    return """# Final Project Positioning

Safe final positioning:

This project does not claim to invent a new fundamental machine-learning algorithm. Instead, it proposes and evaluates SCRE-Credit as a reliability-aware credit-risk modeling framework that combines calibrated prediction, cost-sensitive decisioning, manual-review policy design, external validation, and explanation reliability checks.

The revised SCRE-Credit variants, SCRE-Pareto and SCRE-Optimized, are evaluated against strong single-model baselines including CatBoost, XGBoost, LightGBM, monotonic boosting, and WOE/Scorecard logistic models. The final evidence does not support a blanket claim that SCRE-Credit outperforms all individual models. On Taiwan, CatBoost is the strongest observed operational model under the FN=5, FP=1 cost scenario, while SCRE-Optimized achieves the strongest PR-AUC among the final comparison set. On HELOC external validation, Scorecard is the expected-cost winner, while SCRE variants remain competitive but do not dominate.

The final conclusion therefore distinguishes between operational deployment performance, interpretable benchmarking, calibration/reliability evidence, external validation behavior, and the proposed framework contribution. SCRE-Credit should be positioned as a reliability-aware framework for structured credit-risk experimentation and decision analysis, not as a universally superior standalone predictor.
"""


def build_revision_audit() -> str:
    """Build the final revision audit markdown."""

    taiwan = read_csv(TABLES_DIR / "scre_revision_comparison_taiwan.csv")
    heloc = read_csv(TABLES_DIR / "scre_revision_comparison_heloc.csv")
    final_matrix = read_csv(TABLES_DIR / "final_decision_matrix.csv")
    claims = read_csv(TABLES_DIR / "claim_control_matrix.csv")
    kendalls = read_csv(TABLES_DIR / "kendalls_w_stability_taiwan.csv")
    stats = read_csv(TABLES_DIR / "statistical_tests_cleaned.csv")
    manual = read_csv(TABLES_DIR / "manual_review_band_revision_taiwan.csv")

    taiwan_cat = model_row(taiwan, "Best CatBoost")
    taiwan_scre_opt = model_row(taiwan, "SCRE-Optimized")
    taiwan_scre_pareto = model_row(taiwan, "SCRE-Pareto")
    heloc_scorecard = model_row(heloc, "Best Scorecard")
    heloc_scre_opt = model_row(heloc, "SCRE-Optimized")
    heloc_scre_pareto = model_row(heloc, "SCRE-Pareto")
    heloc_old_scre = model_row(heloc, "Old SCRE-Credit")

    revised_successfully = all(
        [
            (TABLES_DIR / "scre_pareto_results_taiwan.csv").exists(),
            (TABLES_DIR / "scre_optimized_results_taiwan.csv").exists(),
            (TABLES_DIR / "scre_revision_global_comparison.csv").exists(),
            "SCRE-Optimized" in set(taiwan["model"]),
            "SCRE-Optimized" in set(heloc["model"]),
        ]
    )
    scre_beats_cat_cost = min(
        taiwan_scre_opt["expected_cost"],
        taiwan_scre_pareto["expected_cost"],
    ) < taiwan_cat["expected_cost"]
    scre_beats_cat_pr_auc = max(
        taiwan_scre_opt["pr_auc"],
        taiwan_scre_pareto["pr_auc"],
    ) > taiwan_cat["pr_auc"]
    scre_beats_scorecard_heloc_cost = min(
        heloc_scre_opt["expected_cost"],
        heloc_scre_pareto["expected_cost"],
        heloc_old_scre["expected_cost"],
    ) < heloc_scorecard["expected_cost"]

    kendalls_added = bool(
        (
            kendalls["feature_subset"].eq("top_5_features")
            & kendalls["kendalls_w"].notna()
        ).any()
    )
    required_stat_cols = {
        "dataset",
        "test_name",
        "metric",
        "baseline_model",
        "candidate_model",
        "baseline_value",
        "candidate_value",
        "difference_candidate_minus_baseline",
        "p_value",
        "ci_lower",
        "ci_upper",
        "significant_0_05",
        "better_direction",
        "winner",
        "interpretation",
    }
    stats_cleaned = required_stat_cols.issubset(stats.columns) and not stats.empty
    manual_revised = {
        "low_risk_threshold",
        "high_risk_threshold",
        "selection_split",
        "remaining_expected_cost",
        "cost_reduction_vs_threshold_only",
    }.issubset(manual.columns) and manual["selection_split"].eq("validation").all()
    final_recommendation_clear = (
        not final_matrix.loc[final_matrix["category"].eq("Final recommended deployment model")].empty
        and final_matrix["ready_for_report"].astype(bool).all()
    )
    safe_claims_ready = (
        set(claims["claim"]).issuperset(
            {
                "SCRE-Credit outperforms all individual models.",
                "SCRE-Credit is the best model on Taiwan.",
                "SCRE-Credit is the best model on HELOC.",
            }
        )
        and claims.loc[
            claims["claim"].isin(
                [
                    "SCRE-Credit outperforms all individual models.",
                    "SCRE-Credit is the best model on Taiwan.",
                    "SCRE-Credit is the best model on HELOC.",
                ]
            ),
            "final_status",
        ].eq("DO NOT CLAIM").all()
    )
    proceed_to_report = all(
        [
            revised_successfully,
            kendalls_added,
            stats_cleaned,
            manual_revised,
            final_recommendation_clear,
            safe_claims_ready,
        ]
    )

    decisions = [
        ("SCRE revised successfully", revised_successfully, "SCRE-Pareto and SCRE-Optimized outputs exist and are included in global comparison."),
        (
            "SCRE beats CatBoost on Taiwan operational cost",
            scre_beats_cat_cost,
            f"CatBoost cost={fmt(taiwan_cat['expected_cost'])}; SCRE-Pareto cost={fmt(taiwan_scre_pareto['expected_cost'])}; SCRE-Optimized cost={fmt(taiwan_scre_opt['expected_cost'])}.",
        ),
        (
            "SCRE beats CatBoost on Taiwan PR-AUC",
            scre_beats_cat_pr_auc,
            f"CatBoost PR-AUC={fmt(taiwan_cat['pr_auc'])}; SCRE-Pareto PR-AUC={fmt(taiwan_scre_pareto['pr_auc'])}; SCRE-Optimized PR-AUC={fmt(taiwan_scre_opt['pr_auc'])}.",
        ),
        (
            "SCRE beats Scorecard on HELOC cost",
            scre_beats_scorecard_heloc_cost,
            f"Scorecard cost={fmt(heloc_scorecard['expected_cost'])}; Old SCRE cost={fmt(heloc_old_scre['expected_cost'])}; SCRE-Pareto cost={fmt(heloc_scre_pareto['expected_cost'])}; SCRE-Optimized cost={fmt(heloc_scre_opt['expected_cost'])}.",
        ),
        ("Kendall's W added", kendalls_added, "Kendall's W top-5 rows are present in the Taiwan stability table."),
        ("Statistical tests cleaned", stats_cleaned, f"Clean statistical test table has {len(stats)} rows and required winner/direction columns."),
        ("Manual review band revised", manual_revised, "Manual-review bands use validation-selected thresholds and include remaining-cost fields."),
        ("Final model recommendation clear", final_recommendation_clear, "Final decision matrix includes an objective-specific deployment recommendation."),
        ("Safe claims ready", safe_claims_ready, "Claim-control matrix marks the three unsupported SCRE dominance claims as DO NOT CLAIM."),
        ("Proceed to final report", proceed_to_report, "Proceed only with scoped, non-overclaiming language."),
    ]

    artifact_rows = [artifact_status(path) for path in REQUIRED_ARTIFACTS]
    artifact_table = pd.DataFrame(artifact_rows)

    lines = [
        "# Final Revision Audit",
        "",
        "## Required Artifact Check",
        "",
        "| Artifact | Exists | Non-empty | Shape |",
        "|---|---:|---:|---|",
    ]
    for _, row in artifact_table.iterrows():
        lines.append(f"| `{row['artifact']}` | {row['exists']} | {row['non_empty']} | {row['shape']} |")

    lines.extend(
        [
            "",
            "## Final Decisions",
            "",
            "| Check | Decision | Evidence |",
            "|---|---:|---|",
        ]
    )
    for check, passed, evidence in decisions:
        lines.append(f"| {check} | {'YES' if passed else 'NO'} | {evidence} |")

    lines.extend(
        [
            "",
            "## Bottom Line",
            "",
            "SCRE-Credit was revised successfully as SCRE-Pareto and SCRE-Optimized, and the revised framework is now positioned more honestly. It should not be claimed as the best operational model on every dataset.",
            "",
            "The final operational recommendation is CatBoost for Taiwan under the FN=5, FP=1 expected-cost objective, and Scorecard for HELOC expected cost. SCRE-Optimized should be presented as the proposed reliability-aware framework and PR-AUC-oriented SCRE variant, not as a universal cost winner.",
            "",
            f"Proceed to final report: {'YES' if proceed_to_report else 'NO'}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Write final positioning and audit markdown files."""

    positioning_path = write_text(OUTPUTS_DIR / "final_positioning_statement.md", build_positioning_statement())
    LOGGER.info("Wrote %s", positioning_path)
    audit_path = write_text(OUTPUTS_DIR / "final_revision_audit.md", build_revision_audit())
    LOGGER.info("Wrote %s", audit_path)
    print(f"Wrote {positioning_path}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
