"""Create final model decision matrix and recommendations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import EXPERIMENT_LOGS_DIR, OUTPUTS_DIR, TABLES_DIR
from src.utils.logging_utils import create_experiment_logger


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def dataset_table(dataset: str) -> pd.DataFrame:
    """Load the final comparison table for one dataset."""

    return read_csv(TABLES_DIR / f"scre_revision_comparison_{dataset}.csv")


def row_for_model(table: pd.DataFrame, model: str) -> pd.Series:
    """Return one model row from a comparison table."""

    rows = table.loc[table["model"] == model]
    if rows.empty:
        raise ValueError(f"Model not found in comparison table: {model}")
    return rows.iloc[0]


def best_raw_predictive(table: pd.DataFrame) -> pd.Series:
    """Select best raw predictive model by PR-AUC, ROC-AUC, then F1."""

    return table.sort_values(["pr_auc", "roc_auc", "f1"], ascending=[False, False, False]).iloc[0]


def best_operational(table: pd.DataFrame) -> pd.Series:
    """Select best operational model by cost, recall, then precision."""

    return table.sort_values(["expected_cost", "recall", "precision"], ascending=[True, False, False]).iloc[0]


def best_calibrated(table: pd.DataFrame) -> pd.Series:
    """Select best calibrated model by ECE, Brier, then ROC-AUC."""

    return table.sort_values(["ece", "brier", "roc_auc"], ascending=[True, True, False]).iloc[0]


def best_interpretable(table: pd.DataFrame) -> pd.Series:
    """Select the most interpretable model using the project hierarchy."""

    order = ["Best Scorecard", "Best Monotonic model", "SCRE-Pareto", "SCRE-Optimized", "Old SCRE-Credit"]
    for model in order:
        match = table.loc[table["model"] == model]
        if not match.empty:
            return match.iloc[0]
    return table.sort_values("interpretability_rank").iloc[0]


def best_framework(table: pd.DataFrame) -> pd.Series:
    """Select the best SCRE-family proposed framework variant."""

    candidates = table.loc[table["model"].isin(["Old SCRE-Credit", "SCRE-Pareto", "SCRE-Optimized"])].copy()
    return candidates.sort_values(["pr_auc", "expected_cost", "ece"], ascending=[False, True, True]).iloc[0]


def lightgbm_stability_row(dataset: str) -> pd.Series:
    """Return the computed top-5 Kendall's W row for LightGBM stability."""

    table = read_csv(TABLES_DIR / f"kendalls_w_stability_{dataset}.csv")
    match = table.loc[
        (table["analysis_target"] == "Available LightGBM SHAP stability protocol")
        & (table["feature_subset"] == "top_5_features")
    ]
    if match.empty:
        raise ValueError(f"No top-5 Kendall's W row for {dataset}.")
    return match.iloc[0]


def manual_review_winner(dataset: str) -> pd.Series:
    """Return the test-set manual-review winner by remaining cost."""

    table = read_csv(TABLES_DIR / f"manual_review_band_revision_{dataset}.csv")
    test = table.loc[table["split"] == "test"].copy()
    return test.sort_values(["remaining_expected_cost", "manual_review_rate", "auto_decision_rate"], ascending=[True, True, False]).iloc[0]


def statistical_summary(dataset: str, baseline: str, candidate: str) -> str:
    """Return the statistical-test summary for a comparison when available."""

    table = read_csv(TABLES_DIR / "statistical_tests_summary.csv")
    match = table.loc[
        (table["dataset"] == dataset)
        & (table["baseline_model"] == baseline)
        & (table["candidate_model"] == candidate)
    ]
    if match.empty:
        return "No direct statistical-test summary row."
    row = match.iloc[0]
    return (
        f"Expected-cost winner={row['expected_cost_winner']}; "
        f"diff candidate-baseline={row['expected_cost_difference_candidate_minus_baseline']:.1f}; "
        f"p={row['expected_cost_p_value']:.3g}."
    )


def evidence_string(row: pd.Series) -> str:
    """Build compact metric evidence for decision rows."""

    return (
        f"PR-AUC={row['pr_auc']:.4f}; ROC-AUC={row['roc_auc']:.4f}; F1={row['f1']:.4f}; "
        f"Recall={row['recall']:.4f}; Precision={row['precision']:.4f}; "
        f"ECE={row['ece']:.4f}; Brier={row['brier']:.4f}; Cost={row['expected_cost']:.0f}."
    )


def decision_row(
    category_id: int,
    category: str,
    dataset_scope: str,
    winner: str,
    primary_basis: str,
    evidence: str,
    recommendation: str,
    caveat: str,
) -> dict[str, object]:
    """Create one final decision-matrix row."""

    return {
        "category_id": category_id,
        "category": category,
        "dataset_scope": dataset_scope,
        "winner": winner,
        "primary_basis": primary_basis,
        "evidence": evidence,
        "recommendation": recommendation,
        "caveat": caveat,
        "ready_for_report": True,
    }


def build_decision_matrix() -> pd.DataFrame:
    """Build the final decision matrix."""

    taiwan = dataset_table("taiwan")
    heloc = dataset_table("heloc")
    taiwan_raw = best_raw_predictive(taiwan)
    heloc_raw = best_raw_predictive(heloc)
    taiwan_operational = best_operational(taiwan)
    heloc_operational = best_operational(heloc)
    taiwan_calibrated = best_calibrated(taiwan)
    heloc_calibrated = best_calibrated(heloc)
    taiwan_interpretable = best_interpretable(taiwan)
    heloc_interpretable = best_interpretable(heloc)
    taiwan_framework = best_framework(taiwan)
    heloc_framework = best_framework(heloc)
    taiwan_stability = lightgbm_stability_row("taiwan")
    heloc_stability = lightgbm_stability_row("heloc")
    taiwan_review = manual_review_winner("taiwan")
    heloc_review = manual_review_winner("heloc")

    rows = [
        decision_row(
            1,
            "Best raw predictive model",
            "Taiwan",
            str(taiwan_raw["model"]),
            "PR-AUC, then ROC-AUC, then F1",
            evidence_string(taiwan_raw),
            "Use as Taiwan discrimination winner, not necessarily lowest-cost deployment winner.",
            "SCRE-Optimized wins Taiwan PR-AUC but does not win expected cost.",
        ),
        decision_row(
            1,
            "Best raw predictive model",
            "HELOC",
            str(heloc_raw["model"]),
            "PR-AUC, then ROC-AUC, then F1",
            evidence_string(heloc_raw),
            "Use as HELOC discrimination winner.",
            "HELOC winner differs from cost winner; this is a domain-shift/tradeoff result.",
        ),
        decision_row(
            2,
            "Best operational model",
            "Taiwan",
            str(taiwan_operational["model"]),
            "Lowest expected_cost, then recall, then acceptable precision",
            evidence_string(taiwan_operational)
            + " Manual-review remaining cost="
            + f"{taiwan_review['remaining_expected_cost']:.0f} for {taiwan_review['model']}.",
            "Choose Best CatBoost for Taiwan operational decisioning under FN=5, FP=1.",
            statistical_summary("taiwan", "Best CatBoost", "SCRE-Optimized"),
        ),
        decision_row(
            2,
            "Best operational model",
            "HELOC",
            str(heloc_operational["model"]),
            "Lowest expected_cost, then recall, then acceptable precision",
            evidence_string(heloc_operational)
            + " Manual-review remaining cost="
            + f"{heloc_review['remaining_expected_cost']:.0f} for {heloc_review['model']}.",
            "Choose Best Scorecard as HELOC cost winner; CatBoost is close and strongest in manual-review remaining cost.",
            "HELOC operational and manual-review winners are not identical.",
        ),
        decision_row(
            3,
            "Best calibrated model",
            "Taiwan",
            str(taiwan_calibrated["model"]),
            "Lowest ECE, then lowest Brier, then acceptable AUC",
            evidence_string(taiwan_calibrated),
            "Use CatBoost as Taiwan calibrated operational model.",
            "Calibration is reliability evidence, not proof of superior discrimination.",
        ),
        decision_row(
            3,
            "Best calibrated model",
            "HELOC",
            str(heloc_calibrated["model"]),
            "Lowest ECE, then lowest Brier, then acceptable AUC",
            evidence_string(heloc_calibrated),
            "Use CatBoost as HELOC calibration winner.",
            "CatBoost calibration is strong even though Scorecard has lower HELOC cost.",
        ),
        decision_row(
            4,
            "Best interpretable model",
            "Both",
            "Best Scorecard",
            "Scorecard first, monotonic model second, explanation simplicity",
            f"Taiwan Scorecard cost={row_for_model(taiwan, 'Best Scorecard')['expected_cost']:.0f}; "
            f"HELOC Scorecard cost={row_for_model(heloc, 'Best Scorecard')['expected_cost']:.0f}.",
            "Use Scorecard as the interpretable benchmark and external-validation anchor.",
            "Scorecard is not Taiwan predictive/cost winner.",
        ),
        decision_row(
            5,
            "Best XAI stability model",
            "Computed stability protocol",
            "LightGBM SHAP stability protocol",
            "Kendall's W top-5, top-k overlap, Spearman rank stability",
            f"Taiwan top-5 W={taiwan_stability['kendalls_w']:.4f}; "
            f"HELOC top-5 W={heloc_stability['kendalls_w']:.4f}; "
            "direct CatBoost/Scorecard/SCRE ensemble W not computed.",
            "Report LightGBM as the directly measured explanation-stability protocol.",
            "Do not claim CatBoost or SCRE ensemble-specific SHAP stability from LightGBM evidence.",
        ),
        decision_row(
            6,
            "Best external validation model",
            "HELOC",
            str(heloc_operational["model"]),
            "HELOC performance, HELOC cost, HELOC calibration",
            evidence_string(heloc_operational)
            + f" CatBoost calibration ECE={row_for_model(heloc, 'Best CatBoost')['ece']:.4f}.",
            "Use Scorecard as the most robust external-validation cost winner; discuss CatBoost/SCRE as close alternatives.",
            "HELOC result does not replicate Taiwan CatBoost cost dominance exactly.",
        ),
        decision_row(
            7,
            "Best proposed framework",
            "SCRE family",
            "SCRE-Optimized",
            "Among Old SCRE, SCRE-Pareto, SCRE-Optimized: PR-AUC, then expected cost, then calibration",
            f"Taiwan SCRE-Optimized PR-AUC={taiwan_framework['pr_auc']:.4f}, cost={taiwan_framework['expected_cost']:.0f}; "
            f"HELOC SCRE-Optimized PR-AUC={heloc_framework['pr_auc']:.4f}, cost={heloc_framework['expected_cost']:.0f}.",
            "Present SCRE-Optimized as the revised reliability-aware framework.",
            "It is not the best standalone cost predictor on Taiwan or HELOC.",
        ),
        decision_row(
            8,
            "Final recommended deployment model",
            "Project-level",
            "Operational: CatBoost on Taiwan; Scorecard on HELOC. Benchmark: Scorecard. Framework: SCRE-Optimized.",
            "Objective-specific deployment recommendation rather than one universal winner",
            "Taiwan cost winner=Best CatBoost; HELOC cost winner=Best Scorecard; SCRE-Optimized wins PR-AUC among SCRE variants and is useful as research framework.",
            "Deploy operational model by use case; present SCRE-Optimized as proposed framework, with Scorecard as interpretable benchmark.",
            "Do not claim one model dominates all datasets and metrics.",
        ),
    ]
    return pd.DataFrame(rows)


def build_recommendations(matrix: pd.DataFrame) -> pd.DataFrame:
    """Build a compact final recommendations table."""

    rows = [
        {
            "recommendation_area": "Operational decision",
            "recommended_model": "Best CatBoost for Taiwan; Best Scorecard for HELOC external validation",
            "supporting_evidence": "Lowest final-test expected cost in each dataset-specific comparison.",
            "report_language": "Model choice is objective-specific and cost-sensitive.",
        },
        {
            "recommendation_area": "Interpretable benchmark",
            "recommended_model": "Best Scorecard",
            "supporting_evidence": "Most transparent model family and HELOC operational winner.",
            "report_language": "Use as benchmark and governance-friendly comparator, not as Taiwan winner.",
        },
        {
            "recommendation_area": "Research framework",
            "recommended_model": "SCRE-Optimized",
            "supporting_evidence": "Best revised SCRE family variant by PR-AUC and reliability-aware construction.",
            "report_language": "SCRE is a reliability-aware framework, not a universal best predictor.",
        },
        {
            "recommendation_area": "XAI stability claim",
            "recommended_model": "LightGBM SHAP stability protocol",
            "supporting_evidence": "Computed Kendall's W top-5: Taiwan 0.8928, HELOC 1.0000.",
            "report_language": "Restrict direct stability claims to the protocol actually computed.",
        },
        {
            "recommendation_area": "Manual review",
            "recommended_model": "Best CatBoost",
            "supporting_evidence": "Lowest remaining expected cost under revised manual-review band on both datasets.",
            "report_language": "Manual review improves decision support but requires operational capacity assumptions.",
        },
        {
            "recommendation_area": "Claim boundary",
            "recommended_model": "No universal winner",
            "supporting_evidence": "Taiwan, HELOC, cost, calibration, PR-AUC, and interpretability winners differ.",
            "report_language": "Report winners by objective; avoid dominance claims.",
        },
    ]
    result = pd.DataFrame(rows)
    result["ready_for_report_writing"] = True
    return result


def write_summary(matrix: pd.DataFrame, recommendations: pd.DataFrame) -> Path:
    """Write the final decision Markdown summary."""

    taiwan = dataset_table("taiwan")
    heloc = dataset_table("heloc")
    taiwan_operational = best_operational(taiwan)
    heloc_operational = best_operational(heloc)
    taiwan_raw = best_raw_predictive(taiwan)
    heloc_raw = best_raw_predictive(heloc)
    scre_taiwan = best_framework(taiwan)
    scre_heloc = best_framework(heloc)
    stats = read_csv(TABLES_DIR / "statistical_tests_summary.csv")

    cat_vs_old = stats.loc[
        (stats["dataset"] == "taiwan")
        & (stats["baseline_model"] == "Best CatBoost")
        & (stats["candidate_model"] == "Old SCRE-Credit")
    ].iloc[0]
    cat_vs_opt = stats.loc[
        (stats["dataset"] == "taiwan")
        & (stats["baseline_model"] == "Best CatBoost")
        & (stats["candidate_model"] == "SCRE-Optimized")
    ].iloc[0]

    content = f"""# Final Decision Summary

## 1. What wins on Taiwan?

Best operational model on Taiwan is **{taiwan_operational['model']}** with expected cost {taiwan_operational['expected_cost']:.0f}, PR-AUC {taiwan_operational['pr_auc']:.4f}, ROC-AUC {taiwan_operational['roc_auc']:.4f}, recall {taiwan_operational['recall']:.4f}, and ECE {taiwan_operational['ece']:.4f}.

For raw predictive ranking, **{taiwan_raw['model']}** has the highest PR-AUC ({taiwan_raw['pr_auc']:.4f}), but it does not beat CatBoost on operational expected cost.

## 2. What wins on HELOC?

Best operational model on HELOC is **{heloc_operational['model']}** with expected cost {heloc_operational['expected_cost']:.0f}. The raw PR-AUC winner is **{heloc_raw['model']}** with PR-AUC {heloc_raw['pr_auc']:.4f}. This difference should be treated as a domain-shift and objective-tradeoff finding, not as an error.

## 3. Did SCRE-Credit improve after revision?

Yes, methodologically. Old SCRE was revised into SCRE-Pareto and SCRE-Optimized. SCRE-Optimized improved the SCRE-family discrimination profile: Taiwan SCRE-Optimized PR-AUC is {scre_taiwan['pr_auc']:.4f}, and HELOC SCRE-Optimized PR-AUC is {scre_heloc['pr_auc']:.4f}.

However, revision did not make SCRE the best operational cost model. On Taiwan, CatBoost vs SCRE-Optimized expected-cost difference is {cat_vs_opt['expected_cost_difference_candidate_minus_baseline']:.0f} against SCRE-Optimized, p={cat_vs_opt['expected_cost_p_value']:.3g}.

## 4. Is SCRE-Credit the best predictor?

No. It is not the best standalone predictor under the main operational cost criterion. Taiwan CatBoost is stronger for deployment-style cost, and HELOC Scorecard is the external-validation cost winner.

Old SCRE is significantly worse than CatBoost on Taiwan expected cost in the paired bootstrap summary: difference {cat_vs_old['expected_cost_difference_candidate_minus_baseline']:.0f}, p={cat_vs_old['expected_cost_p_value']:.3g}.

## 5. Is SCRE-Credit still useful as a framework?

Yes. SCRE-Optimized is useful as a reliability-aware research framework because it integrates calibrated probabilities, validation-only objective weighting, Pareto model selection, cost-sensitive thresholding, and explicit guardrails against test leakage.

Its correct positioning is: **a reliability-aware ensemble framework**, not **the universally best classifier**.

## 6. What should be claimed?

- Taiwan operational winner: Best CatBoost.
- HELOC external-validation cost winner: Best Scorecard.
- Best interpretable benchmark: Scorecard.
- Best proposed SCRE-family framework: SCRE-Optimized.
- SCRE-Optimized can improve PR-AUC within the SCRE family and provides a structured reliability-aware framework.
- Manual-review bands can reduce remaining auto-decision cost, but they require operational review capacity assumptions.

## 7. What should not be claimed?

- Do not claim SCRE-Credit beats CatBoost on Taiwan operational expected cost.
- Do not claim SCRE-Credit is the best predictor across all metrics.
- Do not claim LightGBM SHAP stability proves CatBoost, Scorecard, or SCRE ensemble stability.
- Do not claim HELOC results are identical to Taiwan results; they show domain shift.
- Do not claim statistical significance means the candidate model won; direction matters.

## 8. Ready for report writing: YES/NO

**YES**, with careful claim boundaries. The report should be written as an objective-specific model validation study: CatBoost for Taiwan operational cost, Scorecard as interpretable/external-validation benchmark, and SCRE-Optimized as the proposed reliability-aware framework.
"""
    path = OUTPUTS_DIR / "final_decision_summary.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> None:
    """Generate final decision outputs."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger("final_decision_matrix", EXPERIMENT_LOGS_DIR / "27_final_decision_matrix.log")

    matrix = build_decision_matrix()
    recommendations = build_recommendations(matrix)
    matrix_path = TABLES_DIR / "final_decision_matrix.csv"
    recommendations_path = TABLES_DIR / "final_model_recommendations.csv"
    summary_path = write_summary(matrix, recommendations)
    matrix.to_csv(matrix_path, index=False)
    recommendations.to_csv(recommendations_path, index=False)

    logger.info("Saved %s", matrix_path)
    logger.info("Saved %s", recommendations_path)
    logger.info("Saved %s", summary_path)
    print(matrix[["category", "dataset_scope", "winner", "recommendation", "caveat"]].to_string(index=False))


if __name__ == "__main__":
    main()
