"""Build the final claim-control matrix for report-safe wording.

The script does not train or tune models. It reads the current evidence
artifacts and marks each planned claim as supported, partially supported, or
unsafe to claim.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import EXPERIMENT_LOGS_DIR, PROJECT_ROOT as CONFIG_PROJECT_ROOT, TABLES_DIR
from src.utils.logging_utils import create_experiment_logger
from src.utils.save_utils import save_table


LOGGER = create_experiment_logger(
    "28_claim_control_matrix",
    EXPERIMENT_LOGS_DIR / "28_claim_control_matrix.log",
)


def read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV artifact and raise a clear error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Required evidence artifact is missing: {path}")
    return pd.read_csv(path)


def read_text(path: Path) -> str:
    """Read a text artifact if it exists, otherwise return an empty string."""

    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def as_float(value: Any) -> float:
    """Convert a scalar-like value to float with NaN for unavailable values."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric value for compact audit evidence."""

    number = as_float(value)
    if np.isnan(number):
        return "NA"
    if abs(number) >= 100:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def model_row(table: pd.DataFrame, model: str) -> pd.Series:
    """Return a model row from a comparison table."""

    rows = table.loc[table["model"].eq(model)]
    if rows.empty:
        raise ValueError(f"Model not found in comparison table: {model}")
    return rows.iloc[0]


def matrix_row(matrix: pd.DataFrame, category: str, dataset_scope: str | None = None) -> pd.Series:
    """Return a row from the final decision matrix."""

    rows = matrix.loc[matrix["category"].eq(category)]
    if dataset_scope is not None:
        rows = rows.loc[rows["dataset_scope"].eq(dataset_scope)]
    if rows.empty:
        raise ValueError(f"Decision matrix row not found: {category}, {dataset_scope}")
    return rows.iloc[0]


def best_manual_review_row(table: pd.DataFrame) -> pd.Series:
    """Return the top test-split manual-review row by remaining cost."""

    test_rows = table.loc[table["split"].eq("test")].copy()
    if test_rows.empty:
        raise ValueError("Manual-review table has no test split rows.")
    return test_rows.sort_values(["remaining_expected_cost", "captured_defaults_high_risk"], ascending=[True, False]).iloc[0]


def calibration_evidence(taiwan_cal: pd.DataFrame, heloc_cal: pd.DataFrame) -> str:
    """Summarize calibration reliability changes from validation artifacts."""

    parts: list[str] = []
    for dataset, table in [("Taiwan", taiwan_cal), ("HELOC", heloc_cal)]:
        validation = table.loc[table["split"].eq("validation")].copy()
        if validation.empty:
            continue
        grouped = validation.groupby("calibration_method", as_index=True).agg(
            roc_auc=("roc_auc", "mean"),
            pr_auc=("pr_auc", "mean"),
            brier=("brier_score", "mean"),
            ece=("ece", "mean"),
        )
        if "uncalibrated" not in grouped.index:
            continue
        best_ece_method = grouped["ece"].idxmin()
        best_brier_method = grouped["brier"].idxmin()
        parts.append(
            f"{dataset}: ECE uncal={fmt(grouped.loc['uncalibrated', 'ece'])} "
            f"vs best {best_ece_method}={fmt(grouped.loc[best_ece_method, 'ece'])}; "
            f"Brier uncal={fmt(grouped.loc['uncalibrated', 'brier'])} "
            f"vs best {best_brier_method}={fmt(grouped.loc[best_brier_method, 'brier'])}; "
            f"ROC-AUC uncal={fmt(grouped.loc['uncalibrated', 'roc_auc'])}"
        )
    return " | ".join(parts)


def kendalls_w_evidence(taiwan_w: pd.DataFrame, heloc_w: pd.DataFrame) -> str:
    """Summarize direct SHAP stability evidence from Kendall's W tables."""

    pieces: list[str] = []
    for dataset, table in [("Taiwan", taiwan_w), ("HELOC", heloc_w)]:
        rows = table.loc[
            table["direct_model_specific"].eq(True)
            & table["feature_subset"].eq("top_5_features")
            & table["kendalls_w"].notna()
        ]
        if rows.empty:
            continue
        row = rows.iloc[0]
        pieces.append(
            f"{dataset} {row['model']} top-5 Kendall's W={fmt(row['kendalls_w'])} "
            f"over {fmt(row['n_seeds'], digits=0)} seeds"
        )
    return "; ".join(pieces)


def background_evidence(background: pd.DataFrame) -> str:
    """Summarize background sensitivity drift evidence."""

    comparisons = background.loc[background["result_type"].eq("background_comparison")].copy()
    if comparisons.empty:
        return "No background-comparison rows found."
    return (
        f"top-k overlap range={fmt(comparisons['top_k_overlap_rate'].min())}-"
        f"{fmt(comparisons['top_k_overlap_rate'].max())}; "
        f"Spearman range={fmt(comparisons['spearman_rho'].min())}-"
        f"{fmt(comparisons['spearman_rho'].max())}; "
        f"max mean abs SHAP drift={fmt(comparisons['mean_abs_shap_drift'].max())}; "
        f"max relative drift={fmt(comparisons['mean_relative_shap_drift'].max())}"
    )


def faithfulness_evidence(faithfulness: pd.DataFrame) -> str:
    """Summarize faithfulness perturbation/deletion evidence."""

    topk = faithfulness.loc[faithfulness["test_type"].eq("topk_deletion")]
    if topk.empty:
        return "No top-k deletion rows found."
    interpretation = str(faithfulness["analysis_interpretation"].dropna().iloc[0])
    return (
        f"top-k deletion max PR-AUC drop={fmt(topk['pr_auc_drop'].max())}; "
        f"max ROC-AUC drop={fmt(topk['roc_auc_drop'].max())}; "
        f"max cost change={fmt(topk['cost_change'].max())}; "
        f"interpretation='{interpretation}'"
    )


def build_claim_control_matrix() -> pd.DataFrame:
    """Build the claim-control matrix from existing result artifacts."""

    final_matrix = read_required_csv(TABLES_DIR / "final_decision_matrix.csv")
    taiwan_revision = read_required_csv(TABLES_DIR / "scre_revision_comparison_taiwan.csv")
    heloc_revision = read_required_csv(TABLES_DIR / "scre_revision_comparison_heloc.csv")
    taiwan_review = read_required_csv(TABLES_DIR / "manual_review_band_revision_taiwan.csv")
    heloc_review = read_required_csv(TABLES_DIR / "manual_review_band_revision_heloc.csv")
    taiwan_w = read_required_csv(TABLES_DIR / "kendalls_w_stability_taiwan.csv")
    heloc_w = read_required_csv(TABLES_DIR / "kendalls_w_stability_heloc.csv")
    background = read_required_csv(TABLES_DIR / "background_sensitivity_results.csv")
    faithfulness = read_required_csv(TABLES_DIR / "faithfulness_results.csv")
    heloc_external = read_required_csv(TABLES_DIR / "external_validation_heloc_results.csv")
    taiwan_cal = read_required_csv(TABLES_DIR / "calibration_results_taiwan.csv")
    heloc_cal = read_required_csv(TABLES_DIR / "calibration_results_heloc.csv")

    project_scope = read_text(CONFIG_PROJECT_ROOT / "PROJECT_SCOPE.md")

    taiwan_cat = model_row(taiwan_revision, "Best CatBoost")
    taiwan_scre_opt = model_row(taiwan_revision, "SCRE-Optimized")
    taiwan_scre_pareto = model_row(taiwan_revision, "SCRE-Pareto")
    taiwan_old_scre = model_row(taiwan_revision, "Old SCRE-Credit")
    taiwan_scorecard = model_row(taiwan_revision, "Best Scorecard")

    heloc_scorecard = model_row(heloc_revision, "Best Scorecard")
    heloc_cat = model_row(heloc_revision, "Best CatBoost")
    heloc_scre_opt = model_row(heloc_revision, "SCRE-Optimized")
    heloc_scre_pareto = model_row(heloc_revision, "SCRE-Pareto")
    heloc_old_scre = model_row(heloc_revision, "Old SCRE-Credit")

    taiwan_review_best = best_manual_review_row(taiwan_review)
    heloc_review_best = best_manual_review_row(heloc_review)

    calibration_summary = calibration_evidence(taiwan_cal, heloc_cal)
    kendall_summary = kendalls_w_evidence(taiwan_w, heloc_w)
    background_summary = background_evidence(background)
    faithfulness_summary = faithfulness_evidence(faithfulness)

    external_validation_test = heloc_external.loc[heloc_external["split"].eq("test")].copy()
    external_models = external_validation_test["model"].nunique() if not external_validation_test.empty else 0

    rows: list[dict[str, str]] = [
        {
            "claim": "SCRE-Credit outperforms all individual models.",
            "supported": "NO",
            "evidence_file": "outputs/tables/final_decision_matrix.csv; outputs/tables/scre_revision_comparison_taiwan.csv; outputs/tables/scre_revision_comparison_heloc.csv",
            "evidence_metric": (
                f"Taiwan cost winner={taiwan_cat['model']} ({fmt(taiwan_cat['expected_cost'])}) "
                f"vs SCRE-Optimized ({fmt(taiwan_scre_opt['expected_cost'])}) and SCRE-Pareto ({fmt(taiwan_scre_pareto['expected_cost'])}); "
                f"HELOC cost winner={heloc_scorecard['model']} ({fmt(heloc_scorecard['expected_cost'])}) "
                f"vs SCRE-Optimized ({fmt(heloc_scre_opt['expected_cost'])})."
            ),
            "safe_wording": "SCRE-Optimized is a reliability-aware ensemble framework that improves some SCRE-family metrics, especially PR-AUC, but individual models remain stronger for some operational objectives.",
            "unsafe_wording": "SCRE-Credit outperforms all individual models.",
            "final_status": "DO NOT CLAIM",
        },
        {
            "claim": "SCRE-Credit is the best model on Taiwan.",
            "supported": "NO",
            "evidence_file": "outputs/tables/scre_revision_comparison_taiwan.csv; outputs/tables/final_decision_matrix.csv",
            "evidence_metric": (
                f"Taiwan operational winner={taiwan_cat['model']} cost={fmt(taiwan_cat['expected_cost'])}; "
                f"SCRE-Optimized cost={fmt(taiwan_scre_opt['expected_cost'])}, PR-AUC={fmt(taiwan_scre_opt['pr_auc'])}; "
                f"SCRE-Pareto cost={fmt(taiwan_scre_pareto['expected_cost'])}."
            ),
            "safe_wording": "On Taiwan, CatBoost is the observed operational winner under FN=5, FP=1; SCRE-Optimized is the strongest SCRE-family/raw PR-AUC framework candidate.",
            "unsafe_wording": "SCRE-Credit is the best model on Taiwan.",
            "final_status": "DO NOT CLAIM",
        },
        {
            "claim": "SCRE-Credit is the best model on HELOC.",
            "supported": "NO",
            "evidence_file": "outputs/tables/scre_revision_comparison_heloc.csv; outputs/tables/final_decision_matrix.csv",
            "evidence_metric": (
                f"HELOC operational winner={heloc_scorecard['model']} cost={fmt(heloc_scorecard['expected_cost'])}; "
                f"Best CatBoost cost={fmt(heloc_cat['expected_cost'])}; "
                f"SCRE-Optimized cost={fmt(heloc_scre_opt['expected_cost'])}, PR-AUC={fmt(heloc_scre_opt['pr_auc'])}."
            ),
            "safe_wording": "On HELOC, Scorecard is the observed cost winner, while SCRE-Optimized is competitive and wins PR-AUC in the final comparison.",
            "unsafe_wording": "SCRE-Credit is the best model on HELOC.",
            "final_status": "DO NOT CLAIM",
        },
        {
            "claim": "CatBoost is the strongest operational model on Taiwan.",
            "supported": "YES",
            "evidence_file": "outputs/tables/scre_revision_comparison_taiwan.csv; outputs/tables/manual_review_band_revision_taiwan.csv",
            "evidence_metric": (
                f"Threshold-only expected cost={fmt(taiwan_cat['expected_cost'])}, operational_rank={fmt(taiwan_cat['operational_rank'], 0)}; "
                f"manual-review remaining cost={fmt(taiwan_review_best['remaining_expected_cost'])} "
                f"for {taiwan_review_best['model']} with auto_decision_rate={fmt(taiwan_review_best['auto_decision_rate'])}."
            ),
            "safe_wording": "CatBoost is the strongest observed Taiwan operational model under the explicit FN=5, FP=1 cost scenario and the revised manual-review policy.",
            "unsafe_wording": "CatBoost is universally the best credit-risk model.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "Scorecard remains competitive and highly interpretable.",
            "supported": "YES",
            "evidence_file": "outputs/tables/final_decision_matrix.csv; outputs/tables/scre_revision_comparison_taiwan.csv; outputs/tables/scre_revision_comparison_heloc.csv",
            "evidence_metric": (
                f"Final interpretable winner=Best Scorecard; Taiwan Scorecard cost={fmt(taiwan_scorecard['expected_cost'])}; "
                f"HELOC Scorecard cost={fmt(heloc_scorecard['expected_cost'])}, operational_rank={fmt(heloc_scorecard['operational_rank'], 0)}."
            ),
            "safe_wording": "Scorecard is the most interpretable benchmark and remains competitive, especially as the HELOC expected-cost winner; it is not the Taiwan predictive or cost winner.",
            "unsafe_wording": "Scorecard is as strong as every black-box model on every metric.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "Calibration improves probability reliability but not necessarily AUC.",
            "supported": "YES",
            "evidence_file": "outputs/tables/calibration_results_taiwan.csv; outputs/tables/calibration_results_heloc.csv",
            "evidence_metric": calibration_summary,
            "safe_wording": "Calibration is reported as a probability-reliability step that improves ECE/Brier behavior in validation averages, not as an AUC-improvement method.",
            "unsafe_wording": "Calibration guarantees better discrimination or higher AUC.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "Cost-sensitive thresholding improves default capture under explicit cost assumptions.",
            "supported": "YES",
            "evidence_file": "outputs/tables/manual_review_band_revision_taiwan.csv; outputs/tables/manual_review_band_revision_heloc.csv; outputs/tables/scre_revision_comparison_taiwan.csv; outputs/tables/scre_revision_comparison_heloc.csv",
            "evidence_metric": (
                f"Taiwan CatBoost recall={fmt(taiwan_cat['recall'])}, cost={fmt(taiwan_cat['expected_cost'])}, "
                f"manual-review high-risk captured defaults={fmt(taiwan_review_best['captured_defaults_high_risk'])}, "
                f"cost reduction vs threshold-only={fmt(taiwan_review_best['cost_reduction_vs_threshold_only'])}; "
                f"HELOC best-review captured defaults={fmt(heloc_review_best['captured_defaults_high_risk'])}, "
                f"cost reduction vs threshold-only={fmt(heloc_review_best['cost_reduction_vs_threshold_only'])}."
            ),
            "safe_wording": "Cost-sensitive thresholding and manual-review bands improve decision support under the stated FN=5, FP=1 assumption; conclusions change if the cost ratio changes.",
            "unsafe_wording": "Thresholding always improves model quality independent of cost assumptions.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "SHAP explanations are stable across repeated seeds.",
            "supported": "PARTIAL",
            "evidence_file": "outputs/tables/kendalls_w_stability_taiwan.csv; outputs/tables/kendalls_w_stability_heloc.csv; outputs/tables/shap_stability_literature_comparison.csv",
            "evidence_metric": kendall_summary,
            "safe_wording": "The computed LightGBM SHAP stability protocol shows high top-k rank stability across repeated seeds; direct CatBoost, Scorecard, and SCRE ensemble-specific stability was not computed.",
            "unsafe_wording": "All SHAP explanations for all final models are stable.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "SHAP explanations are sensitive to intentionally shifted background distributions.",
            "supported": "YES",
            "evidence_file": "outputs/tables/background_sensitivity_results.csv",
            "evidence_metric": background_summary,
            "safe_wording": "Background-sensitivity analysis shows SHAP ranking and magnitude drift under intentionally shifted background sets.",
            "unsafe_wording": "SHAP explanations are invariant to background choice.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "Faithfulness analysis supports model reliance on top SHAP features but does not prove causality.",
            "supported": "YES",
            "evidence_file": "outputs/tables/faithfulness_results.csv; outputs/tables/faithfulness_group_results.csv",
            "evidence_metric": faithfulness_summary,
            "safe_wording": "Faithfulness perturbation/deletion results support a model-behavior sanity check around top-ranked SHAP features, but they are not causal evidence.",
            "unsafe_wording": "SHAP has proven the causal drivers of default risk.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "HELOC external validation supports framework robustness.",
            "supported": "PARTIAL",
            "evidence_file": "outputs/tables/external_validation_heloc_results.csv; outputs/tables/taiwan_vs_heloc_framework_comparison.csv; outputs/tables/scre_revision_comparison_heloc.csv",
            "evidence_metric": (
                f"HELOC external validation includes {external_models} test-split models; "
                f"Scorecard cost={fmt(heloc_scorecard['expected_cost'])}; CatBoost cost={fmt(heloc_cat['expected_cost'])}; "
                f"Old SCRE cost={fmt(heloc_old_scre['expected_cost'])}; SCRE-Pareto cost={fmt(heloc_scre_pareto['expected_cost'])}; "
                f"SCRE-Optimized PR-AUC={fmt(heloc_scre_opt['pr_auc'])}, cost={fmt(heloc_scre_opt['expected_cost'])}."
            ),
            "safe_wording": "HELOC external validation supports pipeline portability and shows SCRE variants remain competitive under domain shift, but it does not prove universal SCRE dominance.",
            "unsafe_wording": "HELOC proves the SCRE-Credit framework is robust in all external settings.",
            "final_status": "CLAIM WITH SCOPE",
        },
        {
            "claim": "The project proposes a reliability-aware framework, not a new fundamental ML algorithm.",
            "supported": "YES",
            "evidence_file": "PROJECT_SCOPE.md; outputs/final_decision_summary.md",
            "evidence_metric": (
                "PROJECT_SCOPE.md states SCRE-Credit is a framework-level contribution and "
                f"no-new-base-classifier guardrail="
                f"{'yes' if 'framework-level contribution' in project_scope.lower() and 'does not claim to invent a new base classifier' in project_scope.lower() else 'check manually'}."
            ),
            "safe_wording": "SCRE-Credit is presented as a reliability-aware ensemble and evaluation framework for credit risk, not as a new fundamental machine-learning algorithm.",
            "unsafe_wording": "We invented a new fundamental ML algorithm for credit scoring.",
            "final_status": "SAFE TO CLAIM",
        },
    ]

    expected_columns = [
        "claim",
        "supported",
        "evidence_file",
        "evidence_metric",
        "safe_wording",
        "unsafe_wording",
        "final_status",
    ]
    result = pd.DataFrame(rows, columns=expected_columns)
    if result[expected_columns].isna().any().any():
        raise ValueError("Claim-control matrix contains missing required fields.")
    return result


def main() -> None:
    """Write the claim-control matrix to disk."""

    LOGGER.info("Building claim-control matrix from current evidence artifacts.")
    claim_control = build_claim_control_matrix()
    output_path = save_table(claim_control, TABLES_DIR / "claim_control_matrix.csv")
    status_counts = claim_control["final_status"].value_counts().to_dict()
    LOGGER.info("Wrote %s with %d claims. Status counts: %s", output_path, len(claim_control), status_counts)
    print(f"Wrote {output_path}")
    print(claim_control[["claim", "supported", "final_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
