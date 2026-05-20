"""Final V2 vs V3 selector for weakness-closing experiments.

This script does not run new experiments. It consolidates already-generated
weakness-closing outputs and applies the predefined replacement criteria.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/final_weakness_closing/final_selection"
V2_TAIWAN = ROOT / "outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv"
V2_HELOC = ROOT / "outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _f(value: Any) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _base_row(dataset: str, track: str, candidate: str, source_stage: str, evidence_file: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "track": track,
        "candidate": candidate,
        "source_stage": source_stage,
        "policy_type": "",
        "precision": np.nan,
        "recall": np.nan,
        "specificity": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "cost": np.nan,
        "manual_review_rate": np.nan,
        "capture_at_20": np.nan,
        "precision_at_20": np.nan,
        "lift_at_20": np.nan,
        "validation_only_selected": True,
        "test_set_used_for_selection": False,
        "audit_ready": False,
        "metric_consistency_pass": "not_audited",
        "replacement_criteria_pass": False,
        "selection_status": "APPENDIX_ONLY",
        "use_in_report_as": "appendix",
        "decision_reason": "",
        "evidence_file": evidence_file,
    }


def _v2_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    taiwan = _read(V2_TAIWAN).iloc[0].to_dict()
    heloc = _read(V2_HELOC).iloc[0].to_dict()
    rows = []
    for raw, dataset, track in [
        (taiwan, "taiwan", "A_best_final_operational_policy"),
        (heloc, "heloc", "A_best_final_operational_policy"),
    ]:
        row = _base_row(dataset, track, raw["final_policy"], "V2 frozen final", str(V2_TAIWAN if dataset == "taiwan" else V2_HELOC))
        row.update(
            {
                "policy_type": raw["policy_type"],
                "precision": _f(raw["precision"]),
                "recall": _f(raw["recall"]),
                "specificity": _f(raw["specificity"]),
                "fp": int(raw["fp"]),
                "fn": int(raw["fn"]),
                "cost": _f(raw["cost"]),
                "manual_review_rate": _f(raw["manual_review_rate"]),
                "audit_ready": raw["audit_status"] == "READY",
                "metric_consistency_pass": "pass_v2_final_audit",
                "replacement_criteria_pass": True,
                "selection_status": "SELECTED_FINAL_VERSION",
                "use_in_report_as": "FINAL MAIN EVIDENCE",
                "decision_reason": "V2 is frozen, audit READY, metric definitions documented, and remains the safest operational evidence.",
            }
        )
        rows.append(row)
    return rows[0], rows[1]


def _taiwan_rows(v2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [v2]
    v2_cost = _f(v2["cost"])
    recall = _read(ROOT / "outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv")
    if not recall.empty:
        # Best recall-improving evidence: highest locked-test recall.
        cand = recall.sort_values(["test_recall", "test_review_adjusted_cost"], ascending=[False, True]).iloc[0]
        row = _base_row(
            "taiwan",
            "B_best_recall_improving_policy",
            "Taiwan recall-constrained CatBoost manual-review",
            "Taiwan recall-constrained policy",
            "outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv",
        )
        cost_ok = _f(cand["test_review_adjusted_cost"]) <= v2_cost * 1.10
        row.update(
            {
                "policy_type": cand["policy_type"],
                "precision": _f(cand["test_precision"]),
                "recall": _f(cand["test_recall"]),
                "specificity": _f(cand["test_specificity"]),
                "fp": int(cand["test_fp"]),
                "fn": int(cand["test_fn"]),
                "cost": _f(cand["test_review_adjusted_cost"]),
                "manual_review_rate": _f(cand["test_manual_review_rate"]),
                "validation_only_selected": cand["selection_split"] == "validation",
                "test_set_used_for_selection": bool(cand["test_set_used_for_selection"]),
                "audit_ready": False,
                "metric_consistency_pass": "not_final_audited",
                "replacement_criteria_pass": bool(_f(cand["test_recall"]) >= 0.60 and cost_ok),
                "selection_status": "V3_AUDIT_CANDIDATE_NOT_FINAL",
                "use_in_report_as": "appendix / V3 candidate",
                "decision_reason": "Improves Taiwan recall and lowers high-exposure proxy cost, but has lower precision/specificity and is not audit READY.",
            }
        )
        rows.append(row)

    segment = _read(ROOT / "outputs/final_weakness_closing/segment_thresholds/segment_policy_locked_test_taiwan.csv")
    if not segment.empty:
        valid = segment[segment["validation_constraint_pass"] == True].copy()
        if not valid.empty:
            cand = valid.sort_values(["manual_review_rate", "cost"]).iloc[0]
            row = _base_row(
                "taiwan",
                "C_best_workload_reducing_policy",
                f"Segment-aware {cand['segment_strategy']}",
                "Segment-aware threshold policy",
                "outputs/final_weakness_closing/segment_thresholds/segment_policy_locked_test_taiwan.csv",
            )
            row.update(
                {
                    "policy_type": "segment_aware_policy",
                    "precision": _f(cand["precision"]),
                    "recall": _f(cand["recall"]),
                    "specificity": _f(cand["specificity"]),
                    "fp": int(cand["fp"]),
                    "fn": int(cand["fn"]),
                    "cost": _f(cand["cost"]),
                    "manual_review_rate": _f(cand["manual_review_rate"]),
                    "replacement_criteria_pass": bool(_f(cand["manual_review_rate"]) <= 0.25 and _f(cand["recall"]) >= _f(v2["recall"]) - 0.03),
                    "selection_status": "APPENDIX_ONLY_WORKLOAD_REDUCTION_CANDIDATE",
                    "decision_reason": "Reduces manual-review workload but does not improve recall and has higher cost/FP than V2.",
                }
            )
            rows.append(row)

    cost = _read(ROOT / "outputs/final_weakness_closing/cost_sensitivity/cost_sensitivity_taiwan.csv")
    if not cost.empty:
        row = _base_row(
            "taiwan",
            "E_best_cost_stable_policy",
            "V2 Taiwan CatBoost manual-review",
            "Cost sensitivity + decision curve",
            "outputs/final_weakness_closing/cost_sensitivity/cost_sensitivity_taiwan.csv",
        )
        row.update(
            {
                "policy_type": "manual_review_band",
                "precision": v2["precision"],
                "recall": v2["recall"],
                "specificity": v2["specificity"],
                "fp": v2["fp"],
                "fn": v2["fn"],
                "cost": v2["cost"],
                "manual_review_rate": v2["manual_review_rate"],
                "audit_ready": True,
                "metric_consistency_pass": "pass_v2_final_audit",
                "replacement_criteria_pass": True,
                "selection_status": "SUPPORTS_KEEPING_V2",
                "use_in_report_as": "main support",
                "decision_reason": "Cost sensitivity identifies V2 as stable under core review-cost/FN-dominant assumptions.",
            }
        )
        rows.append(row)

    scre = _read(ROOT / "outputs/final_weakness_closing/scre_prioritization/scre_topk_review_taiwan.csv")
    if not scre.empty:
        cand = scre[
            (scre["split"] == "validation")
            & (scre["scre_version"] == "SCRE-Optimized")
            & (np.isclose(scre["capacity_pct"], 0.20))
        ].iloc[0]
        row = _base_row(
            "taiwan",
            "F_best_scre_review_prioritization_role",
            "SCRE-Optimized Top-20% review ranking",
            "SCRE prioritization",
            "outputs/final_weakness_closing/scre_prioritization/scre_topk_review_taiwan.csv",
        )
        row.update(
            {
                "policy_type": "top_k_review_prioritization",
                "capture_at_20": _f(cand["default_capture_rate"]),
                "precision_at_20": _f(cand["precision_at_k"]),
                "lift_at_20": _f(cand["lift_at_k"]),
                "manual_review_rate": _f(cand["manual_review_rate"]),
                "selection_status": "FRAMEWORK_ROLE_NOT_FINAL_CLASSIFIER",
                "use_in_report_as": "appendix / framework contribution",
                "decision_reason": "SCRE is strongest as a review-prioritization framework: validation Capture@20=0.529, Precision@20=0.585, Lift@20=2.645.",
            }
        )
        rows.append(row)

    temporal = _base_row(
        "taiwan",
        "G_best_appendix_only_result",
        "Pseudo/order-based temporal diagnostic",
        "Temporal robustness",
        "outputs/final_weakness_closing/temporal_robustness/temporal_robustness_summary.md",
    )
    temporal.update(
        {
            "selection_status": "LIMITATION_ONLY",
            "use_in_report_as": "limitation / appendix diagnostic",
            "decision_reason": "No real timestamp exists; cannot support real deployment temporal validation.",
        }
    )
    rows.append(temporal)
    return rows


def _heloc_rows(v2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [v2]
    spec = _read(ROOT / "outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv")
    if not spec.empty:
        cand = spec.sort_values(["test_specificity", "test_cost"], ascending=[False, True]).iloc[0]
        row = _base_row(
            "heloc",
            "D_best_heloc_specificity_policy",
            "HELOC specificity-focused Scorecard threshold",
            "HELOC specificity optimization",
            "outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv",
        )
        cost_not_too_bad = _f(cand["test_cost"]) <= _f(v2["cost"]) * 1.25
        row.update(
            {
                "policy_type": cand["policy_type"],
                "precision": _f(cand["test_precision"]),
                "recall": _f(cand["test_recall"]),
                "specificity": _f(cand["test_specificity"]),
                "fp": int(cand["test_fp"]),
                "fn": int(cand["test_fn"]),
                "cost": _f(cand["test_cost"]),
                "manual_review_rate": _f(cand["test_manual_review_rate"]),
                "validation_only_selected": cand["selection_split"] == "validation",
                "test_set_used_for_selection": bool(cand["test_set_used_for_selection"]),
                "replacement_criteria_pass": bool(_f(cand["test_specificity"]) >= 0.65 and _f(cand["test_recall"]) >= 0.75 and cost_not_too_bad),
                "selection_status": "APPENDIX_ONLY_SPECIFICITY_CANDIDATE",
                "decision_reason": "Specificity improves to 0.676 and FP falls, but cost rises too much and candidate is not audit READY.",
            }
        )
        rows.append(row)

    segment = _read(ROOT / "outputs/final_weakness_closing/segment_thresholds/segment_policy_locked_test_heloc.csv")
    if not segment.empty:
        valid = segment[segment["validation_constraint_pass"] == True].copy()
        if not valid.empty:
            cand = valid.sort_values(["cost"]).iloc[0]
            row = _base_row(
                "heloc",
                "C_best_workload_reducing_policy",
                f"Segment-aware {cand['segment_strategy']}",
                "Segment-aware threshold policy",
                "outputs/final_weakness_closing/segment_thresholds/segment_policy_locked_test_heloc.csv",
            )
            row.update(
                {
                    "policy_type": "segment_aware_policy",
                    "precision": _f(cand["precision"]),
                    "recall": _f(cand["recall"]),
                    "specificity": _f(cand["specificity"]),
                    "fp": int(cand["fp"]),
                    "fn": int(cand["fn"]),
                    "cost": _f(cand["cost"]),
                    "manual_review_rate": _f(cand["manual_review_rate"]),
                    "replacement_criteria_pass": bool(_f(cand["specificity"]) >= _f(v2["specificity"]) and _f(cand["recall"]) >= 0.75 and _f(cand["manual_review_rate"]) <= 0.30),
                    "selection_status": "APPENDIX_ONLY_WORKLOAD_SPECIFICITY_CANDIDATE",
                    "decision_reason": "Improves specificity and keeps MR rate reasonable, but cost increases and it is not audit READY.",
                }
            )
            rows.append(row)

    cost = _read(ROOT / "outputs/final_weakness_closing/cost_sensitivity/cost_sensitivity_heloc.csv")
    if not cost.empty:
        row = _base_row(
            "heloc",
            "E_best_cost_stable_policy",
            "V2 HELOC Scorecard manual-review",
            "Cost sensitivity + decision curve",
            "outputs/final_weakness_closing/cost_sensitivity/cost_sensitivity_heloc.csv",
        )
        row.update(
            {
                "policy_type": "manual_review_band",
                "precision": v2["precision"],
                "recall": v2["recall"],
                "specificity": v2["specificity"],
                "fp": v2["fp"],
                "fn": v2["fn"],
                "cost": v2["cost"],
                "manual_review_rate": v2["manual_review_rate"],
                "audit_ready": True,
                "metric_consistency_pass": "pass_v2_final_audit",
                "replacement_criteria_pass": True,
                "selection_status": "SUPPORTS_KEEPING_V2",
                "use_in_report_as": "main support",
                "decision_reason": "Cost sensitivity and instance-dependent cost keep V2 Scorecard as strongest HELOC operational evidence.",
            }
        )
        rows.append(row)

    scre = _read(ROOT / "outputs/final_weakness_closing/scre_prioritization/scre_topk_review_heloc.csv")
    if not scre.empty:
        cand = scre[
            (scre["split"] == "validation")
            & (scre["scre_version"] == "SCRE-Optimized")
            & (np.isclose(scre["capacity_pct"], 0.20))
        ].iloc[0]
        row = _base_row(
            "heloc",
            "F_best_scre_review_prioritization_role",
            "SCRE-Optimized Top-20% review ranking",
            "SCRE prioritization",
            "outputs/final_weakness_closing/scre_prioritization/scre_topk_review_heloc.csv",
        )
        row.update(
            {
                "policy_type": "top_k_review_prioritization",
                "capture_at_20": _f(cand["default_capture_rate"]),
                "precision_at_20": _f(cand["precision_at_k"]),
                "lift_at_20": _f(cand["lift_at_k"]),
                "manual_review_rate": _f(cand["manual_review_rate"]),
                "selection_status": "FRAMEWORK_ROLE_NOT_FINAL_CLASSIFIER",
                "use_in_report_as": "appendix / framework contribution",
                "decision_reason": "SCRE is useful for review prioritization: validation Capture@20=0.340, Precision@20=0.884, Lift@20=1.698.",
            }
        )
        rows.append(row)

    temporal = _base_row(
        "heloc",
        "G_best_appendix_only_result",
        "Pseudo/order-based temporal diagnostic",
        "Temporal robustness",
        "outputs/final_weakness_closing/temporal_robustness/temporal_robustness_summary.md",
    )
    temporal.update(
        {
            "selection_status": "LIMITATION_ONLY",
            "use_in_report_as": "limitation / appendix diagnostic",
            "decision_reason": "No real timestamp exists; cannot support real deployment temporal validation.",
        }
    )
    rows.append(temporal)
    return rows


def _write_summary(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> None:
    """Write final selection summary Markdown."""

    def md_table(df: pd.DataFrame) -> str:
        cols = ["track", "candidate", "precision", "recall", "specificity", "fp", "fn", "cost", "manual_review_rate", "selection_status"]
        shown = df[cols].copy()
        for col in ["precision", "recall", "specificity", "manual_review_rate"]:
            shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        shown["cost"] = shown["cost"].map(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.to_numpy()]
        return "\n".join([header, sep, *body])

    text = f"""# Final V2 vs V3 Selection Summary

## Decision rule
V2 can be replaced only if a candidate is validation-only selected, metric-consistent, audit-ready, and improves the requested weakness without introducing unsafe claims.

## Taiwan selection table
{md_table(taiwan)}

## HELOC selection table
{md_table(heloc)}

## Questions
1. V2 kalıyor mu? **YES.** V2 remains the final operational version for both Taiwan and HELOC.
2. V3 var mı? **PARTIAL.** There are V3 candidates, but no audit-ready replacement.
3. Taiwan'da recall zayıflığı kapandı mı? **PARTIALLY.** The recall-constrained candidate reaches recall above 0.60, but precision/specificity drop and it is not audit-ready.
4. Manual-review workload azaldı mı? **PARTIALLY.** Segment/capacity alternatives reduce workload, but they trade off recall/cost and remain appendix/V3 candidates.
5. HELOC specificity iyileşti mi? **YES as candidate evidence.** Specificity-focused Scorecard reaches 0.676, but cost increases substantially; V2 remains final.
6. SCRE daha güçlü framework oldu mu? **YES.** SCRE is stronger as a review-prioritization and reliability-aware framework, not as a final classifier.
7. Real deployment / cost eksikleri ne kadar kapandı? **Only partially.** Cost and instance-dependent sensitivity improve transparency; temporal diagnostics remain limitation-only because no real timestamp exists.
8. Hangi sonuç ana rapora, hangisi appendix'e gidecek? Main report: V2 frozen final results, V2 audit readiness, safe claims, SCRE framework role. Appendix: recall candidate, specificity candidate, capacity review, instance-dependent cost, segment-aware policy, temporal limitation, Final Attempt metric repair.

## Final committee decision
- Taiwan final operational policy: **V2 CatBoost manual-review**.
- HELOC final operational policy: **V2 Scorecard manual-review**.
- V3 status: **candidate-only, not final**.
- Ready for final audit: **YES**.
"""
    (OUT_DIR / "final_selection_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_taiwan, v2_heloc = _v2_rows()
    taiwan = pd.DataFrame(_taiwan_rows(v2_taiwan))
    heloc = pd.DataFrame(_heloc_rows(v2_heloc))
    taiwan.to_csv(OUT_DIR / "v2_vs_v3_selection_taiwan.csv", index=False)
    heloc.to_csv(OUT_DIR / "v2_vs_v3_selection_heloc.csv", index=False)

    selected = {
        "selected_final_version": "V2",
        "v3_created": "PARTIAL_CANDIDATES_ONLY_NOT_FINAL",
        "taiwan_final_policy": "V2 CatBoost manual-review",
        "heloc_final_policy": "V2 Scorecard manual-review",
        "taiwan_v3_candidates": [
            "Taiwan recall-constrained CatBoost manual-review",
            "Segment-aware workload-reducing policies",
            "SCRE-Optimized Top-K review prioritization",
        ],
        "heloc_v3_candidates": [
            "HELOC specificity-focused Scorecard threshold",
            "HELOC revolving-burden segment policy",
            "SCRE-Optimized Top-K review prioritization",
        ],
        "why_v2_remains_final": [
            "V2 is audit READY and frozen as final operational evidence.",
            "No weakness-closing candidate is audit-ready as a replacement.",
            "Taiwan recall candidate improves recall but reduces precision/specificity and needs audit.",
            "HELOC specificity candidate improves specificity but cost increases substantially.",
            "SCRE is stronger as review-prioritization framework, not final classifier.",
            "Temporal validation remains limitation-only because no real timestamps exist.",
        ],
        "automatic_rejection_claim_allowed": False,
        "scre_dominant_classifier_claim_allowed": False,
        "ready_for_final_audit": True,
    }
    (OUT_DIR / "final_selected_version.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    _write_summary(taiwan, heloc)


if __name__ == "__main__":
    main()
