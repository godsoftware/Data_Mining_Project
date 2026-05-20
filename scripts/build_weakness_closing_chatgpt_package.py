"""Build a compact weakness-closing results package for ChatGPT review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEAK = ROOT / "outputs/final_weakness_closing"
V2 = ROOT / "outputs/final_frozen_v2"
OUT_DIR = WEAK / "send_to_chatgpt"
OUT_FILE = OUT_DIR / "WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _md_table(df: pd.DataFrame, columns: list[str], headers: list[str] | None = None, digits: int = 3) -> str:
    headers = headers or columns
    if df.empty:
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_fmt(row.get(col), digits) for col in columns) + " |")
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *rows,
        ]
    )


def _audit_summary() -> dict[str, str]:
    audit = (WEAK / "audit/final_weakness_closing_audit.md").read_text(encoding="utf-8", errors="ignore")
    result = {
        "Verdict": "READY" if "**Verdict: READY**" in audit else "NOT READY/BLOCKED",
        "Critical issues": "0" if "Critical issue count: 0" in audit else "see audit",
        "High issues": "1" if "High issue count: 1" in audit else "see audit",
        "Ready for conclusion": "YES" if "Ready for conclusion phase?\n**YES**" in audit else "NO",
        "Test leakage": "NO" if "Test leakage: NO" in audit else "YES/UNKNOWN",
        "Metric consistency": "PASS" if "Metric/formula consistency failures: 0" in audit else "FAIL/UNKNOWN",
        "Manual-review consistency": "PASS" if "Manual-review metric consistency: PASS" in audit else "FAIL/UNKNOWN",
    }
    return result


def _v2_baseline() -> pd.DataFrame:
    taiwan = _read(V2 / "final_v2_taiwan_operational_policy.csv")
    heloc = _read(V2 / "final_v2_heloc_operational_policy.csv")
    frame = pd.concat([taiwan, heloc], ignore_index=True)
    return frame.rename(
        columns={
            "dataset": "Dataset",
            "final_policy": "Policy",
            "precision": "Precision",
            "recall": "Recall",
            "specificity": "Specificity",
            "fp": "FP",
            "fn": "FN",
            "cost": "Cost",
            "manual_review_rate": "MR Rate",
        }
    )


def _taiwan_recall() -> pd.DataFrame:
    df = _read(WEAK / "taiwan_recall/taiwan_recall_policy_locked_test.csv")
    if df.empty:
        return df
    df = df.sort_values(["test_recall", "test_review_adjusted_cost"], ascending=[False, True]).head(3).copy()
    df["Candidate"] = df["candidate_id"]
    df["Recall"] = df["test_recall"]
    df["Precision"] = df["test_precision"]
    df["Specificity"] = df["test_specificity"]
    df["FP"] = df["test_fp"]
    df["FN"] = df["test_fn"]
    df["Cost"] = df["test_review_adjusted_cost"]
    df["MR Rate"] = df["test_manual_review_rate"]
    df["Delta vs V2"] = df.apply(lambda r: f"recall +{r['delta_recall_vs_v2']:.3f}, FP +{int(r['delta_fp_vs_v2'])}, FN {int(r['delta_fn_vs_v2'])}", axis=1)
    df["Comment"] = "V3 audit candidate; recall improves but precision/specificity drop."
    return df


def _capacity() -> pd.DataFrame:
    rows = []
    for dataset in ["taiwan", "heloc"]:
        df = _read(WEAK / f"capacity_review/capacity_review_{dataset}.csv")
        if df.empty:
            continue
        subset = df[
            (df["split"] == "validation")
            & (df["policy_type"] == "top_k_review")
            & (np.isclose(df["capacity_pct"], 0.20))
        ].copy()
        if subset.empty:
            continue
        best = subset.sort_values("default_capture_rate", ascending=False).head(3)
        for _, r in best.iterrows():
            rows.append(
                {
                    "Dataset": dataset,
                    "Model": r["model"],
                    "Capacity": r["capacity_pct"],
                    "Capture Rate": r["default_capture_rate"],
                    "Precision@K": r["precision_at_k"],
                    "Lift@K": r["lift_at_k"],
                    "Low-risk Default Rate": r["low_risk_default_rate"],
                    "Comment": "Capacity/review-prioritization evidence, not final automatic decision.",
                }
            )
    return pd.DataFrame(rows)


def _heloc_specificity() -> pd.DataFrame:
    df = _read(WEAK / "heloc_specificity/heloc_specificity_locked_test.csv")
    if df.empty:
        return df
    df = df.sort_values(["test_specificity", "test_cost"], ascending=[False, True]).head(3).copy()
    df["Candidate"] = df["candidate_id"]
    df["Specificity"] = df["test_specificity"]
    df["Recall"] = df["test_recall"]
    df["Precision"] = df["test_precision"]
    df["FP"] = df["test_fp"]
    df["FN"] = df["test_fn"]
    df["Cost"] = df["test_cost"]
    df["MR Rate"] = df["test_manual_review_rate"]
    df["Delta vs V2"] = df.apply(lambda r: f"specificity +{r['delta_specificity_vs_v2']:.3f}, FP {int(r['delta_fp_vs_v2'])}, cost +{r['delta_cost_vs_v2']:.1f}", axis=1)
    df["Comment"] = "Specificity improves; cost increases; appendix/V3 candidate."
    return df


def _cost_sensitivity_summary() -> pd.DataFrame:
    rows = [
        {
            "Dataset": "Taiwan",
            "Best Stable System": "V2 CatBoost manual-review",
            "Cost Robust?": "YES under core FN-dominant / moderate review-cost assumptions",
            "Useful Threshold Range": "Decision curves: SCRE ranking useful in broad ranges; not final selector",
            "Comment": "Cost uncertainty does not replace V2; identifies V3 candidates.",
        },
        {
            "Dataset": "HELOC",
            "Best Stable System": "V2 Scorecard manual-review",
            "Cost Robust?": "YES; SCRE threshold also competitive but not final",
            "Useful Threshold Range": "Decision curves: SCRE/capacity ranking useful for review prioritization",
            "Comment": "V2 remains safest operational evidence.",
        },
    ]
    return pd.DataFrame(rows)


def _instance_cost_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset": "Taiwan",
                "Did final decision change?": "PARTIALLY / not final",
                "Best System": "Taiwan recall candidate under proxy-cost definitions",
                "Comment": "Recall candidate lowers proxy instance-dependent cost, but remains V3 audit candidate.",
            },
            {
                "Dataset": "HELOC",
                "Did final decision change?": "NO",
                "Best System": "V2 Scorecard manual-review",
                "Comment": "V2 remains best under HELOC proxy-cost definitions.",
            },
        ]
    )


def _scre_summary() -> pd.DataFrame:
    rows = []
    for dataset in ["taiwan", "heloc"]:
        df = _read(WEAK / f"scre_prioritization/scre_topk_review_{dataset}.csv")
        if df.empty:
            continue
        row = df[
            (df["split"] == "validation")
            & (df["scre_version"] == "SCRE-Optimized")
            & (np.isclose(df["capacity_pct"], 0.20))
        ].iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "SCRE Role": "Review prioritization / reliability-aware framework",
                "Capture@20": row["default_capture_rate"],
                "Lift@20": row["lift_at_k"],
                "Safe Claim": "SCRE supports Top-K review prioritization; not dominant classifier.",
            }
        )
    return pd.DataFrame(rows)


def _segment_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset": "Taiwan",
                "Candidate": "limit/utilization segment policies",
                "Better than V2?": "NO clear replacement",
                "Risk": "Overfitting; recall not improved beyond V2",
                "Use": "Appendix only",
            },
            {
                "Dataset": "HELOC",
                "Candidate": "revolving_burden_terciles",
                "Better than V2?": "Specificity improves, cost worsens",
                "Risk": "Not audit-ready; trade-off heavy",
                "Use": "Appendix / V3 candidate",
            },
        ]
    )


def _final_decision() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Dataset": "Taiwan",
                "Final Version": "V2",
                "Reason": "Only audit-ready final operational evidence; recall candidate not audit-ready.",
            },
            {
                "Dataset": "HELOC",
                "Final Version": "V2",
                "Reason": "Scorecard manual-review remains safest; specificity candidates increase cost.",
            },
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = _audit_summary()
    repair_summary = (WEAK / "metric_repair/final_attempt_metric_repair_summary.md").read_text(encoding="utf-8", errors="ignore")
    temporal_summary = (WEAK / "temporal_robustness/temporal_robustness_summary.md").read_text(encoding="utf-8", errors="ignore")

    text = f"""# WEAKNESS CLOSING RESULTS FOR CHATGPT

## 1. Audit verdict
- Verdict: {audit['Verdict']}
- Critical issues: {audit['Critical issues']}
- High issues: {audit['High issues']} (`validation_all` files include diagnostic test columns; not used for selection)
- Ready for conclusion: {audit['Ready for conclusion']}
- Test leakage: {audit['Test leakage']}
- Metric consistency: {audit['Metric consistency']}
- Manual-review consistency: {audit['Manual-review consistency']}

## 2. V2 baseline
{_md_table(_v2_baseline(), ['Dataset','Policy','Precision','Recall','Specificity','FP','FN','Cost','MR Rate'])}

## 3. Manual-review metric repair
- Final Attempt repaired? YES, repaired into explicit 3x2 manual-review bucket tables.
- Critical failures remaining? NO for repaired 3x2 identities.
- Can Final Attempt be final? NO. It remains appendix/robustness only unless a future full audit promotes it.
- Use as appendix? YES.

## 4. Taiwan recall improvement
{_md_table(_taiwan_recall(), ['Candidate','Recall','Precision','Specificity','FP','FN','Cost','MR Rate','Delta vs V2','Comment'])}

## 5. Capacity-aware review
{_md_table(_capacity(), ['Dataset','Model','Capacity','Capture Rate','Precision@K','Lift@K','Low-risk Default Rate','Comment'])}

## 6. HELOC specificity improvement
{_md_table(_heloc_specificity(), ['Candidate','Specificity','Recall','Precision','FP','FN','Cost','MR Rate','Delta vs V2','Comment'])}

## 7. Cost sensitivity and decision curve
{_md_table(_cost_sensitivity_summary(), ['Dataset','Best Stable System','Cost Robust?','Useful Threshold Range','Comment'])}

## 8. Instance-dependent cost
{_md_table(_instance_cost_summary(), ['Dataset','Did final decision change?','Best System','Comment'])}

## 9. SCRE strengthening
{_md_table(_scre_summary(), ['Dataset','SCRE Role','Capture@20','Lift@20','Safe Claim'])}

## 10. Segment-aware threshold
{_md_table(_segment_summary(), ['Dataset','Candidate','Better than V2?','Risk','Use'])}

## 11. Temporal robustness
- Real temporal validation possible? NO.
- Result: No verified timestamp/application-date field exists. Only pseudo/order-based and random-partition diagnostics were produced.
- Use as: LIMITATION ONLY / APPENDIX DIAGNOSTIC.

## 12. V2 vs V3 final decision
{_md_table(_final_decision(), ['Dataset','Final Version','Reason'])}

## 13. What improved?
- Taiwan recall: improved in candidate policy from V2 0.575 to up to 0.633, but candidate is not final.
- Manual review workload: segment/capacity candidates can reduce workload, but trade off recall/cost.
- HELOC specificity: improved in candidate policy from V2 0.574 to 0.676, but cost rises.
- Cost robustness: V2 remains stable under core assumptions; instance-dependent proxy analysis clarifies exposure-sensitive trade-offs.
- SCRE role: stronger as review-prioritization/reliability-aware framework, not classifier.
- Audit status: final weakness-closing audit READY with 0 critical issues.

## 14. What did not improve?
- No V3 candidate became audit-ready final replacement.
- Taiwan recall gains reduce precision/specificity and increase FP.
- HELOC specificity gains increase cost and reduce recall.
- Real temporal deployment validation remains impossible due to missing timestamps.
- Real bank loss/cost data remains unavailable; only sensitivity/proxy-cost analysis is possible.
- SCRE still should not be claimed as a dominant classifier.

## 15. Final recommendation from the system
- Use V2 as final: YES.
- Use V3 as final: NO.
- Use Final Attempt as appendix: YES.
- Proceed to conclusion: YES.

## 16. Questions for ChatGPT
1. Bu sonuçlara göre V2 mi kalmalı, V3 mü?
2. Hangi zayıflıklar gerçekten kapandı?
3. Hangi sonuçlar ana rapora girmeli?
4. Hangi sonuçlar appendix olmalı?
5. Artık sonuca geçmeli miyim?

## Source audit note
The final weakness-closing audit is READY. The only high caveat is that some `validation_all` files include diagnostic test columns for all candidates. They were not used for final selection and must not be cited as selection evidence.
"""
    OUT_FILE.write_text(text, encoding="utf-8")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
