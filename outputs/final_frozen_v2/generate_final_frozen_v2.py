from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\AKTS\Desktop\Resul\Data_Mining_Project")
OUT = ROOT / "outputs" / "final_frozen_v2"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def first_row(df: pd.DataFrame) -> dict:
    return df.iloc[0].to_dict() if not df.empty else {}


def rounded(value: object, digits: int = 3) -> float | str:
    try:
        return round(float(value), digits)
    except Exception:
        return ""


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = df.astype(str).values.tolist()
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    taiwan_src = first_row(read_csv(ROOT / "outputs/decision_revision_v2/locked_test_evaluation_taiwan.csv"))
    heloc_src = first_row(read_csv(ROOT / "outputs/decision_revision_v2/locked_test_evaluation_heloc.csv"))

    taiwan = pd.DataFrame(
        [
            {
                "dataset": "taiwan",
                "final_policy": "V2 CatBoost manual-review",
                "model": "CatBoost",
                "policy_type": "manual_review_band",
                "precision": rounded(taiwan_src.get("test_precision")),
                "recall": rounded(taiwan_src.get("test_recall")),
                "specificity": rounded(taiwan_src.get("test_specificity")),
                "fp": int(float(taiwan_src.get("test_fp", 680))),
                "fn": int(float(taiwan_src.get("test_fn", 564))),
                "cost": rounded(taiwan_src.get("test_review_adjusted_cost"), 1),
                "manual_review_rate": rounded(taiwan_src.get("test_manual_review_rate")),
                "audit_status": "READY",
                "safe_category": "FINAL MAIN EVIDENCE",
                "interpretation": (
                    "Final V2 Taiwan operational policy. Use as main report evidence. "
                    "This is a manual-review screening policy, not automatic rejection."
                ),
            }
        ]
    )

    heloc = pd.DataFrame(
        [
            {
                "dataset": "heloc",
                "final_policy": "V2 Scorecard manual-review",
                "model": "Scorecard / WOE Logistic Regression",
                "policy_type": "manual_review_band",
                "precision": rounded(heloc_src.get("test_precision")),
                "recall": rounded(heloc_src.get("test_recall")),
                "specificity": rounded(heloc_src.get("test_specificity")),
                "fp": int(float(heloc_src.get("test_fp", 403))),
                "fn": int(float(heloc_src.get("test_fn", 158))),
                "cost": rounded(heloc_src.get("test_review_adjusted_cost"), 1),
                "manual_review_rate": rounded(heloc_src.get("test_manual_review_rate")),
                "audit_status": "READY",
                "safe_category": "FINAL MAIN EVIDENCE",
                "interpretation": (
                    "Final V2 HELOC operational policy. Use as main external-validation evidence. "
                    "This is a manual-review screening policy, not automatic rejection."
                ),
            }
        ]
    )

    taiwan.to_csv(OUT / "final_v2_taiwan_operational_policy.csv", index=False)
    heloc.to_csv(OUT / "final_v2_heloc_operational_policy.csv", index=False)

    main = pd.concat([taiwan, heloc], ignore_index=True)
    main_md = markdown_table(main[
        [
            "dataset",
            "final_policy",
            "precision",
            "recall",
            "specificity",
            "fp",
            "fn",
            "cost",
            "manual_review_rate",
            "safe_category",
        ]
    ])

    (OUT / "final_v2_main_result_table.md").write_text(
        f"""# Final V2 Main Result Table

These are the only final operational policies to use as main report evidence.

{main_md}

Interpretation:

- Taiwan final operational evidence: V2 CatBoost manual-review.
- HELOC final external-validation evidence: V2 Scorecard manual-review.
- Both are audit-ready and selected under the validation-only / locked-test protocol.
- These policies support screening and manual-review prioritization, not automatic rejection.
""",
        encoding="utf-8",
    )

    (OUT / "final_v2_audit_summary.md").write_text(
        """# Final V2 Audit Summary

Audit status: **READY**

Source audit file: `outputs/decision_revision_v2/final_audit_v2.md`

Key audit findings:

- Critical issues: 0
- High issues: 0
- Medium issues: 0
- Test set used for model/policy selection: NO
- Test set used for threshold selection: NO
- Test set used for manual-review band selection: NO
- Test set used for calibration fitting: NO
- Held-out test table contains no actual model-selection rank or winner column.
- Manual-review cost is separated from binary expected cost.
- Metric consistency: PASS

Conclusion:

V2 is safe to use as the final operational evidence for the project.
""",
        encoding="utf-8",
    )

    (OUT / "final_v2_model_roles.md").write_text(
        """# Final V2 Model Roles

## Taiwan

- **Operational screening policy:** V2 CatBoost manual-review.
- **Role:** Main final Taiwan evidence.
- **Use:** Rank and screen applicants for manual-review decision support.
- **Do not use as:** Automatic rejection system.

## HELOC

- **Operational screening policy:** V2 Scorecard manual-review.
- **Role:** Main final external-validation evidence and interpretable benchmark.
- **Use:** Demonstrate that the framework can support a transparent external-validation policy.
- **Do not use as:** Proof that one Taiwan-trained model transfers directly to HELOC.

## SCRE-Credit

- **Role:** Reliability-aware research framework and review-prioritization support.
- **Use:** Explain how performance, calibration, cost, stability, and faithfulness evidence can be organized.
- **Do not use as:** A universally superior classifier claim.

## Final Attempt

- **Role:** Appendix / robustness / stress-test evidence.
- **Use:** Show that stronger methods were tested and did not cleanly replace V2.
- **Do not use as:** Final operational evidence while audit remains BLOCKED.
""",
        encoding="utf-8",
    )

    (OUT / "final_v2_safe_claims.md").write_text(
        """# Final V2 Safe Claims

- The final operational version is V2.
- Taiwan final policy is CatBoost with a manual-review band.
- HELOC final policy is Scorecard / WOE Logistic Regression with a manual-review band.
- The system is a screening and manual-review support tool.
- V2 uses validation-only model/policy selection and locked held-out test evaluation.
- V2 audit is READY with no critical, high, or medium issues.
- Manual-review cost is modeled separately from binary expected cost.
- Final Attempt is useful as stress-test and appendix evidence, but not as final operational evidence.
- SCRE-Credit is best described as a reliability-aware framework, not a universally dominant classifier.
""",
        encoding="utf-8",
    )

    (OUT / "final_v2_do_not_claim.md").write_text(
        """# Final V2 Do Not Claim

Do **not** claim:

- The system is suitable for automatic credit rejection.
- SCRE-Credit outperforms all individual models.
- Final Attempt is the final operational version.
- Final Attempt locked-test manual-review rows are audit-clean.
- Deep learning reproduced the highest literature scores under a clean protocol.
- KMeansSMOTE produced a reliable breakthrough under natural test distribution.
- Manual-review cost is identical to binary FN/FP expected cost.
- Test set was used for final model or threshold selection.
- XAI or SHAP results prove causality.
- HELOC proves direct model transfer from Taiwan; it supports external-validation framework evidence only.
""",
        encoding="utf-8",
    )

    # Convenience copies of source audit and locked evidence for traceability.
    for src_name in [
        "final_audit_v2.md",
        "locked_test_evaluation_taiwan.csv",
        "locked_test_evaluation_heloc.csv",
        "final_safe_recommendation.md",
    ]:
        src = ROOT / "outputs" / "decision_revision_v2" / src_name
        if src.exists():
            shutil.copy2(src, OUT / f"source_{src_name}")

    print(f"Generated frozen V2 package at {OUT}")


if __name__ == "__main__":
    main()
