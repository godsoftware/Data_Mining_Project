from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\AKTS\Desktop\Resul\Data_Mining_Project")
OUT = ROOT / "outputs" / "final_frozen_v2"


def main() -> None:
    rows = [
        {
            "claim": "V2 is the final operational version.",
            "status": "SUPPORTED",
            "evidence": "outputs/final_frozen_v2/final_v2_audit_summary.md; V2 audit READY.",
            "safe_wording": "V2 is used as the final operational version in this report.",
            "unsafe_wording": "All later experiments supersede V2.",
            "notes": "Final Attempt remains appendix/stress-test evidence only.",
        },
        {
            "claim": "Taiwan final policy is CatBoost manual-review.",
            "status": "SUPPORTED",
            "evidence": "outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv",
            "safe_wording": "For Taiwan, the final operational screening policy is V2 CatBoost manual-review.",
            "unsafe_wording": "CatBoost should automatically reject Taiwan applicants.",
            "notes": "Use screening/manual-review wording.",
        },
        {
            "claim": "HELOC final policy is Scorecard manual-review.",
            "status": "SUPPORTED",
            "evidence": "outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv",
            "safe_wording": "For HELOC, the final operational screening policy is V2 Scorecard manual-review.",
            "unsafe_wording": "The Taiwan model directly transfers to HELOC.",
            "notes": "HELOC supports external-validation framework evidence, not direct model transfer.",
        },
        {
            "claim": "Final Attempt replaced V2.",
            "status": "DO NOT CLAIM",
            "evidence": "outputs/final_frozen_v2/final_attempt_usage_status.md; Final Attempt audit BLOCKED.",
            "safe_wording": "Final Attempt is used as appendix robustness and stress-test evidence.",
            "unsafe_wording": "Final Attempt replaced V2 as the final operational version.",
            "notes": "Final Attempt locked-test manual-review rows are not audit-clean.",
        },
        {
            "claim": "SCRE-Credit beats all individual models.",
            "status": "DO NOT CLAIM",
            "evidence": "outputs/final_frozen_v2/final_v2_model_roles.md; outputs/final_project_inventory/scre_role_assessment.md",
            "safe_wording": "SCRE-Credit is evaluated as a reliability-aware framework and review-prioritization support tool.",
            "unsafe_wording": "SCRE-Credit outperforms all individual models.",
            "notes": "CatBoost and Scorecard remain final operational winners for Taiwan and HELOC.",
        },
        {
            "claim": "SCRE-Credit is a reliability-aware framework.",
            "status": "SUPPORTED",
            "evidence": "outputs/final_frozen_v2/final_v2_model_roles.md",
            "safe_wording": "SCRE-Credit is positioned as a reliability-aware framework that organizes performance, calibration, cost, stability, faithfulness, and external-validation evidence.",
            "unsafe_wording": "SCRE-Credit is a new fundamental machine-learning algorithm.",
            "notes": "Framework claim is safe; universal dominance is not.",
        },
        {
            "claim": "The model is suitable for automatic credit rejection.",
            "status": "DO NOT CLAIM",
            "evidence": "outputs/final_frozen_v2/final_v2_do_not_claim.md",
            "safe_wording": "The model is suitable for screening and manual-review support, not automatic rejection.",
            "unsafe_wording": "The model can automatically reject credit applicants.",
            "notes": "Precision/FP behavior and governance requirements require manual-review framing.",
        },
        {
            "claim": "The model is suitable for screening/manual-review prioritization.",
            "status": "SUPPORTED",
            "evidence": "outputs/final_frozen_v2/final_v2_main_result_table.md; outputs/final_frozen_v2/manual_review_metric_definitions.md",
            "safe_wording": "The final system should be presented as screening/manual-review prioritization support.",
            "unsafe_wording": "The final system is a fully automated lending decision engine.",
            "notes": "This is the safest operational framing.",
        },
        {
            "claim": "Final Attempt can be used as appendix robustness evidence.",
            "status": "SUPPORTED WITH CAVEAT",
            "evidence": "outputs/final_frozen_v2/final_attempt_usage_status.md",
            "safe_wording": "Final Attempt is included as appendix robustness evidence, clearly labeled as not final operational evidence.",
            "unsafe_wording": "Final Attempt results are the main final results.",
            "notes": "Always mention BLOCKED audit if discussing final locked-test rows.",
        },
        {
            "claim": "Final Attempt locked-test rows are final evidence.",
            "status": "DO NOT CLAIM",
            "evidence": "outputs/final_attempt/audit/final_attempt_audit.md",
            "safe_wording": "Final Attempt locked-test rows are diagnostic only until metric-consistency issues are repaired.",
            "unsafe_wording": "Final Attempt locked-test rows are final audit-clean evidence.",
            "notes": "Audit reported 9 critical metric-consistency failures.",
        },
        {
            "claim": "Literature-inspired resampling/DNN gave a clean operational replacement.",
            "status": "DO NOT CLAIM",
            "evidence": "outputs/final_attempt/report_to_user/FINAL_ATTEMPT_RESULTS_FOR_CHATGPT.md; outputs/final_project_inventory/literature_positioning_summary.md",
            "safe_wording": "Literature-inspired resampling and DNN experiments were useful robustness checks but did not cleanly replace V2.",
            "unsafe_wording": "KMeansSMOTE or DNN reproduced the high literature scores and replaced V2.",
            "notes": "Use as negative/robustness finding.",
        },
        {
            "claim": "Manual-review cost is identical to binary FN/FP cost.",
            "status": "DO NOT CLAIM",
            "evidence": "outputs/final_frozen_v2/manual_review_metric_definitions.md",
            "safe_wording": "Manual-review cost is separate from binary FN/FP cost and includes workflow/review assumptions.",
            "unsafe_wording": "Manual-review cost can be interpreted exactly like binary expected cost.",
            "notes": "Manual-review policies have low-risk, manual-review, and high-risk buckets.",
        },
    ]
    pd.DataFrame(rows).to_csv(OUT / "final_claim_control_matrix.csv", index=False)
    print(OUT / "final_claim_control_matrix.csv")


if __name__ == "__main__":
    main()
