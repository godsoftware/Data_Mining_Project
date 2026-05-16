from __future__ import annotations

import math
import shutil
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\Users\AKTS\Desktop\Resul\Data_Mining_Project")
REPORT_DIR = ROOT / "outputs" / "reporting"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HTML_PATH = REPORT_DIR / "2540041007_Ozkale_Week13_Progress_Report_complete_colored_template_source.html"
DOCX_PATH = REPORT_DIR / "2540041007_Ozkale_Week13_Progress_Report.docx"
PDF_PATH = REPORT_DIR / "2540041007_Ozkale_Week13_Progress_Report.pdf"
STYLE_BACKUP = REPORT_DIR / "2540041007_Ozkale_Week13_Progress_Report_week12style_backup.docx"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick_first(df: pd.DataFrame, contains: str | None = None, col: str | None = None) -> dict:
    if df.empty:
        return {}
    if contains and col and col in df.columns:
        mask = df[col].astype(str).str.contains(contains, case=False, na=False)
        if mask.any():
            return df[mask].iloc[0].to_dict()
    return df.iloc[0].to_dict()


def fmt(value: object, digits: int = 3) -> str:
    try:
        if value is None:
            return "Not available"
        if isinstance(value, str) and value.strip() == "":
            return "Not available"
        if pd.isna(value):
            return "Not available"
        if isinstance(value, str):
            return value
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_int(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "Not available"
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def best_by_dataset(
    df: pd.DataFrame,
    dataset: str,
    rank_col: str | None = None,
    sort_col: str | None = None,
    ascending: bool = True,
    split_col: str | None = None,
    split_val: str | None = None,
) -> dict:
    if df.empty or "dataset" not in df.columns:
        return {}
    sub = df[df["dataset"].astype(str).str.lower() == dataset.lower()].copy()
    if split_col and split_val and split_col in sub.columns:
        sub = sub[sub[split_col].astype(str).str.lower() == split_val.lower()]
    if sub.empty:
        return {}
    if rank_col and rank_col in sub.columns:
        sub = sub.sort_values(rank_col, ascending=True)
    elif sort_col and sort_col in sub.columns:
        sub = sub.sort_values(sort_col, ascending=ascending)
    return sub.iloc[0].to_dict()


def td(text: object, bg: str = "white", color: str = "#111111", bold: bool = False) -> str:
    weight = "bold" if bold else "normal"
    return (
        f'<td style="background:{bg}; color:{color}; font-weight:{weight}; '
        f'border:1px solid #AAAAAA; padding:5px; vertical-align:top;">{escape(str(text))}</td>'
    )


def th(text: object) -> str:
    return (
        '<th style="background:#1F3864; color:white; font-weight:bold; '
        f'border:1px solid #AAAAAA; padding:5px; vertical-align:top;">{escape(str(text))}</th>'
    )


def table(headers: list[str], rows: list[list[object]], cls: str = "evidence", zebra: bool = True) -> str:
    out = [f'<table class="{cls}">']
    out.append("<tr>" + "".join(th(header) for header in headers) + "</tr>")
    for idx, row in enumerate(rows):
        bg = "#F2F7FC" if zebra and idx % 2 == 1 else "white"
        out.append("<tr>" + "".join(td(value, bg=bg) for value in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def status_cell(text: str, bg: str = "#E6F4EA", color: str = "#0B5D1E") -> str:
    return (
        f'<td style="background:{bg}; color:{color}; font-weight:bold; '
        f'border:1px solid #AAAAAA; padding:5px; vertical-align:top;">{escape(text)}</td>'
    )


def main() -> None:
    if DOCX_PATH.exists() and not STYLE_BACKUP.exists():
        shutil.copy2(DOCX_PATH, STYLE_BACKUP)

    v2_tw = pick_first(read_csv(ROOT / "outputs/decision_revision_v2/locked_test_evaluation_taiwan.csv"))
    v2_he = pick_first(read_csv(ROOT / "outputs/decision_revision_v2/locked_test_evaluation_heloc.csv"))
    fa_tw = pick_first(
        read_csv(ROOT / "outputs/final_attempt/locked_test/final_locked_test_evaluation_taiwan.csv"),
        "operational",
        "system_name",
    )
    fa_he = pick_first(
        read_csv(ROOT / "outputs/final_attempt/locked_test/final_locked_test_evaluation_heloc.csv"),
        "operational",
        "system_name",
    )

    lit_df = read_csv(ROOT / "outputs/final_attempt/literature_reproduction/literature_reproduction_best_configs.csv")
    adv_df = read_csv(ROOT / "outputs/final_attempt/imbalance_boosting/advanced_imbalance_best_configs.csv")
    deep_df = read_csv(ROOT / "outputs/final_attempt/deep_tabular/deep_tabular_best_configs.csv")
    interp_df = read_csv(ROOT / "outputs/final_attempt/interpretable_models/interpretable_models_best_configs.csv")
    cap_df = read_csv(ROOT / "outputs/final_attempt/capacity_review/capacity_model_comparison.csv")
    dca_df = read_csv(ROOT / "outputs/final_attempt/decision_curve/decision_curve_best_ranges.csv")
    sel_tw_df = read_csv(ROOT / "outputs/final_attempt/final_selection/final_multitrack_selection_taiwan.csv")
    sel_he_df = read_csv(ROOT / "outputs/final_attempt/final_selection/final_multitrack_selection_heloc.csv")

    lit_tw = best_by_dataset(lit_df, "taiwan", rank_col="rank_by_validation_pr_auc")
    lit_he = best_by_dataset(lit_df, "heloc", rank_col="rank_by_validation_pr_auc")
    adv_tw = best_by_dataset(adv_df, "taiwan", rank_col="rank_validation_operational")
    adv_he = best_by_dataset(adv_df, "heloc", rank_col="rank_validation_operational")
    deep_tw = best_by_dataset(deep_df, "taiwan", rank_col="rank_validation_deep")
    deep_he = best_by_dataset(deep_df, "heloc", rank_col="rank_validation_deep")
    interp_tw = best_by_dataset(interp_df, "taiwan", rank_col="rank_validation")
    interp_he = best_by_dataset(interp_df, "heloc", rank_col="rank_validation")
    cap_tw = best_by_dataset(cap_df, "taiwan", sort_col="top_20_capture", ascending=False, split_col="split", split_val="validation")
    cap_he = best_by_dataset(cap_df, "heloc", sort_col="top_20_capture", ascending=False, split_col="split", split_val="validation")
    dca_tw = best_by_dataset(dca_df, "taiwan", sort_col="max_net_benefit", ascending=False, split_col="split", split_val="validation")
    dca_he = best_by_dataset(dca_df, "heloc", sort_col="max_net_benefit", ascending=False, split_col="split", split_val="validation")

    section1_rows = [
        ("Done", "Data loading & exploration", "No scope change this week. Taiwan remains the primary dataset and HELOC remains the external validation dataset. German Credit / Option A remains archived and out of active scope."),
        ("Done", "Data cleaning", "No new dataset was added and no target-derived feature was created. The cleaned Taiwan and HELOC datasets from the reproducible pipeline remain the basis for all experiments."),
        ("Done", "Preprocessing implementation", "The V2 protocol protects the test split: model/policy selection is validation-only; test is used only after locking. No split-before-resampling issue was found in the final audit."),
        ("Done", "Feature engineering review", "Existing row-wise Taiwan and HELOC features were kept frozen. The final-attempt experiments did not introduce new features, so the earlier leakage controls remain intact."),
        ("Done", "Model training / model-policy evaluation", "Evaluated V2 locked policies, strict literature reproduction, advanced imbalance boosting, deep tabular models, interpretable middle models, capacity-aware review, and decision curves."),
        ("Done", "Validation protocol repair", "Fixed the previous NOT READY problem by creating selection_protocol_v2, validation-only candidate registries, locked final policies, and separate held-out test evidence tables."),
        ("Done", "Manual-review cost model", "Separated binary FN/FP expected cost from review-adjusted operational cost. Manual-review workload and review-cost assumptions are now explicitly documented."),
        ("Done", "False-positive reduction", "Completed precision-constrained thresholds, cost-matrix sensitivity, manual-review band optimization, eligible model selection, and false-positive profile analysis."),
        ("Done", "Literature / advanced reproduction", "Tested SMOTE, BorderlineSMOTE, SMOTE-Tomek, KMeansSMOTE, class weighting, MLP/BP neural networks, EBM, and monotonic models under leakage-free protocol."),
        ("Partial", "Final-attempt audit", "No test-selection leakage was detected, but final locked-test manual-review rows failed metric-consistency identities. V2 remains report-ready; final attempt needs repair before replacing V2."),
    ]

    section1 = ['<table class="tasks">', "<tr>" + th("☐") + th("Task") + th("Details / Evidence") + "</tr>"]
    for status, task, detail in section1_rows:
        if status == "Done":
            section1.append("<tr>" + status_cell("Done") + td(task) + td(detail) + "</tr>")
        else:
            section1.append("<tr>" + status_cell("Partial", "#FFF8E1", "#7A3E00") + td(task) + td(detail) + "</tr>")
    section1.append("</table>")

    v2_evidence = [
        ["Taiwan V2 locked policy", "Best CatBoost manual-review band", fmt(v2_tw.get("test_accuracy")), fmt(v2_tw.get("test_precision")), fmt(v2_tw.get("test_recall")), fmt(v2_tw.get("test_specificity")), fmt(v2_tw.get("test_f1")), fmt(v2_tw.get("test_roc_auc")), fmt(v2_tw.get("test_pr_auc")), fmt(v2_tw.get("test_review_adjusted_cost")), "Report-ready baseline."],
        ["HELOC V2 locked policy", "Best Scorecard manual-review band", fmt(v2_he.get("test_accuracy")), fmt(v2_he.get("test_precision")), fmt(v2_he.get("test_recall")), fmt(v2_he.get("test_specificity")), fmt(v2_he.get("test_f1")), fmt(v2_he.get("test_roc_auc")), fmt(v2_he.get("test_pr_auc")), fmt(v2_he.get("test_review_adjusted_cost")), "Report-ready external-validation baseline."],
        ["Taiwan final attempt", "XGBoost manual-review", fmt(fa_tw.get("test_accuracy")), fmt(fa_tw.get("test_precision")), fmt(fa_tw.get("test_recall")), fmt(fa_tw.get("test_specificity")), fmt(fa_tw.get("test_f1")), fmt(fa_tw.get("test_roc_auc")), fmt(fa_tw.get("test_pr_auc")), fmt(fa_tw.get("test_review_adjusted_cost")), "Diagnostic only; audit blocked metric consistency."],
        ["HELOC final attempt", "Scorecard manual-review", fmt(fa_he.get("test_accuracy")), fmt(fa_he.get("test_precision")), fmt(fa_he.get("test_recall")), fmt(fa_he.get("test_specificity")), fmt(fa_he.get("test_f1")), fmt(fa_he.get("test_roc_auc")), fmt(fa_he.get("test_pr_auc")), fmt(fa_he.get("test_review_adjusted_cost")), "Diagnostic only; same policy family as V2."],
    ]

    old_vs_v2 = [
        ["Taiwan old aggressive CatBoost", "Cost-threshold binary policy", "0.351", "0.809", "0.576", "1982", "253", "3247", "High recall but too many false positives."],
        ["Taiwan V2 CatBoost", "Manual-review band", fmt(v2_tw.get("test_precision")), fmt(v2_tw.get("test_recall")), fmt(v2_tw.get("test_specificity")), fmt_int(v2_tw.get("test_fp")), fmt_int(v2_tw.get("test_fn")), fmt(v2_tw.get("test_review_adjusted_cost")), "FP reduced by 1302 vs old CatBoost; precision and specificity improved."],
        ["HELOC old aggressive CatBoost reference", "Cost-threshold binary policy", "0.558", "0.981", "0.157", "798", "20", "898", "Very high recall but low specificity."],
        ["HELOC V2 Scorecard", "Manual-review band", fmt(v2_he.get("test_precision")), fmt(v2_he.get("test_recall")), fmt(v2_he.get("test_specificity")), fmt_int(v2_he.get("test_fp")), fmt_int(v2_he.get("test_fn")), fmt(v2_he.get("test_review_adjusted_cost")), "FP reduced by 395 vs old reference; better operational balance."],
    ]

    exp_summary = [
        ["Strict literature reproduction - Taiwan", lit_tw.get("config_display_name", "Not available"), fmt(lit_tw.get("validation_pr_auc")), fmt(lit_tw.get("validation_roc_auc")), fmt(lit_tw.get("validation_precision_cost_policy")), fmt(lit_tw.get("validation_recall_cost_policy")), fmt(lit_tw.get("validation_best_specificity_at_cost_policy")), fmt(lit_tw.get("validation_expected_cost_cost_policy")), "No reliable V2 replacement."],
        ["Strict literature reproduction - HELOC", lit_he.get("config_display_name", "Not available"), fmt(lit_he.get("validation_pr_auc")), fmt(lit_he.get("validation_roc_auc")), fmt(lit_he.get("validation_precision_cost_policy")), fmt(lit_he.get("validation_recall_cost_policy")), fmt(lit_he.get("validation_best_specificity_at_cost_policy")), fmt(lit_he.get("validation_expected_cost_cost_policy")), "High literature-style jump not reproduced."],
        ["Advanced imbalance - Taiwan", f"{adv_tw.get('model_family', '')} {adv_tw.get('variant_name', '')}", fmt(adv_tw.get("validation_pr_auc")), fmt(adv_tw.get("validation_roc_auc")), fmt(adv_tw.get("validation_precision")), fmt(adv_tw.get("validation_recall")), fmt(adv_tw.get("validation_specificity")), fmt(adv_tw.get("validation_review_adjusted_cost")), "Useful validation candidate; test evidence did not beat V2."],
        ["Advanced imbalance - HELOC", f"{adv_he.get('model_family', '')} {adv_he.get('variant_name', '')}", fmt(adv_he.get("validation_pr_auc")), fmt(adv_he.get("validation_roc_auc")), fmt(adv_he.get("validation_precision")), fmt(adv_he.get("validation_recall")), fmt(adv_he.get("validation_specificity")), fmt(adv_he.get("validation_review_adjusted_cost")), "Scorecard remains safer for HELOC."],
        ["Deep tabular - Taiwan", deep_tw.get("config_id", "Not available"), fmt(deep_tw.get("validation_pr_auc")), fmt(deep_tw.get("validation_roc_auc")), fmt(deep_tw.get("validation_precision")), fmt(deep_tw.get("validation_recall")), fmt(deep_tw.get("validation_specificity")), fmt(deep_tw.get("validation_expected_cost")), "DNN did not deliver better operational policy."],
        ["Deep tabular - HELOC", deep_he.get("config_id", "Not available"), fmt(deep_he.get("validation_pr_auc")), fmt(deep_he.get("validation_roc_auc")), fmt(deep_he.get("validation_precision")), fmt(deep_he.get("validation_recall")), fmt(deep_he.get("validation_specificity")), fmt(deep_he.get("validation_expected_cost")), "Raw PR-AUC useful, but not final operational winner."],
        ["Interpretable - Taiwan", f"{interp_tw.get('model_family', '')} {interp_tw.get('model', '')}", fmt(interp_tw.get("validation_pr_auc")), fmt(interp_tw.get("validation_roc_auc")), fmt(interp_tw.get("validation_precision")), fmt(interp_tw.get("validation_recall")), fmt(interp_tw.get("validation_specificity")), fmt(interp_tw.get("validation_manual_review_adjusted_cost")), "Governance-friendly alternative."],
        ["Interpretable - HELOC", f"{interp_he.get('model_family', '')} {interp_he.get('model', '')}", fmt(interp_he.get("validation_pr_auc")), fmt(interp_he.get("validation_roc_auc")), fmt(interp_he.get("validation_precision")), fmt(interp_he.get("validation_recall")), fmt(interp_he.get("validation_specificity")), fmt(interp_he.get("validation_manual_review_adjusted_cost")), "Scorecard/EBM remain strong interpretable baselines."],
    ]

    capacity_rows = [
        ["Taiwan", cap_tw.get("model", "Not available"), fmt(cap_tw.get("top_10_capture")), fmt(cap_tw.get("top_20_capture")), fmt(cap_tw.get("top_30_capture")), fmt(cap_tw.get("precision_at_20")), fmt(cap_tw.get("lift_at_20")), "Good framing for review prioritization."],
        ["HELOC", cap_he.get("model", "Not available"), fmt(cap_he.get("top_10_capture")), fmt(cap_he.get("top_20_capture")), fmt(cap_he.get("top_30_capture")), fmt(cap_he.get("precision_at_20")), fmt(cap_he.get("lift_at_20")), "Useful external validation for ranking."],
    ]

    dca_rows = [
        ["Taiwan", dca_tw.get("model", "Not available"), dca_tw.get("useful_threshold_range", "Not available"), fmt(dca_tw.get("max_net_benefit")), fmt(dca_tw.get("max_net_benefit_threshold")), "Supports screening/manual-review use."],
        ["HELOC", dca_he.get("model", "Not available"), dca_he.get("useful_threshold_range", "Not available"), fmt(dca_he.get("max_net_benefit")), fmt(dca_he.get("max_net_benefit_threshold")), "Models provide benefit over naive strategies in useful ranges."],
    ]

    selection_rows: list[list[object]] = []
    for df, dataset in [(sel_tw_df, "Taiwan"), (sel_he_df, "HELOC")]:
        if not df.empty:
            for _, row in df.iterrows():
                note = row.get("why", "") or row.get("selection_reason", "") or "Selected using validation-only evidence."
                selection_rows.append([dataset, row.get("track_name", ""), row.get("selected_system", ""), note])

    comparison_rows = [
        ["Alam et al. (2020)", "GBDT + K-means SMOTE on Taiwan", "~0.887", "Not reported", "Not reported", "Reported high accuracy", "Not matched", "Week 13 leakage-free reproduction did not show a comparable jump."],
        ["K-means SMOTE + BP NN (2021)", "Class imbalance handling + neural net", "Not reported", "Not reported", "0.929", "AUC improved 0.765 to 0.929", "Not matched", "KMeansSMOTE and MLP/BP reproduction did not produce a reliable operational breakthrough."],
        ["Interpretable DL + XAI (2023)", "Deep model with XAI", "0.835", "Not reported", "Not reported", "Sensitivity 0.882; specificity 0.988", "Not claimed", "Current project remains more conservative and audit-focused."],
        ["DNN empirical study (2025)", "Deep neural network on Taiwan", "0.818", "0.482", "0.770", "G-mean 0.603", "Partial context", "Comparable to some raw metrics, but not enough to justify automatic rejection."],
        ["Week 12 old aggressive CatBoost", "Cost-sensitive binary threshold", "0.627", "0.490", "0.785", "Precision 0.351; recall 0.809; PR-AUC 0.564; cost 3247", "Problem baseline", "High recall but excessive false positives."],
        ["Week 13 V2 Taiwan", "CatBoost manual-review band", fmt(v2_tw.get("test_accuracy")), fmt(v2_tw.get("test_f1")), fmt(v2_tw.get("test_roc_auc")), f"PR-AUC {fmt(v2_tw.get('test_pr_auc'))}; precision {fmt(v2_tw.get('test_precision'))}; specificity {fmt(v2_tw.get('test_specificity'))}; review-adjusted cost {fmt(v2_tw.get('test_review_adjusted_cost'))}", "Best safe Taiwan baseline", "Report-ready and more operationally defensible."],
        ["Week 13 V2 HELOC", "Scorecard manual-review band", fmt(v2_he.get("test_accuracy")), fmt(v2_he.get("test_f1")), fmt(v2_he.get("test_roc_auc")), f"PR-AUC {fmt(v2_he.get('test_pr_auc'))}; precision {fmt(v2_he.get('test_precision'))}; specificity {fmt(v2_he.get('test_specificity'))}; review-adjusted cost {fmt(v2_he.get('test_review_adjusted_cost'))}", "Best safe HELOC baseline", "Interpretable and externally useful."],
        ["SCRE-Credit role", "Reliability-aware framework / ranking component", "0.645", "0.492", "0.785", "SCRE-Optimized test PR-AUC 0.567; cost 3312 under FN=5/FP=1", "Framework, not universal winner", "Useful for reliability-aware comparison and ranking, not a dominance claim."],
    ]

    problem_rows = [
        ["Final-attempt audit BLOCKED", "The audit found 9 critical metric-consistency failures in final locked-test manual-review rows. Recall, specificity, and F1 were not always recomputable from the displayed confusion counts.", "Ongoing - repair needed before final-attempt tables can replace V2."],
        ["V2 vs final-attempt decision", "Final-attempt Taiwan XGBoost improved recall slightly but worsened precision, specificity, FP count, and review-adjusted cost compared with V2.", "Ongoing - V2 remains the safer report baseline."],
        ["Manual-review denominator complexity", "Manual-review policies mix full-population metrics, auto-decision-only metrics, and review workload cost. These must be separated in final tables.", "Ongoing - next technical repair."],
        ["Automatic rejection risk", "Precision and false-positive behavior are not strong enough for direct automatic credit rejection.", "Solved by safe screening/manual-review wording."],
        ["Literature score gap", "Leakage-free reproduction did not match very high published scores. Likely reasons include protocol differences, resampled test distributions, threshold choices, or weaker leakage controls in some comparisons.", "Solved as methodological discussion."],
    ]

    next_rows = [
        ["1", "Repair final-attempt locked-test manual-review metric tables.", "Separate full-population metrics, auto-decision metrics, manual-review counts, review workload cost, and binary diagnostic cost."],
        ["2", "Re-run final-attempt brutal audit after the table repair.", "Updated final_attempt_audit.md showing whether the BLOCKED status is resolved."],
        ["3", "Choose final report baseline.", "Use V2 as the main report result unless the repaired final attempt clearly passes audit and improves operational balance."],
        ["4", "Draft final project report sections.", "Methods, validation protocol, datasets, model comparison, manual-review policy, capacity analysis, limitations, and safe claims."],
        ["5", "Prepare final figures and appendix tables.", "Manual-review workflow, capacity@K/lift curves, decision curve, V2 locked results, claim-control matrix, and audit summary."],
    ]

    self_rows = [
        ["Overall project completion (%)", "Approximately 90% if V2 is used as the final report baseline; approximately 80-85% if the final-attempt package must become the main result."],
        ["Did you complete last week's planned tasks? (Yes / Partially / No)", "Yes. Precision-constrained thresholds, cost sensitivity, decision curve analysis, training-level imbalance testing, and manual-review optimization were all implemented."],
        ["If not, why?", "The planned experiments were completed. The remaining issue is not missing experimentation; it is final-attempt metric-consistency cleanup before those results can replace V2."],
        ["Are you on track to finish the project on time?", "Yes, if the final report uses V2 as the report-ready baseline and the final attempt is either repaired or presented as a stress-test appendix."],
        ["Do you need help from the instructor? If yes, on what?", "Potentially yes: feedback on whether the report should emphasize the V2 READY policy or include the final-attempt stress tests as a secondary appendix."],
    ]

    milestone_rows = [
        ["1", "Data Loading & EDA", "Load dataset, check shape/types/missing values, compute basic statistics, create distribution plots", "Completed earlier"],
        ["2", "Data Cleaning & Preprocessing", "Handle missing values, duplicates, category cleaning, scaling, train-only preprocessing", "Completed earlier"],
        ["3", "Feature Engineering", "Create domain-specific features and final feature set", "Completed earlier and frozen"],
        ["4", "Baseline Model", "Train LR/RF/boosting/scorecard baselines", "Completed earlier"],
        ["5", "Imbalance Handling & Ablation", "Compare sampling, cost-sensitive, and ablation variants", "Completed and extended this week"],
        ["6", "Proposed Model Implementation", "Implement SCRE-Credit and compare with baselines", "Completed earlier; role refined this week"],
        ["7", "Hyperparameter Optimization", "Run Optuna / tuning and compare before/after", "Completed earlier"],
        ["8", "Explainability & Analysis", "Run SHAP/LIME, stability, faithfulness, subgroup checks", "Completed earlier; stability claims controlled"],
        ["9", "Final Evaluation & Writing", "Final hold-out evaluation, audit, figures, report writing", "Current focus"],
        ["10", "Final Report Submission", "Write discussion, conclusion, limitations, future work", "Next step after final metric cleanup"],
    ]

    grading_rows = [
        ["Progress Evidence", "30%", "Concrete artifacts, tables, figures, and audit results are included."],
        ["Results & Comparison", "30%", "Numerical results are compared with proposal literature and current V2/final-attempt evidence."],
        ["Critical Thinking", "20%", "The report explains why V2 remains safer and why final attempt is blocked."],
        ["Planning & Commitment", "20%", "Next-week tasks are specific and measurable."],
    ]

    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Week 13 Progress Report</title><style>
@page {{ size: A4; margin: 0.62in; }}
body {{ font-family: Calibri, Arial, sans-serif; font-size: 10.5pt; color: #111; line-height: 1.22; }}
h1 {{ font-size: 16pt; text-align: center; margin: 10px 0 12px; font-weight: bold; color:#1F3864; }}
h2 {{ font-size: 12pt; margin: 15px 0 6px; font-weight: bold; color:#1F3864; border-bottom: 1px solid #2F5496; }}
p {{ margin: 5px 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 6px 0 10px; }}
th, td {{ border: 1px solid #AAAAAA; padding: 5px; vertical-align: top; }}
.header {{ text-align: center; color:#1F3864; font-size: 11pt; }}
.instructions th {{ background:#FFF8E1 !important; color:#1F3864 !important; border:1px solid #2F5496; }}
.instructions td {{ background:#FFF8E1; color:#111; border:1px solid #2F5496; }}
.student td:nth-child(1), .student td:nth-child(3) {{ background:#E8EEF7; color:#1F3864; font-weight:bold; }}
.student td:nth-child(2), .student td:nth-child(4) {{ background:white; }}
.evidence tr:nth-child(even) td {{ background:#F2F7FC; }}
.leader tr:nth-child(even) td {{ background:#F2F7FC; }}
.warning {{ background:#FCE5CD; border:1px solid #CC0000; padding:7px; color:#111; }}
.small {{ font-size: 9pt; color:#666666; }}
</style></head><body>
<div class="header"><strong>ECE 565 - DATA MINING</strong><br>Weekly Progress Report Template<br>Deadline: Every Tuesday by 11:59 PM<br>Spring 2026<br>Instructor: Asst. Prof. Dr. Asli Eyecioglu Ozmutlu</div>
<br>
<table class="instructions">
<tr><th style="background:#1F3864; color:white; font-weight:bold; border:1px solid #AAAAAA; padding:5px; vertical-align:top;">INSTRUCTIONS</th></tr>
<tr><td style="background:white; color:#111111; font-weight:normal; border:1px solid #AAAAAA; padding:5px; vertical-align:top;">1. Fill out this report every week with your project progress.<br>2. Submit as PDF (StudentID_Surname_Week#.pdf) on Canvas by Tuesday 11:59 PM.<br>3. Include actual results (numbers, tables, figures) — not just descriptions of what you plan to do.<br>4. Compare your results against the papers from your proposal. Use the comparison table provided.<br>5. Be honest about problems. If something did not work, explain why and what you tried.<br>6. The 'Next Week Plan' section is your commitment — I will check it against the following week's report.</td></tr>
</table>
<h1>WEEKLY PROGRESS REPORT</h1>
<table class="student">
<tr>{td('Student Name:')}{td('Resul Ozkale')}{td('Student ID:')}{td('2540041007')}</tr>
<tr>{td('Week Number:')}{td('Week 13 - Validation Protocol Repair and Final Attempt Stress Testing')}{td('Date:')}{td('May 12, 2026')}</tr>
<tr>{td('Project Title:')}{td('SCRE-Credit: Stability- and Cost-Regularized Ensemble for Credit Risk')}{td('')}{td('')}</tr>
<tr>{td('Dataset:')}{td('Primary: UCI Default of Credit Card Clients / Taiwan. External validation: FICO HELOC. German Credit / Option A is archived and out of active scope.')}{td('')}{td('')}</tr>
</table>
<h2>SECTION 1: Work Completed This Week</h2>
<p>Describe concretely what you accomplished this week. Use the checklist below and provide details for each completed item.</p>
{''.join(section1)}
<h2>SECTION 1b: Show Your Work</h2>
<p>Paste or attach at least one piece of evidence from this week: a code output, a plot, a table of results, a screenshot, or a confusion matrix. Do NOT submit a report with only text and no evidence.</p>
{table(['Evidence', 'Model / Policy', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'ROC-AUC', 'PR-AUC', 'Cost', 'Comment'], v2_evidence)}
<p class="warning"><strong>Important transparency note:</strong> V2 is currently the report-ready result. The later final-attempt package is useful and methodologically informative, but its final locked-test manual-review table is blocked by metric-consistency audit failures. No test-selection leakage was detected.</p>
<h2>Additional Evidence: Old Aggressive Threshold vs V2 Manual-Review Policy</h2>
{table(['Dataset / system', 'Policy', 'Precision', 'Recall', 'Specificity', 'FP', 'FN', 'Cost', 'Comment'], old_vs_v2)}
<h2>Additional Evidence: Final-Attempt Experiment Summary</h2>
{table(['Experiment', 'Best validation config / model', 'PR-AUC', 'ROC-AUC', 'Precision', 'Recall', 'Specificity', 'Cost', 'Interpretation'], exp_summary)}
<h2>Additional Evidence: Capacity-Aware Manual Review</h2>
{table(['Dataset', 'Best validation ranking model', 'Capture@10%', 'Capture@20%', 'Capture@30%', 'Precision@20%', 'Lift@20%', 'Comment'], capacity_rows)}
<h2>Additional Evidence: Decision Curve / Net Benefit</h2>
{table(['Dataset', 'Best validation model', 'Useful threshold range', 'Max net benefit', 'Max NB threshold', 'Comment'], dca_rows)}
<h2>Additional Evidence: Final Multi-Track Selection</h2>
{table(['Dataset', 'Track', 'Selected system', 'Reason / note'], selection_rows[:14])}
<h2>SECTION 2: Preliminary Results & Literature Comparison</h2>
<p>Report your current numerical results and compare them against the papers from your proposal. Fill in the table below.</p>
{table(['Model / Paper', 'Method', 'Accuracy', 'F1-Score', 'AUC-ROC', 'Other Metric', 'Best?', 'Notes'], comparison_rows)}
<p class="small">Note: Literature rows use “Not reported” only when that exact metric was not reported in the referenced proposal source. Project rows use actual values from generated output tables.</p>
<h2>SECTION 2b: Results Discussion</h2>
<p><strong>Q1:</strong> Compared with Week 12, the most important improvement is methodological. Final model and policy selection is now validation-only, and held-out test results are reported only after policy locking.</p>
<p><strong>Q2:</strong> The Week 12 problem was that the cost-optimal CatBoost threshold was too aggressive: high recall, low precision, and many false positives. Week 13 addressed this by moving toward constrained thresholding and manual-review policies.</p>
<p><strong>Q3:</strong> The V2 Taiwan policy is more operationally defensible than the old cost-threshold policy because precision increased from 0.351 to {fmt(v2_tw.get('test_precision'))}, specificity increased from 0.576 to {fmt(v2_tw.get('test_specificity'))}, and false positives fell from 1982 to {fmt_int(v2_tw.get('test_fp'))}.</p>
<p><strong>Q4:</strong> Strict literature reproduction did not produce the very high scores reported in some papers when resampling was kept inside training folds and the test distribution remained natural. This supports a careful, leakage-aware discussion rather than a claim of state-of-the-art accuracy.</p>
<p><strong>Q5:</strong> SCRE-Credit should be retained as a reliability-aware framework and review-ranking component. It should not be presented as a model that beats CatBoost, Scorecard, XGBoost, and LightGBM in every metric.</p>
<h2>SECTION 3: Challenges & Problems Encountered</h2>
<p>Describe any problems you faced this week. Be specific about error messages, unexpected results, or conceptual difficulties.</p>
{table(['Problem', 'What I Tried / Found', 'Status (Solved / Ongoing)'], problem_rows)}
<h2>SECTION 4: Plan for Next Week</h2>
<p>List 3-5 specific, measurable tasks you will complete by next Tuesday. These will be checked against your next report.</p>
{table(['#', 'Planned Task (be specific)', 'Expected Deliverable'], next_rows)}
<h2>SECTION 5: Self-Assessment</h2>
{table(['Question', 'Your Answer'], self_rows)}
<h2>SUGGESTED WEEKLY MILESTONES</h2>
<p>Use this as a guide for pacing your project. We assume you have already done 1-5. Your weekly reports should roughly follow this timeline.</p>
{table(['Week', 'Focus Area', 'Expected Tasks', 'Current status in this project'], milestone_rows)}
<h2>WEEKLY REPORT GRADING CRITERIA</h2>
{table(['Criterion', 'Weight', 'How this report addresses it'], grading_rows)}
<h2>IMPORTANT WARNINGS</h2>
<p class="warning">Reports with no numerical results or evidence receive no credit for progress evidence. This report includes locked V2 metrics, final-attempt diagnostic metrics, literature comparison, capacity-review results, decision-curve evidence, and audit status. The safe final claim is: the project supports credit-risk screening and manual-review prioritization, not automatic credit rejection. V2 is the current defensible report-ready baseline; the final-attempt experiments need one more metric-consistency repair before they can replace V2 as the main final result.</p>
<p class="small">GitHub/code reference: godsoftware / Data_Mining_Project. Main local evidence folders: outputs/decision_revision_v2/ and outputs/final_attempt/.</p>
</body></html>'''

    HTML_PATH.write_text(html, encoding="utf-8")
    if ">NA<" in html or ">N/A<" in html:
        raise RuntimeError("Unexpected NA marker remained in HTML")
    print(HTML_PATH)


if __name__ == "__main__":
    main()
