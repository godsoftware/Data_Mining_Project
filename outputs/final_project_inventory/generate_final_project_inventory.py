from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"C:\Users\AKTS\Desktop\Resul\Data_Mining_Project")
OUT = ROOT / "outputs" / "final_project_inventory"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        if isinstance(value, str) and value.strip() == "":
            return ""
        if isinstance(value, str):
            return value
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def as_int(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
        return str(int(round(float(value))))
    except Exception:
        return fmt(value)


def first_row(df: pd.DataFrame, contains: str | None = None, col: str | None = None) -> dict[str, Any]:
    if df.empty:
        return {}
    if contains and col and col in df.columns:
        mask = df[col].astype(str).str.contains(contains, case=False, na=False)
        if mask.any():
            return df[mask].iloc[0].to_dict()
    return df.iloc[0].to_dict()


def best_by_dataset(
    df: pd.DataFrame,
    dataset: str,
    rank_col: str | None = None,
    sort_col: str | None = None,
    ascending: bool = True,
    split_col: str | None = None,
    split_val: str | None = None,
    model_contains: str | None = None,
) -> dict[str, Any]:
    if df.empty or "dataset" not in df.columns:
        return {}
    sub = df[df["dataset"].astype(str).str.lower() == dataset.lower()].copy()
    if split_col and split_val and split_col in sub.columns:
        sub = sub[sub[split_col].astype(str).str.lower() == split_val.lower()]
    if model_contains and "model" in sub.columns:
        sub = sub[sub["model"].astype(str).str.contains(model_contains, case=False, na=False)]
    if sub.empty:
        return {}
    if rank_col and rank_col in sub.columns:
        sub = sub.sort_values(rank_col, ascending=True)
    elif sort_col and sort_col in sub.columns:
        sub = sub.sort_values(sort_col, ascending=ascending)
    return sub.iloc[0].to_dict()


def source_stage(path: Path) -> str:
    s = rel(path).lower()
    if "decision_revision_v2" in s:
        return "V2"
    if "final_attempt/audit" in s or "audit" in path.name.lower():
        return "Audit"
    if "final_attempt/literature_reproduction" in s:
        return "Literature Reproduction"
    if "final_attempt/capacity_review" in s:
        return "Capacity Review"
    if "final_attempt/decision_curve" in s:
        return "Decision Curve"
    if "calibration" in s:
        return "Calibration"
    if "threshold" in s:
        return "Threshold"
    if "manual_review" in s or "manual" in s:
        return "Manual Review"
    if "scre" in path.name.lower():
        return "Original SCRE"
    if "final_attempt" in s:
        return "Final Attempt"
    if "explainability" in s or "shap" in s or "lime" in s or "faithfulness" in s:
        return "Explainability"
    return "Unknown"


def safety_for_file(path: Path, columns: list[str], text: str) -> tuple[bool, bool, str, str]:
    s = rel(path).lower()
    name = path.name.lower()
    if "final_attempt/locked_test/final_locked_test_evaluation" in s:
        return False, False, "Final Attempt locked-test manual-review rows have metric-consistency/auditability failures; do not use as final evidence.", "DO NOT USE as final."
    if "final_attempt_audit" in name:
        return False, True, "Audit verdict is BLOCKED; file is useful only to explain why Final Attempt cannot be final.", "Use as appendix/audit explanation."
    if "retrospective_all_candidate_test_diagnostics" in name:
        return False, True, "Retrospective diagnostic file explicitly not for model selection.", "Diagnostic only."
    if "decision_revision_v2" in s and ("locked_test_evaluation" in name or "final_audit_v2" in name or "final_validation_selection_table" in name or "final_heldout_test_evidence_table" in name or "final_safe_recommendation" in name):
        return True, True, "", "V2 audit-clean evidence."
    if "decision_revision_v2" in s:
        return False, True, "", "V2 supporting artifact."
    if "final_attempt" in s:
        return False, True, "", "Final Attempt stress-test/robustness artifact; not final operational evidence."
    if "outputs/tables" in s and ("final_decision" in name or "claim_control" in name or "scre_revision" in name or "final_leaderboard" in name):
        return False, True, "Earlier pre-V2 result; use only with protocol caveat.", "Supporting historical artifact."
    return False, True, "", "Supporting or inventory artifact."


def file_inventory() -> pd.DataFrame:
    scan_roots = [
        ROOT / "outputs" / "decision_revision_v2",
        ROOT / "outputs" / "final_attempt",
        ROOT / "outputs" / "tables",
        ROOT / "outputs" / "models",
    ]
    exts = {".csv", ".md", ".json", ".xlsx", ".joblib", ".png", ".log"}
    files: list[Path] = []
    for base in scan_roots:
        if base.exists():
            files.extend([p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    rows = []
    for path in sorted(set(files)):
        columns: list[str] = []
        row_count: Any = ""
        col_count: Any = ""
        text = ""
        if path.suffix.lower() == ".csv":
            df = read_csv(path)
            row_count = len(df) if not df.empty else 0
            col_count = len(df.columns) if not df.empty else 0
            columns = list(df.columns)
        elif path.suffix.lower() in {".md", ".log", ".json"}:
            text = read_text(path)
            row_count = len(text.splitlines())
            col_count = ""
            if path.suffix.lower() == ".json":
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        columns = list(obj.keys())
                        col_count = len(columns)
                except Exception:
                    pass
        elif path.suffix.lower() == ".xlsx":
            row_count = ""
            col_count = ""
        all_names = " ".join(columns).lower() + " " + text[:5000].lower()
        contains_validation = "validation" in all_names
        contains_test = "test_" in all_names or "held-out" in all_names or "heldout" in all_names
        contains_rank = any(("rank" in c.lower() or "winner" in c.lower()) for c in columns) or " winner" in text.lower()
        contains_audit = "audit" in rel(path).lower() or "verdict" in all_names or "critical" in all_names
        safe_main, safe_appendix, unsafe_reason, notes = safety_for_file(path, columns, text)
        rows.append(
            {
                "file_path": rel(path),
                "exists": True,
                "source_stage": source_stage(path),
                "row_count": row_count,
                "column_count": col_count,
                "contains_validation_metrics": contains_validation,
                "contains_test_metrics": contains_test,
                "contains_rank_or_winner": contains_rank,
                "contains_audit_status": contains_audit,
                "safe_for_main_report": safe_main,
                "safe_for_appendix": safe_appendix,
                "unsafe_reason": unsafe_reason,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def parse_audit(path: Path, stage: str) -> dict[str, Any]:
    text = read_text(path)
    verdict = "UNKNOWN"
    m = re.search(r"Verdict:\s*\**([A-Z ]+)\**", text, re.I)
    if m:
        verdict = m.group(1).strip().upper()
    elif "READY" in text[:1000].upper():
        verdict = "READY"
    elif "BLOCKED" in text[:1000].upper():
        verdict = "BLOCKED"
    critical = 0
    high = 0
    medium = 0
    for label, key in [("critical", "critical"), ("high", "high"), ("medium", "medium")]:
        m = re.search(label + r"(?: failures| issues/warnings| issues| warnings)?\s*:\s*(\d+)", text, re.I)
        if m:
            if key == "critical":
                critical = int(m.group(1))
            elif key == "high":
                high = int(m.group(1))
            else:
                medium = int(m.group(1))
    if "no critical" in text.lower():
        critical = 0
    if "no high" in text.lower():
        high = 0
    if "no medium" in text.lower() or "medium warnings: 0" in text.lower():
        medium = 0
    test_leakage = "test-selection leakage" in text.lower() and "no detected" not in text.lower()
    if "test leakage found?\nyes" in text.lower():
        test_leakage = True
    test_ranking = "test ranking" in text.lower() and "removed" not in text.lower() and "no test ranking" not in text.lower()
    metric_fail = "metric-consistency" in text.lower() or "metric consistency" in text.lower()
    if "metric consistency status" in text.lower() and "pass" in text.lower():
        metric_fail = False
    ready = verdict == "READY" and critical == 0
    blocker = ""
    for line in text.splitlines():
        if "Blocking issue" in line or "Main blocker" in line:
            blocker = line.strip("- ").strip()
            break
    if not blocker and verdict == "BLOCKED":
        blocker = "Audit verdict is BLOCKED; see audit file."
    return {
        "audit_file": rel(path),
        "stage": stage,
        "verdict": verdict,
        "critical_count": critical,
        "high_count": high,
        "medium_count": medium,
        "ready_for_report": ready,
        "test_leakage_detected": test_leakage,
        "test_ranking_detected": test_ranking,
        "metric_consistency_failures": metric_fail and verdict == "BLOCKED",
        "main_blocker": blocker,
        "safe_to_use": ready or "final_attempt_audit" in path.name.lower(),
        "notes": "V2 can support final main evidence." if stage == "V2" and ready else ("Final Attempt explains why that stage is appendix/diagnostic, not final." if "final_attempt" in rel(path).lower() else ""),
    }


def audit_summary() -> pd.DataFrame:
    candidates = [
        (ROOT / "outputs/decision_revision_v2/final_audit_v2.md", "V2"),
        (ROOT / "outputs/final_attempt/audit/final_attempt_audit.md", "Final Attempt"),
        (ROOT / "outputs/final_revision_audit.md", "Original SCRE"),
        (ROOT / "outputs/decision_revision/final_decision_revision_audit.md", "Decision Revision V1"),
    ]
    rows = []
    for path, stage in candidates:
        if path.exists():
            rows.append(parse_audit(path, stage))
        else:
            rows.append(
                {
                    "audit_file": rel(path),
                    "stage": stage,
                    "verdict": "NOT FOUND",
                    "critical_count": "",
                    "high_count": "",
                    "medium_count": "",
                    "ready_for_report": False,
                    "test_leakage_detected": "",
                    "test_ranking_detected": "",
                    "metric_consistency_failures": "",
                    "main_blocker": "audit file not found",
                    "safe_to_use": False,
                    "notes": "not found",
                }
            )
    return pd.DataFrame(rows)


def add_result(rows: list[dict[str, Any]], **kwargs: Any) -> None:
    base = {
        "dataset": "",
        "stage": "",
        "model_or_policy": "",
        "policy_type": "",
        "audit_status": "",
        "safe_category": "",
        "precision": "",
        "recall": "",
        "specificity": "",
        "f1": "",
        "pr_auc": "",
        "roc_auc": "",
        "brier": "",
        "ece": "",
        "fp": "",
        "fn": "",
        "tp": "",
        "tn": "",
        "cost": "",
        "manual_review_rate": "",
        "capture_at_20": "",
        "lift_at_20": "",
        "main_strength": "",
        "main_weakness": "",
        "use_in_report_as": "",
        "notes": "",
    }
    base.update(kwargs)
    rows.append(base)


def key_results_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    v2_tw = first_row(read_csv(ROOT / "outputs/decision_revision_v2/locked_test_evaluation_taiwan.csv"))
    v2_he = first_row(read_csv(ROOT / "outputs/decision_revision_v2/locked_test_evaluation_heloc.csv"))
    fa_tw_df = read_csv(ROOT / "outputs/final_attempt/locked_test/final_locked_test_evaluation_taiwan.csv")
    fa_he_df = read_csv(ROOT / "outputs/final_attempt/locked_test/final_locked_test_evaluation_heloc.csv")
    scre_tw = best_by_dataset(read_csv(ROOT / "outputs/tables/scre_optimized_results_taiwan.csv"), "taiwan", split_col="split", split_val="test")
    scre_he = best_by_dataset(read_csv(ROOT / "outputs/tables/scre_optimized_results_heloc.csv"), "heloc", split_col="split", split_val="test")
    val_reg = read_csv(ROOT / "outputs/decision_revision_v2/validation_candidate_registry_all.csv")
    lit = read_csv(ROOT / "outputs/final_attempt/literature_reproduction/literature_reproduction_best_configs.csv")
    deep = read_csv(ROOT / "outputs/final_attempt/deep_tabular/deep_tabular_best_configs.csv")
    cap = read_csv(ROOT / "outputs/final_attempt/capacity_review/capacity_model_comparison.csv")

    def v2_row(dataset: str, r: dict[str, Any], model: str) -> None:
        add_result(
            rows,
            dataset=dataset,
            stage="V2",
            model_or_policy=model,
            policy_type=r.get("selected_policy_type", "manual_review_band"),
            audit_status="READY",
            safe_category="FINAL MAIN EVIDENCE",
            precision=fmt(r.get("test_precision")),
            recall=fmt(r.get("test_recall")),
            specificity=fmt(r.get("test_specificity")),
            f1=fmt(r.get("test_f1")),
            pr_auc=fmt(r.get("test_pr_auc")),
            roc_auc=fmt(r.get("test_roc_auc")),
            brier=fmt(r.get("test_brier")),
            ece=fmt(r.get("test_ece")),
            fp=as_int(r.get("test_fp")),
            fn=as_int(r.get("test_fn")),
            tp=as_int(r.get("test_tp")),
            tn=as_int(r.get("test_tn")),
            cost=fmt(r.get("test_review_adjusted_cost")),
            manual_review_rate=fmt(r.get("test_manual_review_rate")),
            main_strength="Audit-clean locked manual-review evidence; validation-only selection.",
            main_weakness="Not automatic rejection; manual-review workload exists.",
            use_in_report_as="Main final operational evidence",
            notes="Cost is review-adjusted cost with review_cost=0.5.",
        )

    v2_row("taiwan", v2_tw, "V2 CatBoost manual-review")
    v2_row("heloc", v2_he, "V2 Scorecard manual-review")

    for dataset, scre in [("taiwan", scre_tw), ("heloc", scre_he)]:
        add_result(
            rows,
            dataset=dataset,
            stage="Original SCRE",
            model_or_policy="SCRE-Optimized / research framework",
            policy_type="weighted ensemble threshold policy",
            audit_status="supporting",
            safe_category="SUPPORTING / APPENDIX EVIDENCE",
            precision=fmt(scre.get("precision")),
            recall=fmt(scre.get("recall")),
            specificity=fmt(scre.get("specificity")),
            f1=fmt(scre.get("f1")),
            pr_auc=fmt(scre.get("pr_auc")),
            roc_auc=fmt(scre.get("roc_auc")),
            brier=fmt(scre.get("brier_score")),
            ece=fmt(scre.get("ece")),
            fp=as_int(scre.get("fp")),
            fn=as_int(scre.get("fn")),
            tp=as_int(scre.get("tp")),
            tn=as_int(scre.get("tn")),
            cost=fmt(scre.get("expected_cost")),
            main_strength="Reliability-aware integration framework.",
            main_weakness="Does not dominate best single operational policies.",
            use_in_report_as="Framework/supporting evidence",
            notes="Do not claim SCRE beats all models.",
        )

    # V2 scorecard/interpretable benchmark for Taiwan from validation registry if present.
    score_tw = best_by_dataset(val_reg, "taiwan", sort_col="validation_expected_cost_binary", model_contains="Scorecard")
    if score_tw:
        add_result(
            rows,
            dataset="taiwan",
            stage="V2",
            model_or_policy="V2 Scorecard / interpretable benchmark",
            policy_type=score_tw.get("policy_type", ""),
            audit_status="READY selection artifact",
            safe_category="SUPPORTING / APPENDIX EVIDENCE",
            precision=fmt(score_tw.get("validation_precision")),
            recall=fmt(score_tw.get("validation_recall")),
            specificity=fmt(score_tw.get("validation_specificity")),
            f1=fmt(score_tw.get("validation_f1")),
            pr_auc=fmt(score_tw.get("validation_pr_auc")),
            roc_auc=fmt(score_tw.get("validation_roc_auc")),
            fp=as_int(score_tw.get("validation_fp")),
            fn=as_int(score_tw.get("validation_fn")),
            tp=as_int(score_tw.get("validation_tp")),
            tn=as_int(score_tw.get("validation_tn")),
            cost=fmt(score_tw.get("validation_expected_cost_binary")),
            main_strength="Interpretable benchmark.",
            main_weakness="Not final Taiwan operational winner.",
            use_in_report_as="Interpretable benchmark",
            notes="Validation-only registry evidence.",
        )

    fa_tw = first_row(fa_tw_df, "operational", "system_name")
    fa_mono = first_row(fa_tw_df, "Monotonic", "system_name")
    fa_he = first_row(fa_he_df, "operational", "system_name")
    for dataset, r, name, cat in [
        ("taiwan", fa_tw, "Final Attempt XGBoost manual-review", "DO NOT USE"),
        ("taiwan", fa_mono, "Final Attempt Monotonic LightGBM", "DO NOT USE"),
        ("heloc", fa_he, "Final Attempt Scorecard manual-review", "DIAGNOSTIC ONLY"),
    ]:
        if r:
            add_result(
                rows,
                dataset=dataset,
                stage="Final Attempt",
                model_or_policy=name,
                policy_type=r.get("policy_type", ""),
                audit_status="BLOCKED" if dataset == "taiwan" else "BLOCKED package / diagnostic",
                safe_category=cat,
                precision=fmt(r.get("test_precision")),
                recall=fmt(r.get("test_recall")),
                specificity=fmt(r.get("test_specificity")),
                f1=fmt(r.get("test_f1")),
                pr_auc=fmt(r.get("test_pr_auc")),
                roc_auc=fmt(r.get("test_roc_auc")),
                brier=fmt(r.get("test_brier")),
                ece=fmt(r.get("test_ece")),
                fp=as_int(r.get("test_fp")),
                fn=as_int(r.get("test_fn")),
                tp=as_int(r.get("test_tp")),
                tn=as_int(r.get("test_tn")),
                cost=fmt(r.get("test_review_adjusted_cost")),
                manual_review_rate=fmt(r.get("test_manual_review_rate")),
                main_strength="Stress-test evidence only.",
                main_weakness="Final Attempt audit is BLOCKED or has review-cost auditability warnings.",
                use_in_report_as="Do not use as final; appendix/diagnostic only if clearly labeled.",
                notes="See outputs/final_attempt/audit/final_attempt_audit.md.",
            )

    for dataset in ["taiwan", "heloc"]:
        lit_best = best_by_dataset(lit, dataset, rank_col="rank_by_validation_pr_auc")
        deep_best = best_by_dataset(deep, dataset, rank_col="rank_validation_deep")
        cap_scre = best_by_dataset(cap, dataset, sort_col="top_20_capture", ascending=False, split_col="split", split_val="validation", model_contains="SCRE")
        if lit_best:
            add_result(
                rows,
                dataset=dataset,
                stage="Final Attempt",
                model_or_policy=f"Literature reproduction best: {lit_best.get('config_display_name', '')}",
                policy_type="validation reproduction config",
                audit_status="supporting",
                safe_category="SUPPORTING / APPENDIX EVIDENCE",
                precision=fmt(lit_best.get("validation_precision_cost_policy")),
                recall=fmt(lit_best.get("validation_recall_cost_policy")),
                specificity=fmt(lit_best.get("validation_best_specificity_at_cost_policy")),
                pr_auc=fmt(lit_best.get("validation_pr_auc")),
                roc_auc=fmt(lit_best.get("validation_roc_auc")),
                brier=fmt(lit_best.get("validation_brier")),
                ece=fmt(lit_best.get("validation_ece")),
                cost=fmt(lit_best.get("validation_expected_cost_cost_policy")),
                main_strength="Leakage-free negative/robustness finding.",
                main_weakness="Not final selected policy.",
                use_in_report_as="Appendix robustness / literature reproduction",
                notes="Shows high SMOTE/DNN-style claims did not cleanly replace V2.",
            )
        if deep_best:
            add_result(
                rows,
                dataset=dataset,
                stage="Final Attempt",
                model_or_policy=f"DNN/BP NN: {deep_best.get('config_id', '')}",
                policy_type=deep_best.get("policy_type", ""),
                audit_status="supporting",
                safe_category="SUPPORTING / APPENDIX EVIDENCE",
                precision=fmt(deep_best.get("validation_precision")),
                recall=fmt(deep_best.get("validation_recall")),
                specificity=fmt(deep_best.get("validation_specificity")),
                f1=fmt(deep_best.get("validation_f1")),
                pr_auc=fmt(deep_best.get("validation_pr_auc")),
                roc_auc=fmt(deep_best.get("validation_roc_auc")),
                brier=fmt(deep_best.get("validation_brier")),
                ece=fmt(deep_best.get("validation_ece")),
                fp=as_int(deep_best.get("validation_fp")),
                fn=as_int(deep_best.get("validation_fn")),
                tp=as_int(deep_best.get("validation_tp")),
                tn=as_int(deep_best.get("validation_tn")),
                cost=fmt(deep_best.get("validation_expected_cost")),
                main_strength="Checks BP/DNN literature direction.",
                main_weakness="Does not become final operational policy.",
                use_in_report_as="Appendix negative/robustness result",
                notes="No new feature created; no test selection.",
            )
        if cap_scre:
            add_result(
                rows,
                dataset=dataset,
                stage="Final Attempt",
                model_or_policy="SCRE top-k review prioritization",
                policy_type="capacity-aware ranking",
                audit_status="supporting",
                safe_category="SUPPORTING / APPENDIX EVIDENCE",
                capture_at_20=fmt(cap_scre.get("top_20_capture")),
                lift_at_20=fmt(cap_scre.get("lift_at_20")),
                main_strength="Useful for screening/review prioritization framing.",
                main_weakness="Ranking support is not the same as automatic rejection.",
                use_in_report_as="Appendix capacity-aware review evidence",
                notes=f"Model: {cap_scre.get('model', '')}",
            )

    return pd.DataFrame(rows)


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    inv = file_inventory()
    inv.to_csv(OUT / "evidence_file_inventory.csv", index=False)

    audits = audit_summary()
    audits.to_csv(OUT / "audit_summary_table.csv", index=False)

    key = key_results_summary()
    key.to_csv(OUT / "all_key_results_summary.csv", index=False)

    v2_vs = pd.DataFrame(
        [
            ["audit readiness", "READY, 0 critical/high/medium", "BLOCKED, 9 critical and 4 high issues", "V2", "Final Attempt cannot be final while audit is BLOCKED."],
            ["test leakage status", "No test-selection issue in V2 audit", "No detected test-selection leakage", "Tie", "Both are acceptable on leakage, but Final Attempt fails metric consistency."],
            ["metric consistency", "Pass in final_audit_v2", "Fails manual-review metric identities", "V2", "Displayed Final Attempt metrics are not always recomputable from displayed confusion counts."],
            ["operational performance", "Taiwan CatBoost MR cost 2705.5; HELOC Scorecard MR cost 764.5", "Taiwan XGBoost MR cost 2775.5; HELOC similar to V2", "V2", "Final Attempt did not clearly improve the main operational result."],
            ["false-positive reduction", "Taiwan FP reduced 1982 -> 680 vs old aggressive CatBoost", "Taiwan Final Attempt FP 795", "V2", "V2 has fewer Taiwan false positives than Final Attempt locked operational row."],
            ["manual-review usability", "Clear locked policy and cost interpretation", "Useful but auditability warnings in locked rows", "V2", "Final Attempt needs manual-review denominator repair."],
            ["interpretability", "HELOC Scorecard main; Taiwan scorecard as benchmark", "Monotonic/EBM/Scorecard stress tests useful", "Hybrid", "V2 for main result, Final Attempt for appendix interpretability evidence."],
            ["external validation", "HELOC Scorecard manual-review READY", "HELOC Scorecard essentially preserves V2", "V2", "V2 already provides a clean external-validation story."],
            ["literature reproduction value", "Not primary purpose", "Strong appendix value", "Final Attempt", "Negative leakage-free reproduction is useful but not a final operational result."],
            ["report safety", "Safe as final main evidence", "Safe only as stress-test/appendix with BLOCKED caveat", "V2", "Main report should not depend on blocked rows."],
        ],
        columns=["criterion", "V2", "Final_Attempt", "winner", "reason"],
    )
    v2_vs.to_csv(OUT / "v2_vs_final_attempt_decision.csv", index=False)

    literature = pd.DataFrame(
        [
            ["Classical / boosting credit default models", "Boosting often performs strongly on Taiwan/HELOC style tabular credit data.", "CatBoost/XGBoost/LightGBM are competitive, but the safest final policy is V2 CatBoost MR for Taiwan and Scorecard MR for HELOC.", "No", "Raw accuracy is below some reported high-score papers.", "Protocol and threshold choices differ.", "Use as realistic, audit-clean credit-risk screening evidence."],
            ["Imbalanced learning / SMOTE / KMeansSMOTE", "SMOTE/KMeansSMOTE can produce large accuracy/AUC gains.", "Leakage-free reproduction did not clearly beat V2.", "No", "Did not reproduce the strongest high-score claims.", "Only directly comparable if train/test/resampling protocol matches.", "Negative but useful finding: resampling must be evaluated under natural test distribution."],
            ["Deep learning / BP neural network", "BP/DNN papers sometimes report high AUC or accuracy.", "DNN/BP tests did not become the operational winner.", "No", "No operational win over V2.", "Architectures/protocols may differ.", "Deep models are appendix robustness checks, not final main evidence."],
            ["SHAP stability / explanation reliability", "Stable explanations are important for credit model governance.", "SCRE and stability analyses support reliability framing but should not be overclaimed.", "Partially", "Not all final models have equally direct stability evidence.", "Model type and seed protocol differ across studies.", "Use explanation reliability as supporting evidence."],
            ["Cost-sensitive decisioning", "Thresholds should reflect FN/FP costs.", "Pure cost minimization was too aggressive; V2 revised this into manual-review decision support.", "Yes, operational framing is stronger.", "Not a pure cost-minimizer dominance result.", "Cost matrices are domain-assumption dependent.", "Main contribution: cost-aware but constrained manual-review policy."],
            ["Manual review / reject option / decision support", "Uncertain cases can be routed to human review.", "This is the strongest practical framing of the project.", "Yes, relative to plain binary thresholding.", "Review workload must be modeled carefully.", "Depends on review capacity/cost assumptions.", "Present as screening/manual-review support, not automatic rejection."],
            ["Decision curve / net benefit", "Net benefit checks whether model decisions improve over treat-all/treat-none.", "Decision curve supports model-based decision support in useful ranges.", "Partially", "Not the primary selection criterion.", "Clinical DCA analogy must be adapted to credit decisions.", "Use as appendix decision-support evidence."],
        ],
        columns=["literature_theme", "typical_claim", "how_our_project_compares", "stronger_than_literature", "weaker_than_literature", "not_directly_comparable", "safe_interpretation"],
    )
    literature.to_csv(OUT / "literature_positioning_table.csv", index=False)

    write_md(
        OUT / "audit_interpretation.md",
        """
# Audit Interpretation

1. **READY stage:** V2 is READY. `outputs/decision_revision_v2/final_audit_v2.md` reports no critical, high, or medium issues and resolves the previous test-ranking concern.
2. **BLOCKED stage:** Final Attempt is BLOCKED. `outputs/final_attempt/audit/final_attempt_audit.md` reports 9 critical metric-consistency failures and 4 high review-cost auditability warnings.
3. **Why BLOCKED:** The blocker is not test leakage. The blocker is that several final locked-test manual-review rows do not satisfy the requested metric identities from the displayed confusion-matrix columns.
4. **Safe final result:** V2 is the safe final operational evidence.
5. **Why Final Attempt cannot be main final:** Final Attempt is useful as stress-test/robustness evidence, but blocked locked-test rows cannot support the main final conclusion until repaired.
""",
    )

    write_md(
        OUT / "v2_vs_final_attempt_decision.md",
        """
# V2 vs Final Attempt Decision

**Final operational version should be: V2.**

Final Attempt does not operationally replace V2. It is useful because it stress-tested literature-style resampling, advanced imbalance handling, deep tabular models, interpretable middle models, capacity-aware review, and decision curves. However, its final locked-test manual-review rows are audit-blocked due to metric-consistency failures. Therefore, Final Attempt should be used only as supporting appendix/robustness evidence, not as the final model/policy.
""",
    )

    write_md(
        OUT / "scre_role_assessment.md",
        """
# SCRE-Credit Role Assessment

1. SCRE-Credit is **not** a dominant binary classifier across all metrics.
2. SCRE-Credit does **not** clearly beat CatBoost on Taiwan operational cost.
3. SCRE-Credit does **not** clearly beat Scorecard on HELOC operational cost.
4. SCRE-Optimized is useful in capacity-aware review prioritization, especially as a probability-ranking/review-support component.
5. The strongest SCRE contribution is methodological: it organizes performance, calibration, cost, stability, and faithfulness evidence into a reliability-aware framework.

Safe claim:

> SCRE-Credit should not be claimed as a universally superior classifier. Its strongest role is as a reliability-aware research framework and, where supported by capacity analysis, as a review-prioritization ranking tool.

Unsafe claims:

- SCRE-Credit outperforms all individual models.
- SCRE-Credit is the best automatic credit decision system.
- SCRE-Credit proves causal risk factors.
- SCRE-Credit should replace CatBoost/Scorecard in all settings.
""",
    )

    write_md(
        OUT / "literature_positioning_summary.md",
        """
# Literature Positioning Summary

The project should be positioned as a leakage-aware, decision-aware credit-risk modeling study rather than a high-score competition. Some SMOTE/KMeansSMOTE and BP/DNN papers report substantially higher metrics, but the Final Attempt reproduction did not recover those gains under natural test distribution and train-only resampling. This is a useful negative result.

The main contribution is operational: pure cost minimization produced too many false positives, so the project moved to validation-selected manual-review screening policies. V2 is the audit-clean final operational evidence. SCRE-Credit remains a reliability-aware framework and review-prioritization component, not a universally dominant classifier.
""",
    )

    write_md(
        OUT / "final_story_short.md",
        """
# Final Story - Very Short

The first cost-minimization model caught many defaults but produced too many false positives. V2 converted the project into a validation-selected manual-review screening system. V2 is audit-ready and should be the final operational result. Final Attempt tested more complex methods, but it did not cleanly beat V2 and is audit-blocked for final locked-test metric consistency. The final system should be presented as screening/manual-review support, not automatic credit rejection.
""",
    )

    write_md(
        OUT / "final_story_teacher_tr.md",
        """
# Hocaya Anlatım

Bu projede başlangıçta amaç yalnızca beklenen maliyeti düşürmekti. Fakat Taiwan tarafında saf cost-minimization modeli çok agresif davrandı: recall yükseldi ama precision ve specificity düştü, false positive sayısı çok arttı. Bu yüzden projeyi otomatik kredi reddi sistemi gibi değil, manuel inceleme destekli bir screening sistemi gibi yeniden konumlandırdım.

V2 aşamasında final seçim protokolünü düzelttim. Model ve policy seçimleri validation set üzerinden yapıldı; test set yalnızca kilitlenmiş policy için held-out evidence olarak kullanıldı. V2 audit sonucu READY olduğu için ana raporda en güvenli final sonuç V2 olmalı. Taiwan için CatBoost manual-review policy, HELOC için Scorecard manual-review policy ana sonuç olarak kullanılabilir.

Sonrasında Final Attempt ile daha güçlü yöntemleri de denedim: SMOTE/KMeansSMOTE literatür reprodüksiyonu, class-weight/imbalance boosting, BP/DNN, EBM/monotonic modeller, capacity-aware review ve decision curve. Bunlar faydalı ek analizler üretti, ama Final Attempt V2'yi temiz şekilde geçmedi. Ayrıca final locked-test manual-review tablolarında metric-consistency audit problemi çıktığı için Final Attempt ana final sonuç olarak kullanılmamalı.

SCRE-Credit'i de abartmadan sunmak gerekiyor. SCRE her modeli yenen bir classifier değil; daha doğru rolü reliability-aware framework ve bazı capacity-review analizlerinde review-prioritization aracı olmasıdır. Sonuç olarak proje otomatik karar sistemi değil, kredi riski için screening/manual-review support sistemi olarak anlatılmalı.
""",
    )

    write_md(
        OUT / "final_story_academic_en.md",
        """
# Final Story - Academic English

The project evolved from pure cost-sensitive classification into an audit-aware credit-risk decision-support framework. The initial cost-minimizing Taiwan policy achieved high default recall but created excessive false positives, making it unsuitable for automatic rejection. The V2 revision corrected the selection protocol by using validation-only policy selection and locked held-out test evaluation. Under this protocol, the Taiwan CatBoost manual-review policy and the HELOC Scorecard manual-review policy provide the cleanest final operational evidence.

The Final Attempt phase evaluated stronger and more complex alternatives, including leakage-free literature reproduction, advanced imbalance strategies, deep tabular models, interpretable middle models, capacity-aware review, and decision-curve analysis. These experiments are valuable robustness and appendix evidence, but they do not cleanly replace V2. The Final Attempt audit is blocked due to metric-consistency failures in final locked-test manual-review rows, even though no test-selection leakage was detected.

SCRE-Credit should therefore be positioned as a reliability-aware research framework and review-prioritization component, not as a universally superior classifier. The final claim should emphasize screening and manual-review support rather than automatic credit rejection.
""",
    )

    write_md(
        OUT / "report_usage_plan.md",
        """
# Report Usage Plan

## Report Main Results

- V2 final operational policies.
- V2 locked test results.
- V2 audit READY.
- Taiwan CatBoost manual-review policy.
- HELOC Scorecard manual-review policy.
- SCRE framework role.
- Scorecard/monotonic interpretability benchmark.
- Claim control.
- No detected test leakage in the final V2 protocol.

## Appendix / Robustness Results

- Final Attempt literature reproduction.
- KMeansSMOTE / DNN negative result.
- Advanced imbalance stress test.
- EBM / monotonic middle model.
- Capacity-aware review.
- Decision curve / net benefit.
- Final Attempt audit BLOCKED explanation.

## Do Not Report As Final

- Final Attempt locked-test manual-review rows with metric inconsistency.
- Test-ranked tables.
- Any automatic rejection claim.
- Any “SCRE beats all” claim.
- Any “deep learning reproduced literature high scores” claim unless audit-clean.
""",
    )

    write_md(
        OUT / "final_fix_or_freeze_decision.md",
        """
# Final Fix or Freeze Decision

Proceed to report: **YES**, if V2 is used as the final operational version.

Use V2 as final: **YES**.

Use Final Attempt as final: **NO**.

Use Final Attempt as appendix: **YES**.

Only remaining fix: if you want Final Attempt to replace V2, the final locked-test manual-review metric-consistency issue must be repaired and the final attempt audit must be rerun. If V2 remains final, no new experiment is required before report writing.
""",
    )

    # ChatGPT handoff summary.
    main_rows = key[key["safe_category"] == "FINAL MAIN EVIDENCE"]
    supporting = [
        ["Final Attempt literature reproduction", "Leakage-free SMOTE/KMeansSMOTE/high-score reproduction did not replace V2.", "Appendix robustness / literature positioning"],
        ["Advanced imbalance boosting", "Class-weight and calibration variants found candidates but did not safely replace V2.", "Appendix robustness"],
        ["DNN/BP NN", "Deep tabular reproduction did not become the operational winner.", "Appendix negative result"],
        ["Capacity-aware review", "SCRE/probability ranking can support review prioritization.", "Appendix / discussion"],
        ["Decision curve", "Model-based decision support is useful in threshold ranges.", "Appendix decision-support evidence"],
    ]
    do_not = [
        ["Final Attempt locked-test manual-review rows", "Metric-consistency failures in final_attempt_audit.md."],
        ["Test-ranked tables", "Invalid for model selection."],
        ["Automatic rejection claim", "Precision/FP behavior requires manual review."],
        ["SCRE beats all claim", "Not supported by Taiwan/HELOC operational winners."],
    ]
    audit_md_rows = []
    for _, r in audits.iterrows():
        audit_md_rows.append(f"| {r['stage']} | {r['verdict']} | {r['critical_count']} | {r['ready_for_report']} | {r['main_blocker'] or r['notes']} |")
    main_md_rows = []
    for _, r in main_rows.iterrows():
        main_md_rows.append(
            f"| {r['dataset']} | {r['model_or_policy']} | {r['precision']} | {r['recall']} | {r['specificity']} | {r['fp']} | {r['fn']} | {r['cost']} | {r['manual_review_rate']} | {r['safe_category']} |"
        )
    support_md = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in supporting)
    do_not_md = "\n".join(f"| {a} | {b} |" for a, b in do_not)
    v2_md = "\n".join(f"| {r.criterion} | {r.V2} | {r.Final_Attempt} | {r.winner} | {r.reason} |" for r in v2_vs.itertuples())
    write_md(
        OUT / "SEND_TO_CHATGPT_FINAL_INVENTORY.md",
        f"""
# FINAL PROJECT INVENTORY FOR CHATGPT

## 1. Final decision
- Proceed to report: YES, with V2 as final.
- Final operational version: V2.
- Use V2 as final: YES.
- Use Final Attempt as final: NO.
- Use Final Attempt as appendix: YES.

## 2. Audit summary
| Stage | Verdict | Critical Issues | Ready? | Main Note |
|---|---|---:|---|---|
{chr(10).join(audit_md_rows)}

## 3. Main final results
| Dataset | Final Policy | Precision | Recall | Specificity | FP | FN | Cost | MR Rate | Safe Category |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(main_md_rows)}

## 4. Supporting results
| Result | What it showed | Use in report |
|---|---|---|
{support_md}

## 5. Do not use as final
| Result | Why not |
|---|---|
{do_not_md}

## 6. V2 vs Final Attempt
| Criterion | V2 | Final Attempt | Winner | Reason |
|---|---|---|---|---|
{v2_md}

## 7. SCRE role

SCRE-Credit should not be claimed as a universally superior classifier. Its strongest role is as a reliability-aware research framework and, where supported by capacity analysis, as a review-prioritization ranking tool.

## 8. Literature positioning

The project is strongest as a leakage-aware and decision-aware credit-risk screening study. High SMOTE/DNN literature scores were not reproduced as operational improvements under the strict protocol; this is a useful negative finding rather than a failure.

## 9. Final story

The old cost-minimization policy caught many defaults but produced too many false positives. V2 converted the project into a validation-selected manual-review screening system. V2 is audit-ready and should remain the final operational result. Final Attempt tested stronger methods but did not cleanly beat V2 and is audit-blocked for metric consistency. The project should be presented as screening/manual-review support, not automatic rejection.

## 10. Remaining fix

Only required if Final Attempt is to replace V2: repair manual-review metric-consistency rows and rerun the audit. If V2 remains final, proceed to report.

## 11. Questions for ChatGPT
1. Bu envantere göre rapora geçmeli miyim?
2. V2 final olarak kalmalı mı?
3. Final Attempt appendix olarak nasıl kullanılmalı?
4. Ana sonuç cümlesi ne olmalı?
5. Hocaya bunu nasıl anlatmalıyım?
6. Rapor iskeletini nasıl kurmalıyım?
""",
    )

    print(f"Generated final inventory in {OUT}")


if __name__ == "__main__":
    main()
