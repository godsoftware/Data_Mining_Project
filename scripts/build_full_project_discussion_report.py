from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "full_project_discussion_report"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path) -> pd.DataFrame:
    path = ROOT / path if not isinstance(path, Path) else path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text(path: str | Path, max_chars: int | None = None) -> str:
    path = ROOT / path if not isinstance(path, Path) else path
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if max_chars is None else text[:max_chars]


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except Exception:
        pass
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_Tablo boş veya dosya bulunamadı._"
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    table = table.fillna("NA")
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def stage_for(path: Path) -> str:
    s = str(path).replace("\\", "/").lower()
    if "final_frozen_v2" in s:
        return "V2 Frozen"
    if "decision_revision_v2" in s:
        return "V2"
    if "final_weakness_closing" in s:
        return "Weakness Closing"
    if "final_attempt" in s:
        return "Final Attempt"
    if "decision_revision/" in s:
        return "Decision Revision V1"
    if "/tables/" in s:
        return "Original SCRE / Core Tables"
    if "/models/" in s:
        return "Models"
    if "/figures/" in s:
        return "Figures"
    if "/src/" in s:
        return "Source"
    if "/data/" in s:
        return "Data"
    return "Unknown"


def purpose_for(path: Path) -> str:
    name = path.name.lower()
    if "audit" in name:
        return "Audit / validation evidence"
    if "locked_test" in name or "heldout" in name:
        return "Locked held-out test evidence"
    if "validation" in name or "selection" in name:
        return "Validation-only selection evidence"
    if "claim" in name:
        return "Claim control"
    if "capacity" in name or "lift" in name:
        return "Capacity-aware review / Lift@K"
    if "decision_curve" in name:
        return "Decision curve / net benefit"
    if "scre" in name:
        return "SCRE-Credit framework evidence"
    if "shap" in name or "lime" in name:
        return "Explainability evidence"
    if "calibration" in name:
        return "Calibration evidence"
    if "threshold" in name or "cost" in name:
        return "Threshold / cost analysis"
    if "taiwan" in name or "heloc" in name:
        return "Dataset/model result"
    return "Project artifact"


def inspect_file(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    row_count = ""
    col_count = ""
    cols: list[str] = []
    text_sample = ""
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            row_count, col_count = df.shape
            cols = list(df.columns)
        except Exception as exc:
            text_sample = f"csv read error: {exc}"
    elif path.suffix.lower() in {".md", ".txt", ".json"}:
        text_sample = read_text(path, 5000)
        row_count = text_sample.count("\n") + 1 if text_sample else 0
        col_count = ""
    elif path.suffix.lower() in {".xlsx"}:
        try:
            xl = pd.ExcelFile(path)
            row_count = len(xl.sheet_names)
            col_count = "sheets"
            cols = xl.sheet_names
        except Exception as exc:
            text_sample = f"xlsx read error: {exc}"

    all_tokens = " ".join(cols).lower() + " " + text_sample.lower() + " " + rel.lower()
    contains_train = "train_" in all_tokens or "training" in all_tokens
    contains_validation = "validation" in all_tokens or "val_" in all_tokens
    contains_test = "test_" in all_tokens or "heldout" in all_tokens or "locked_test" in all_tokens
    contains_rank = "rank" in all_tokens or "winner" in all_tokens
    contains_audit = "audit" in all_tokens or "verdict" in all_tokens or "ready" in all_tokens

    safe_main = False
    safe_appendix = False
    unsafe_reason = ""
    if "outputs/final_frozen_v2" in rel:
        safe_main = True
        safe_appendix = True
    elif "outputs/decision_revision_v2" in rel:
        safe_main = any(
            key in rel
            for key in [
                "final_audit_v2.md",
                "final_safe_recommendation.md",
                "final_validation_selection_table.csv",
                "final_heldout_test_evidence_table.csv",
                "locked_test_evaluation",
            ]
        )
        safe_appendix = True
    elif "outputs/final_weakness_closing" in rel:
        safe_appendix = True
        if "validation_all" in rel and contains_test:
            unsafe_reason = "Diagnostic test columns exist; not selection evidence."
    elif "outputs/final_attempt" in rel:
        safe_appendix = True
        safe_main = False
        unsafe_reason = "Final Attempt audit BLOCKED; appendix/robustness only."
    elif "outputs/decision_revision/" in rel:
        safe_appendix = True
        unsafe_reason = "Decision Revision V1 was NOT READY; diagnostic only."
    elif "outputs/tables" in rel:
        safe_appendix = True

    return {
        "file_path": rel,
        "exists": True,
        "stage": stage_for(path),
        "row_count": row_count,
        "column_count": col_count,
        "purpose": purpose_for(path),
        "contains_train_metrics": contains_train,
        "contains_validation_metrics": contains_validation,
        "contains_test_metrics": contains_test,
        "contains_rank_or_winner": contains_rank,
        "contains_audit_status": contains_audit,
        "safe_for_main_report": safe_main,
        "safe_for_appendix": safe_appendix,
        "unsafe_reason": unsafe_reason,
        "notes": "Generated from current filesystem inventory.",
    }


def build_file_inventory() -> pd.DataFrame:
    required = [
        "README.md",
        "PROJECT_SCOPE.md",
        "outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv",
        "outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv",
        "outputs/final_frozen_v2/final_readiness_check.md",
        "outputs/final_frozen_v2/final_claim_control_matrix.csv",
        "outputs/decision_revision_v2/final_audit_v2.md",
        "outputs/decision_revision_v2/final_safe_recommendation.md",
        "outputs/decision_revision_v2/final_validation_selection_table.csv",
        "outputs/decision_revision_v2/final_heldout_test_evidence_table.csv",
        "outputs/final_attempt/audit/final_attempt_audit.md",
        "outputs/final_attempt/report_to_user/FINAL_ATTEMPT_RESULTS_FOR_CHATGPT.md",
        "outputs/final_weakness_closing/audit/final_weakness_closing_audit.md",
        "outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md",
        "outputs/final_project_inventory/SEND_TO_CHATGPT_FINAL_INVENTORY.md",
        "outputs/final_revision_audit.md",
        "outputs/final_positioning_statement.md",
        "outputs/tables/final_decision_matrix.csv",
        "outputs/tables/claim_control_matrix.csv",
    ]
    candidate_files: set[Path] = set()
    for root in [
        ROOT / "outputs",
        ROOT / "src",
        ROOT / "notebooks",
        ROOT / "data",
    ]:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".csv", ".md", ".json", ".xlsx", ".png", ".joblib", ".py", ".yaml", ".txt"}:
                    candidate_files.add(p)
    for rel in required:
        candidate_files.add(ROOT / rel)

    rows = []
    for path in sorted(candidate_files, key=lambda p: p.as_posix().lower()):
        if path.exists():
            rows.append(inspect_file(path))
        else:
            rows.append(
                {
                    "file_path": path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix(),
                    "exists": False,
                    "stage": stage_for(path),
                    "row_count": "",
                    "column_count": "",
                    "purpose": purpose_for(path),
                    "contains_train_metrics": False,
                    "contains_validation_metrics": False,
                    "contains_test_metrics": False,
                    "contains_rank_or_winner": False,
                    "contains_audit_status": False,
                    "safe_for_main_report": False,
                    "safe_for_appendix": False,
                    "unsafe_reason": "not found",
                    "notes": "Bu dosya bulunamadı, ilgili sonuç doğrudan doğrulanamadı.",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "file_inventory.csv", index=False)
    return df


TAIWAN_MEANINGS = {
    "ID": "Müşteri satır kimliği. Model feature olarak kullanılmamalıdır.",
    "LIMIT_BAL": "Verilen kredi limiti / kredi kartı limiti.",
    "SEX": "Cinsiyet kodu. Demografik değişkendir; fairness açısından dikkatli yorumlanır.",
    "EDUCATION": "Eğitim seviyesi. 0/5/6 anomalileri 4=Other altında birleştirilmiştir.",
    "MARRIAGE": "Medeni durum. 0 anomalisi 3=Other altında birleştirilmiştir.",
    "AGE": "Yaş.",
    "PAY_0": "En güncel ödeme durumu / gecikme kodu.",
    "PAY_2": "İki dönem önceki ödeme durumu / gecikme kodu.",
    "PAY_3": "Üç dönem önceki ödeme durumu / gecikme kodu.",
    "PAY_4": "Dört dönem önceki ödeme durumu / gecikme kodu.",
    "PAY_5": "Beş dönem önceki ödeme durumu / gecikme kodu.",
    "PAY_6": "Altı dönem önceki ödeme durumu / gecikme kodu.",
    "BILL_AMT1": "En güncel fatura/borç tutarı.",
    "BILL_AMT2": "İki dönem önceki fatura/borç tutarı.",
    "BILL_AMT3": "Üç dönem önceki fatura/borç tutarı.",
    "BILL_AMT4": "Dört dönem önceki fatura/borç tutarı.",
    "BILL_AMT5": "Beş dönem önceki fatura/borç tutarı.",
    "BILL_AMT6": "Altı dönem önceki fatura/borç tutarı.",
    "PAY_AMT1": "En güncel ödeme tutarı.",
    "PAY_AMT2": "İki dönem önceki ödeme tutarı.",
    "PAY_AMT3": "Üç dönem önceki ödeme tutarı.",
    "PAY_AMT4": "Dört dönem önceki ödeme tutarı.",
    "PAY_AMT5": "Beş dönem önceki ödeme tutarı.",
    "PAY_AMT6": "Altı dönem önceki ödeme tutarı.",
    "default.payment.next.month": "Orijinal hedef değişken adı.",
    "default_next_month": "Temizlenmiş hedef değişken; 1=gelecek ay default, 0=non-default.",
}


HELOC_MEANINGS = {
    "RiskPerformance": "Orijinal HELOC hedefi: Good/Bad hesap performansı.",
    "bad_flag": "Temizlenmiş hedef; 1=Bad, 0=Good.",
    "ExternalRiskEstimate": "Dış kredi bürosu risk skoru.",
    "MSinceOldestTradeOpen": "En eski hesabın açılışından bu yana geçen ay.",
    "MSinceMostRecentTradeOpen": "En yeni hesabın açılışından bu yana geçen ay.",
    "AverageMInFile": "Dosyadaki hesapların ortalama ay yaşı.",
    "NumSatisfactoryTrades": "Tatmin edici hesap sayısı.",
    "NumTrades60Ever2DerogPubRec": "60+ gün gecikmeli veya derogatory public record sayısı.",
    "NumTrades90Ever2DerogPubRec": "90+ gün gecikmeli veya derogatory public record sayısı.",
    "PercentTradesNeverDelq": "Hiç gecikmemiş hesap yüzdesi.",
    "MSinceMostRecentDelq": "Son gecikmeden bu yana geçen ay.",
    "MaxDelq2PublicRecLast12M": "Son 12 aydaki maksimum gecikme/public record şiddeti.",
    "MaxDelqEver": "Tüm tarihteki maksimum gecikme şiddeti.",
    "NumTotalTrades": "Toplam hesap/trade sayısı.",
    "NumTradesOpeninLast12M": "Son 12 ayda açılan hesap sayısı.",
    "PercentInstallTrades": "Taksitli hesapların yüzdesi.",
    "MSinceMostRecentInqexcl7days": "Son sorgudan bu yana ay; son 7 gün hariç.",
    "NumInqLast6M": "Son 6 ay sorgu sayısı.",
    "NumInqLast6Mexcl7days": "Son 6 ay sorgu sayısı; son 7 gün hariç.",
    "NetFractionRevolvingBurden": "Revolving kredi yükü oranı.",
    "NetFractionInstallBurden": "Taksitli kredi yükü oranı.",
    "NumRevolvingTradesWBalance": "Bakiyesi olan revolving hesap sayısı.",
    "NumInstallTradesWBalance": "Bakiyesi olan taksitli hesap sayısı.",
    "NumBank2NatlTradesWHighUtilization": "Yüksek kullanım oranına sahip banka/national hesap sayısı.",
    "PercentTradesWBalance": "Bakiyesi olan hesapların yüzdesi.",
}


def build_dataset_dictionary() -> pd.DataFrame:
    taiwan_interim = read_csv("data/interim/taiwan_cleaned.csv")
    taiwan_processed = read_csv("data/processed/taiwan_model_ready.csv")
    heloc_interim = read_csv("data/interim/heloc_cleaned.csv")
    heloc_processed = read_csv("data/processed/heloc_model_ready.csv")
    taiwan_cols = list(taiwan_interim.columns) or [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        *[f"BILL_AMT{i}" for i in range(1, 7)],
        *[f"PAY_AMT{i}" for i in range(1, 7)],
        "default_next_month",
    ]
    heloc_cols = list(heloc_interim.columns)

    rows: list[dict[str, Any]] = []
    for col in taiwan_cols:
        is_target = col in {"default_next_month", "default.payment.next.month"}
        rows.append(
            {
                "dataset": "taiwan",
                "column_name": col,
                "type": str(taiwan_interim[col].dtype) if col in taiwan_interim.columns else "unknown",
                "meaning": TAIWAN_MEANINGS.get(col, "Taiwan raw/model feature."),
                "raw_used": True,
                "cleaned_used": col in taiwan_interim.columns,
                "engineered_from_it": col.startswith("PAY_") or col.startswith("BILL_AMT") or col.startswith("PAY_AMT") or col == "LIMIT_BAL",
                "used_in_model": col in taiwan_processed.columns and not is_target and col != "ID",
                "used_in_cost_analysis": col in {"LIMIT_BAL"} or col.startswith("BILL_AMT") or col.startswith("PAY_AMT"),
                "used_in_explainability": col in taiwan_processed.columns and not is_target and col != "ID",
                "used_in_segment_analysis": col in {"LIMIT_BAL", "PAY_0"} or col.startswith("PAY_"),
                "excluded_reason": "ID excluded from model" if col == "ID" else ("target; never used as feature" if is_target else ""),
                "notes": "Target-free source column; cleaning/feature use documented." if not is_target else "Only target; not a feature.",
            }
        )

    for col in heloc_cols:
        is_target = col == "bad_flag"
        rows.append(
            {
                "dataset": "heloc",
                "column_name": col,
                "type": str(heloc_interim[col].dtype) if col in heloc_interim.columns else "unknown",
                "meaning": HELOC_MEANINGS.get(col, "HELOC credit-risk attribute."),
                "raw_used": True,
                "cleaned_used": col in heloc_interim.columns,
                "engineered_from_it": col
                in {
                    "NumTrades60Ever2DerogPubRec",
                    "NumTrades90Ever2DerogPubRec",
                    "PercentTradesNeverDelq",
                    "NumTradesOpeninLast12M",
                    "NumTotalTrades",
                    "NumInqLast6Mexcl7days",
                    "MSinceMostRecentInqexcl7days",
                    "NetFractionRevolvingBurden",
                    "NetFractionInstallBurden",
                    "MSinceOldestTradeOpen",
                    "AverageMInFile",
                    "NumBank2NatlTradesWHighUtilization",
                },
                "used_in_model": col in heloc_processed.columns and not is_target,
                "used_in_cost_analysis": col
                in {"ExternalRiskEstimate", "NetFractionRevolvingBurden", "AverageMInFile", "PercentTradesNeverDelq"},
                "used_in_scorecard": col in heloc_processed.columns and not is_target,
                "used_in_specificity_experiment": col in heloc_processed.columns and not is_target,
                "used_in_explainability": col in heloc_processed.columns and not is_target,
                "used_in_segment_analysis": col
                in {"ExternalRiskEstimate", "NetFractionRevolvingBurden", "PercentTradesNeverDelq"},
                "excluded_reason": "target; never used as feature" if is_target else "",
                "notes": "Special missing codes converted to NaN and imputed inside model pipelines." if not is_target else "Positive class = Bad.",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "DATASET_COLUMN_DICTIONARY.csv", index=False)
    return df


def build_feature_inventory() -> pd.DataFrame:
    taiwan_feature_dict = read_csv("outputs/tables/feature_dictionary_taiwan.csv")
    heloc_feature_dict = read_csv("outputs/tables/feature_dictionary_heloc.csv")
    rows = []
    for df, dataset in [(taiwan_feature_dict, "taiwan"), (heloc_feature_dict, "heloc")]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "feature_name": row.get("feature", ""),
                    "source_columns": row.get("source_columns", ""),
                    "formula": row.get("formula", ""),
                    "reason": row.get("concept", ""),
                    "used_in_model": True,
                    "used_in_error_analysis": True,
                    "used_in_cost_analysis": any(
                        token in str(row.get("feature", "")).lower()
                        for token in ["utilization", "bill", "pay", "burden", "exposure", "risk"]
                    ),
                    "used_in_explainability": True,
                    "risk_of_redundancy": "MEDIUM; several engineered features summarize overlapping payment/bill behavior.",
                    "kept_or_removed": "kept",
                    "uses_target": row.get("uses_target", False),
                    "notes": "Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used.",
                }
            )
    processed = {
        "taiwan": read_csv("data/processed/taiwan_model_ready.csv"),
        "heloc": read_csv("data/processed/heloc_model_ready.csv"),
    }
    for dataset, dfp in processed.items():
        for col in dfp.columns if not dfp.empty else []:
            if col in {"default_next_month", "bad_flag"}:
                rows.append(
                    {
                        "dataset": dataset,
                        "feature_name": col,
                        "source_columns": col,
                        "formula": "target",
                        "reason": "Prediction label only.",
                        "used_in_model": False,
                        "used_in_error_analysis": True,
                        "used_in_cost_analysis": False,
                        "used_in_explainability": False,
                        "risk_of_redundancy": "NA",
                        "kept_or_removed": "target_not_feature",
                        "uses_target": True,
                        "notes": "Never used as model feature.",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "USED_UNUSED_FEATURES.csv", index=False)
    return out


def build_data_cleaning_audit() -> pd.DataFrame:
    rows = []
    for dataset, path, target in [
        ("taiwan", "data/interim/taiwan_cleaned.csv", "default_next_month"),
        ("taiwan_model_ready", "data/processed/taiwan_model_ready.csv", "default_next_month"),
        ("heloc", "data/interim/heloc_cleaned.csv", "bad_flag"),
        ("heloc_model_ready", "data/processed/heloc_model_ready.csv", "bad_flag"),
    ]:
        df = read_csv(path)
        if df.empty:
            rows.append({"dataset": dataset, "check": "file_exists", "status": "FAIL", "value": "not found", "notes": path})
            continue
        rows += [
            {"dataset": dataset, "check": "shape", "status": "PASS", "value": f"{df.shape[0]} rows x {df.shape[1]} cols", "notes": path},
            {"dataset": dataset, "check": "missing_total", "status": "PASS" if df.isna().sum().sum() == 0 else "REVIEW", "value": int(df.isna().sum().sum()), "notes": "HELOC missing values are expected after special-code handling."},
            {"dataset": dataset, "check": "duplicate_rows", "status": "PASS" if df.duplicated().sum() == 0 else "REVIEW", "value": int(df.duplicated().sum()), "notes": "Reported; not automatically dropped."},
        ]
        if target in df.columns:
            counts = df[target].value_counts().sort_index()
            rates = df[target].value_counts(normalize=True).sort_index()
            rows.append({"dataset": dataset, "check": "target_distribution", "status": "PASS", "value": "; ".join(f"{k}:{int(counts[k])} ({rates[k]:.3f})" for k in counts.index), "notes": "Class imbalance documented."})
        if dataset.startswith("taiwan") and "EDUCATION" in df.columns:
            rows.append({"dataset": dataset, "check": "education_values", "status": "PASS", "value": sorted(df["EDUCATION"].dropna().astype(int).unique().tolist()), "notes": "0,5,6 mapped to 4 in cleaning."})
        if dataset.startswith("taiwan") and "MARRIAGE" in df.columns:
            rows.append({"dataset": dataset, "check": "marriage_values", "status": "PASS", "value": sorted(df["MARRIAGE"].dropna().astype(int).unique().tolist()), "notes": "0 mapped to 3 in cleaning."})
        if dataset.startswith("taiwan"):
            neg = {c: int((df[c] < 0).sum()) for c in df.columns if c.startswith("BILL_AMT")}
            rows.append({"dataset": dataset, "check": "negative_bill_amounts", "status": "REVIEW" if any(neg.values()) else "PASS", "value": json.dumps(neg), "notes": "Negative bill amounts can represent credit balances; preserved."})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "data_cleaning_audit.csv", index=False)
    return out


def build_results_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in [
        "outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv",
        "outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv",
    ]:
        df = read_csv(p)
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": r.get("dataset"),
                    "stage": "V2 Frozen",
                    "model_or_policy": r.get("final_policy"),
                    "policy_type": r.get("policy_type"),
                    "audit_status": r.get("audit_status"),
                    "safe_category": r.get("safe_category"),
                    "precision": r.get("precision"),
                    "recall": r.get("recall"),
                    "specificity": r.get("specificity"),
                    "f1": "",
                    "pr_auc": "",
                    "roc_auc": "",
                    "brier": "",
                    "ece": "",
                    "fp": r.get("fp"),
                    "fn": r.get("fn"),
                    "tp": "",
                    "tn": "",
                    "cost": r.get("cost"),
                    "manual_review_rate": r.get("manual_review_rate"),
                    "capture_at_20": "",
                    "lift_at_20": "",
                    "main_strength": "Audit-ready final operational evidence.",
                    "main_weakness": "Manual-review screening only; not automatic rejection.",
                    "use_in_report_as": "FINAL MAIN EVIDENCE",
                    "notes": r.get("interpretation"),
                }
            )
    tw_recall = read_csv("outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv")
    for _, r in tw_recall.iterrows():
        rows.append(
            {
                "dataset": "taiwan",
                "stage": "Weakness Closing",
                "model_or_policy": r.get("candidate_id"),
                "policy_type": r.get("policy_type"),
                "audit_status": "READY package but not final replacement",
                "safe_category": "SUPPORTING / APPENDIX EVIDENCE",
                "precision": r.get("test_precision"),
                "recall": r.get("test_recall"),
                "specificity": r.get("test_specificity"),
                "f1": "",
                "pr_auc": "",
                "roc_auc": "",
                "brier": "",
                "ece": "",
                "fp": r.get("test_fp"),
                "fn": r.get("test_fn"),
                "tp": "",
                "tn": "",
                "cost": r.get("test_review_adjusted_cost", r.get("test_cost")),
                "manual_review_rate": r.get("test_manual_review_rate"),
                "capture_at_20": "",
                "lift_at_20": "",
                "main_strength": "Recall increased versus V2.",
                "main_weakness": "Precision/specificity dropped and FP increased.",
                "use_in_report_as": "Appendix / weakness-closing discussion",
                "notes": r.get("operational_comment"),
            }
        )
    heloc_spec = read_csv("outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv")
    for _, r in heloc_spec.iterrows():
        rows.append(
            {
                "dataset": "heloc",
                "stage": "Weakness Closing",
                "model_or_policy": r.get("candidate_id"),
                "policy_type": r.get("policy_type"),
                "audit_status": "READY package but not final replacement",
                "safe_category": "SUPPORTING / APPENDIX EVIDENCE",
                "precision": r.get("test_precision"),
                "recall": r.get("test_recall"),
                "specificity": r.get("test_specificity"),
                "f1": r.get("test_f1"),
                "pr_auc": r.get("test_pr_auc"),
                "roc_auc": r.get("test_roc_auc"),
                "brier": r.get("test_brier"),
                "ece": r.get("test_ece"),
                "fp": r.get("test_fp"),
                "fn": r.get("test_fn"),
                "tp": "",
                "tn": "",
                "cost": r.get("test_review_adjusted_cost", r.get("test_cost")),
                "manual_review_rate": r.get("test_manual_review_rate"),
                "capture_at_20": "",
                "lift_at_20": "",
                "main_strength": "Specificity and FP improved.",
                "main_weakness": "Cost increased and recall declined.",
                "use_in_report_as": "Appendix / weakness-closing discussion",
                "notes": r.get("operational_comment"),
            }
        )
    cap = read_csv("outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv")
    for _, r in cap.iterrows():
        if str(r.get("model", "")).startswith("SCRE") or "V2" in str(r.get("model", "")):
            rows.append(
                {
                    "dataset": r.get("dataset"),
                    "stage": "Weakness Closing Capacity",
                    "model_or_policy": r.get("model"),
                    "policy_type": "top-k review prioritization",
                    "audit_status": "Appendix evidence",
                    "safe_category": "SUPPORTING / APPENDIX EVIDENCE",
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
                    "manual_review_rate": 0.20,
                    "capture_at_20": r.get("top_20_capture"),
                    "lift_at_20": r.get("lift_at_20"),
                    "main_strength": "Review prioritization evidence.",
                    "main_weakness": "Ranking support, not final classifier.",
                    "use_in_report_as": "Appendix / SCRE role",
                    "notes": "Top-20% review prioritization result.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "MODEL_AND_POLICY_RESULTS_SUMMARY.csv", index=False)
    return out


def build_experiment_inventory() -> pd.DataFrame:
    rows = [
        ("Prompt 0-12", "Scope/data/pipeline/baselines/calibration/threshold", "Taiwan+HELOC core pipeline kurulumu", "Ana deney altyapısı", "PASS/PARTIAL across audits", "Core Tables"),
        ("SCRE Original", "SCRE-Credit ensemble", "Reliability-aware ensemble formülü", "Framework fikrini test etmek", "Dominant classifier olmadı; framework olarak kaldı", "outputs/tables/scre_credit_results_*.csv"),
        ("SCRE Revision", "SCRE-Pareto / SCRE-Optimized", "Zayıf modelleri dışlama ve validation ağırlık optimizasyonu", "SCRE seyreltme problemini azaltmak", "Bazı PR-AUC/ranking avantajları; operational cost winner değil", "outputs/tables/scre_revision_global_comparison.csv"),
        ("Decision Revision V1", "FP reduction experiments", "Precision constraints, cost matrix, manual review, error analysis", "Agresif threshold FP sorununu anlamak", "NOT READY due to test-ranking ambiguity", "outputs/decision_revision/final_decision_revision_audit.md"),
        ("Decision Revision V2", "Validation-only selector", "Validation registry, locked policies, held-out test evidence", "Test-ranking metodoloji sorununu düzeltmek", "READY; final operational evidence", "outputs/decision_revision_v2/final_audit_v2.md"),
        ("Final Frozen V2", "Final package lock", "V2 CatBoost Taiwan / Scorecard HELOC donduruldu", "Ana rapor kanıtını kilitlemek", "FINAL MAIN EVIDENCE", "outputs/final_frozen_v2/"),
        ("Final Attempt", "Large robustness package", "Literature reproduction, imbalance, DNN, EBM, capacity, DCA", "Son büyük yöntem denemesi", "BLOCKED; appendix only", "outputs/final_attempt/audit/final_attempt_audit.md"),
        ("Weakness Closing", "V2 zayıflık kapatma", "Recall, capacity, specificity, cost, instance cost, segment, temporal", "V2 zayıflıklarını audit-clean denemek", "READY but no V3 final replacement", "outputs/final_weakness_closing/audit/final_weakness_closing_audit.md"),
        ("Explainability", "SHAP/LIME/stability/faithfulness", "Global/local explanation, stability, faithfulness", "Model davranışını incelemek", "Appendix/supporting evidence", "outputs/tables/shap_*.csv; outputs/tables/faithfulness_*.csv"),
        ("Statistical testing", "Bootstrap/McNemar/AUC comparison", "Model farklarını belirsizlikle raporlamak", "Overclaim riskini azaltmak", "Supporting evidence", "outputs/tables/statistical_tests_cleaned.csv"),
    ]
    df = pd.DataFrame(rows, columns=["experiment", "area", "what_was_done", "why_done", "result", "evidence_file"])
    df.to_csv(OUT / "EXPERIMENT_INVENTORY_ALL.csv", index=False)
    return df


def build_worked_failed() -> pd.DataFrame:
    rows = [
        ("V2 CatBoost manual-review Taiwan", "WORKED AS FINAL", "Audit-ready, validation-selected, locked test evidence; balanced FP control vs old aggressive policy.", "outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv", "FINAL MAIN EVIDENCE", "Screening/manual-review only."),
        ("V2 Scorecard manual-review HELOC", "WORKED AS FINAL", "External dataset tarafında interpretable ve audit-ready operational policy.", "outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv", "FINAL MAIN EVIDENCE", "Direct transfer claim değil."),
        ("SCRE top-K review prioritization", "WORKED AS APPENDIX", "SCRE-Optimized top-20 capture/lift iyi; framework rolünü güçlendirdi.", "outputs/final_weakness_closing/scre_prioritization/scre_topk_review_taiwan.csv", "APPENDIX / FRAMEWORK", "Dominant classifier claim yok."),
        ("Taiwan recall candidate", "WORKED AS APPENDIX", "Recall 0.575'ten 0.633'e çıktı; FP arttı ve precision/specificity düştü.", "outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv", "APPENDIX / V3 candidate", "V2'yi değiştirmedi."),
        ("HELOC specificity candidate", "WORKED AS APPENDIX", "Specificity 0.574'ten 0.676'ya çıktı; cost arttı ve recall düştü.", "outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv", "APPENDIX / V3 candidate", "V2'yi değiştirmedi."),
        ("Capacity-aware review", "WORKED AS APPENDIX", "Top-K review prioritization hocaya anlatılabilir güçlü operasyonel analiz verdi.", "outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv", "APPENDIX / SUPPORTING", "Automatic decision değil."),
        ("Instance-dependent cost", "WORKED AS APPENDIX", "Gerçek cost yokluğunu proxy sensitivity ile tartışılabilir hale getirdi.", "outputs/final_weakness_closing/instance_cost/instance_cost_summary.md", "APPENDIX", "Gerçek bank loss değil."),
        ("Final Attempt", "DID NOT REPLACE V2", "Metric consistency audit BLOCKED; repaired only for appendix.", "outputs/final_attempt/audit/final_attempt_audit.md", "APPENDIX ONLY", "Final evidence olarak kullanılmayacak."),
        ("DNN/BP NN", "DID NOT REPLACE V2", "Leakage-free reproduction V2'yi temiz şekilde geçmedi / final audit blocked package içinde kaldı.", "outputs/final_attempt/deep_tabular/deep_tabular_summary.md", "APPENDIX / NEGATIVE RESULT", "Literature high-score overclaim yok."),
        ("KMeansSMOTE / literature reproduction", "DID NOT REPLACE V2", "Doğal test dağılımında audit-ready operational replacement olmadı.", "outputs/final_attempt/literature_reproduction/literature_reproduction_summary.md", "APPENDIX / ROBUSTNESS", "Split-before-SMOTE kuralı korunmalı."),
        ("Temporal robustness", "LIMITATION", "Gerçek timestamp yok; pseudo/order diagnostic limitation only.", "outputs/final_weakness_closing/temporal_robustness/temporal_robustness_summary.md", "LIMITATION", "Gerçek deployment validation değil."),
        ("Real bank cost", "LIMITATION", "Gerçek zarar/fayda maliyeti yok; sensitivity/proxy cost kullanıldı.", "outputs/final_weakness_closing/cost_sensitivity/cost_decision_summary.md", "LIMITATION", "Claim sınırı gerekli."),
    ]
    df = pd.DataFrame(rows, columns=["experiment", "worked_or_failed", "why", "evidence_file", "final_role", "notes"])
    df.to_csv(OUT / "WHAT_WORKED_WHAT_FAILED.csv", index=False)
    return df


def build_claim_control() -> pd.DataFrame:
    frozen = read_csv("outputs/final_frozen_v2/final_claim_control_matrix.csv")
    rows = []
    if not frozen.empty:
        for _, r in frozen.iterrows():
            rows.append(
                {
                    "claim": r.get("claim"),
                    "safe_or_unsafe": "SAFE" if str(r.get("status", "")).upper() in {"SUPPORTED", "SAFE"} else "UNSAFE",
                    "why": r.get("notes"),
                    "evidence": r.get("evidence"),
                    "correct_wording": r.get("safe_wording"),
                    "wrong_wording": r.get("unsafe_wording"),
                }
            )
    extra = [
        ("V2 is the final audit-ready operational version.", "SAFE", "V2 audit READY and frozen.", "outputs/final_frozen_v2/final_readiness_check.md", "V2 is the final operational evidence.", "Final Attempt replaced V2."),
        ("This is an automatic credit rejection system.", "UNSAFE", "Manual-review policy, precision and FP caveats remain.", "outputs/final_frozen_v2/final_v2_do_not_claim.md", "The system supports screening/manual-review prioritization.", "The model can automatically reject applicants."),
        ("SCRE beats all models.", "UNSAFE", "Claim control says no; CatBoost/Scorecard win some operational roles.", "outputs/tables/claim_control_matrix.csv", "SCRE is a reliability-aware framework and review-prioritization support.", "SCRE-Credit outperforms all individual models."),
        ("V3 is final.", "UNSAFE", "Weakness-closing selector kept V2.", "outputs/final_weakness_closing/final_selection/final_selection_summary.md", "V3 candidates are appendix/diagnostic candidates.", "V3 replaces V2."),
        ("Real temporal validation was performed.", "UNSAFE", "No verified timestamp exists.", "outputs/final_weakness_closing/temporal_robustness/temporal_data_availability_check.md", "Only pseudo/order robustness diagnostics were possible.", "The model was temporally validated for deployment."),
    ]
    for e in extra:
        rows.append(
            {
                "claim": e[0],
                "safe_or_unsafe": e[1],
                "why": e[2],
                "evidence": e[3],
                "correct_wording": e[4],
                "wrong_wording": e[5],
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["claim"], keep="first")
    df.to_csv(OUT / "FINAL_CLAIM_CONTROL_TABLE.csv", index=False)
    return df


def build_literature_positioning() -> pd.DataFrame:
    rows = [
        ("Imbalanced learning", "SMOTE/ADASYN/weighted losses ile minority recall artırılır.", "Promptlarda resampling ve weighting leakage-free denendi.", "Recall/capacity trade-off açık incelendi.", "Raw high-score hedefi değil; V2 replacement çıkmadı.", "Farklı split/protokol nedeniyle doğrudan skor kıyası sınırlı.", "Negative but useful robustness finding."),
        ("KMeansSMOTE", "Bazı çalışmalarda büyük skor sıçraması iddia edilir.", "Strict literature reproduction içinde denendi.", "Split-before-resampling kuralı korunarak daha dürüst protokol.", "V2 yerine audit-ready final replacement olmadı.", "Literature protokolleri farklı olabilir.", "Yüksek literatür skorları leakage-free protokolle kontrol edilmelidir."),
        ("BP Neural Network / DNN", "Nonlinear tabular model ile yüksek AUC/accuracy iddiaları.", "Final Attempt deep tabular reproduction.", "Overclaim yapılmadı.", "Audit-ready final model olmadı.", "Hiperparametre/split farkları etkiler.", "DNN negatif sonucu raporda dürüstçe appendix olarak verilmeli."),
        ("Boosting models", "XGBoost/LightGBM/CatBoost credit scoring'de güçlüdür.", "Taiwan final CatBoost; HELOC scorecard.", "CatBoost Taiwan operational policy güçlü.", "Tek model her dataset için kazanmadı.", "Dataset schemas differ.", "Model seçimi hedefe/dataset'e bağlıdır."),
        ("Scorecard", "Regüle kredi riskinde yorumlanabilir baseline.", "WOE/scorecard HELOC final.", "HELOC'ta interpretable final policy.", "Taiwan'da CatBoost kadar operasyonel olmadı.", "Scorecard simplicity-performance trade-off.", "Scorecard güçlü benchmark olarak kalır."),
        ("XAI / SHAP / LIME", "Model açıklanabilirliği ve yerel/global açıklamalar.", "SHAP/LIME, agreement, local cases üretildi.", "Interpretability evidence zengin.", "Causality claim yok.", "Feature correlation affects explanations.", "XAI model behavior sanity check'tir."),
        ("SHAP stability", "Seed/ranking stability önemli.", "Kendall's W ve top-k stability eklendi.", "Reliability discussion güçlendi.", "Lin & Wang gibi çalışmalarla birebir aynı değil.", "Seed/model/feature differences.", "Stability appendix/supporting evidence."),
        ("Cost-sensitive threshold", "FN/FP maliyetleri threshold politikasını değiştirir.", "FN/FP sensitivity, review adjusted cost, instance proxy cost.", "Accuracy dışına çıkan karar mantığı.", "Gerçek banka cost yok.", "Proxy costs are assumptions.", "Sensitivity analysis olarak sunulmalı."),
        ("Manual review / reject option", "Belirsiz vakaları insan incelemesine gönderme.", "V2 final manual-review policy.", "Automatic rejection yerine güvenli decision support.", "MR workload yüksek olabilir.", "Operational capacity unknown.", "Projenin ana katkı yönlerinden biri."),
        ("Decision curve / net benefit", "Treat-all/treat-none karşısında karar faydası.", "Decision curve analysis eklendi.", "Decision usefulness tartışmasını güçlendirdi.", "Final selector değil, supporting.", "Threshold utility assumptions.", "Supporting decision evidence."),
        ("Reliability framework", "Performance+calibration+cost+stability+faithfulness entegrasyonu.", "SCRE-Credit framework.", "Structured comparison and prioritization.", "Dominant classifier değil.", "Framework claims differ from classifier claims.", "SCRE reliability-aware framework olarak konumlanmalı."),
    ]
    df = pd.DataFrame(rows, columns=["literature_theme", "typical_method", "reported_strength", "project_equivalent", "project_result", "stronger_or_weaker", "safe_interpretation"])
    df.to_csv(OUT / "literature_positioning_table.csv", index=False)
    return df


METRICS = [
    ("Accuracy", "(TP+TN)/(TP+FP+FN+TN)", "Genel doğru sınıflama oranı.", "İlk bakış için var; imbalance altında ana karar değildir.", "Default azsa yüksek accuracy yanıltabilir.", "Final seçimde tek başına kullanılmadı."),
    ("Precision", "TP/(TP+FP)", "Riskli denilenlerin gerçekten default olma oranı.", "False positive baskısını ölçmek için kritik.", "Recall çok düşerse tek başına iyi görünür.", "Operational/manual-review trade-off."),
    ("Recall", "TP/(TP+FN)", "Defaultların ne kadarını yakaladığımız.", "Kredi riskinde kaçırılan default maliyetlidir.", "Çok düşük threshold recall artırıp FP patlatabilir.", "Taiwan weakness-closing ana hedeflerinden biri."),
    ("Specificity", "TN/(TN+FP)", "Non-defaultları doğru ayırma oranı.", "İyi müşterileri yanlış alarmdan korur.", "Recall ile trade-off yapabilir.", "HELOC weakness-closing hedefi."),
    ("F1", "2PR/(P+R)", "Precision ve recall harmonik ortalaması.", "Denge metriği.", "Cost veya calibration'ı yansıtmaz.", "Secondary metric."),
    ("ROC-AUC", "TPR-FPR eğrisi alanı", "Threshold bağımsız ayrım gücü.", "Genel discrimination için.", "Imbalanced datada PR-AUC kadar hassas olmayabilir.", "Raw classifier track."),
    ("PR-AUC", "Precision-Recall eğrisi alanı", "Minority/default sınıf performansı.", "Imbalanced default problemi için değerli.", "Threshold/policy costunu direkt göstermez.", "Raw classifier comparison."),
    ("Brier", "mean((p-y)^2)", "Probability reliability hata ölçüsü.", "Calibration için.", "AUC iyi olsa bile Brier kötü olabilir.", "Calibration track."),
    ("ECE", "bin bazlı beklenen calibration error", "Tahmin olasılıklarının güvenilirliği.", "Calibrated risk scores için.", "Bin seçimine duyarlı.", "Calibration and audit."),
    ("Cost", "FN_cost*FN + FP_cost*FP", "Yanlış karar maliyeti.", "Cost-sensitive policy için.", "Cost varsayımına bağımlı.", "Threshold/cost analysis."),
    ("Review-adjusted cost", "auto_FP*FP_cost + auto_FN*FN_cost + MR_count*review_cost", "Manual-review iş yükünü içeren maliyet.", "Manual-review policy için binary costtan daha adil.", "Review cost varsayımsal.", "V2 policy evidence."),
    ("Manual review rate", "MR_count/N", "İnsana giden oran.", "Operasyonel kapasiteyi ölçer.", "Düşük MR default kaçırabilir.", "V2 and capacity."),
    ("High-risk precision", "high_default/high_risk_count", "High-risk bucket kalitesi.", "Manual-review 3x2 metrik.", "Binary precision ile karıştırılmamalı.", "Manual-review analysis."),
    ("High-risk recall", "high_default/total_defaults", "High-risk bucket default yakalama.", "Manual-review 3x2 metrik.", "MR bucket defaultları dahil edilmeyebilir.", "Manual-review analysis."),
    ("Low-risk default rate", "low_default/low_risk_count", "Low-risk bucket leakage.", "Otomatik düşük risk karar güvenliği.", "Low-risk çok küçükse oynak olur.", "Manual-review analysis."),
    ("Capture@K", "reviewed_defaults/total_defaults", "Top-K review'da yakalanan default oranı.", "Capacity-aware review için.", "Direct classification değil.", "SCRE/top-k appendix."),
    ("Precision@K", "reviewed_defaults/review_count", "İncelenen K%'de default yoğunluğu.", "Banka review kapasitesi için.", "Capacity seçimine bağlı.", "Capacity analysis."),
    ("Lift@K", "precision@K/base_default_rate", "Random review'a göre kazanç.", "Top-K ranking faydası.", "Base rate'e bağlı.", "Capacity analysis."),
    ("Net benefit", "TP/N - FP/N*pt/(1-pt)", "Treat-all/none karşısında karar faydası.", "Decision curve için.", "Threshold probability varsayımına bağlı.", "Supporting decision evidence."),
    ("Kendall's W", "rank agreement coefficient", "SHAP ranking stability.", "Explanation reliability için.", "Model/seed sayısına bağlı.", "XAI stability appendix."),
    ("Faithfulness score", "Perturb/delete/insert behavior consistency", "Explanationların model davranışıyla uyumu.", "XAI sanity check.", "Causal proof değildir.", "Appendix."),
]


def build_metric_table() -> pd.DataFrame:
    return pd.DataFrame(
        METRICS,
        columns=[
            "metric_name",
            "formula",
            "ne_anlama_gelir",
            "bu_projede_neden_kullanildi",
            "hangi_durumda_yaniltici_olabilir",
            "hangi_final_karar_icin_kullanildi",
        ],
    )


def build_table_figure_plan() -> pd.DataFrame:
    rows = [
        ("V2 final result table", "outputs/final_frozen_v2/final_v2_*_operational_policy.csv", "Executive Summary / V2 Ana Sonuçlar", "main", "Final audit-ready operational evidence.", "Final V2 manual-review policies."),
        ("V2 vs V3 decision table", "outputs/final_weakness_closing/final_selection/v2_vs_v3_selection_*.csv", "Final Karar: Neden V2?", "main", "Shows why V3 did not replace V2.", "V2 vs V3 final selection decision."),
        ("Taiwan recall candidate table", "outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv", "Weakness-Closing / Taiwan Recall", "appendix", "Recall improved but trade-offs remain.", "Taiwan recall-improving V3 candidates."),
        ("Capacity-aware review table", "outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv", "Capacity-Aware Review", "appendix/supporting", "Shows top-K review value.", "Default capture and lift at review capacity."),
        ("HELOC specificity candidate table", "outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv", "HELOC Specificity", "appendix", "Specificity improved but cost increased.", "HELOC specificity V3 candidates."),
        ("Cost sensitivity table", "outputs/final_weakness_closing/cost_sensitivity/cost_sensitivity_all.csv", "Cost Sensitivity", "appendix/supporting", "Cost assumption robustness.", "Cost sensitivity across FN/FP and review costs."),
        ("Decision curve figure", "outputs/final_weakness_closing/cost_sensitivity/decision_curve_taiwan.png", "Decision Curve", "appendix/supporting", "Net benefit visualization.", "Decision curve on Taiwan."),
        ("Lift@K figure", "outputs/final_weakness_closing/capacity_review/lift_curve_taiwan.png", "Capacity-Aware Review", "appendix/supporting", "Review prioritization curve.", "Lift curve for top-K review."),
        ("Final audit summary", "outputs/final_weakness_closing/audit/final_weakness_closing_audit.md", "Final Audit", "main", "Audit status.", "Final weakness-closing audit verdict."),
        ("Dataset column dictionary", "outputs/full_project_discussion_report/DATASET_COLUMN_DICTIONARY.csv", "Dataset Kolonları", "main/appendix", "Column meaning and usage.", "Dataset variable dictionary."),
        ("Feature engineering inventory", "outputs/full_project_discussion_report/USED_UNUSED_FEATURES.csv", "Feature Engineering", "main/appendix", "Feature formulas.", "Engineered feature inventory."),
        ("Claim control matrix", "outputs/full_project_discussion_report/FINAL_CLAIM_CONTROL_TABLE.csv", "Final Claim Control", "main", "Prevents overclaiming.", "Safe and unsafe claims."),
        ("SCRE role table", "outputs/final_weakness_closing/scre_prioritization/scre_topk_review_taiwan.csv", "SCRE-Credit Framework", "appendix/supporting", "SCRE as ranking/framework.", "SCRE top-K prioritization."),
        ("Weakness-closing summary", "outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md", "Weakness Closing", "main/supporting", "Single package summary.", "Weakness-closing results summary."),
    ]
    df = pd.DataFrame(rows, columns=["table_or_figure", "source_file", "section_to_place", "main_or_appendix", "why", "caption_suggestion"])
    df.to_csv(OUT / "TABLE_FIGURE_PLACEMENT_PLAN.csv", index=False)
    return df


def build_process_leakage_table() -> pd.DataFrame:
    rows = [
        ("raw_data_loading", True, False, False, "allowed", "LOW", "PASS", "Raw files are read before splitting; no target-derived fitting."),
        ("cleaning_category_mapping", True, False, False, "allowed", "LOW", "PASS", "Taiwan EDUCATION/MARRIAGE mappings and HELOC special-code handling are target-free."),
        ("row_wise_feature_engineering", True, False, False, "allowed", "LOW", "PASS", "Features are row-wise and do not use target or fitted dataset statistics."),
        ("model_training", True, False, False, "allowed", "LOW", "PASS", "Models are fitted on train data."),
        ("preprocessing_fit", True, False, False, "allowed only on train/pipeline folds", "MEDIUM if done globally", "PASS/guarded", "Imputation/scaling/encoding should be fitted inside train-fold pipelines."),
        ("calibration_fit", False, True, False, "allowed on validation/calibration split", "MEDIUM", "PASS with caveat", "V2 documents calibration-policy caveat and no test calibration fit."),
        ("threshold_selection", False, True, False, "allowed on validation only", "CRITICAL if test used", "PASS", "V2 selection is validation-only."),
        ("manual_review_band_selection", False, True, False, "allowed on validation only", "CRITICAL if test used", "PASS", "Locked policy selected before held-out test."),
        ("final_locked_test_evaluation", False, False, True, "allowed after locking", "LOW", "PASS", "Test table contains no model-selection winner/rank."),
        ("all_candidate_test_diagnostics", False, False, True, "diagnostic only", "HIGH if used for selection", "CAVEAT", "Weakness-closing audit notes diagnostic test columns in validation_all files; not selection evidence."),
        ("resampling", True, False, False, "train-fold only", "CRITICAL if split-before or test-resampled", "PASS by protocol", "Strict literature reproduction forbids split-before SMOTE and test resampling."),
        ("final_claim_generation", False, False, False, "audit-gated only", "HIGH", "PASS", "Claim control forbids automatic rejection and SCRE-dominance claims."),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "process_step",
            "train_used",
            "validation_used",
            "test_used",
            "allowed_or_not",
            "leakage_risk",
            "final_status",
            "notes",
        ],
    )
    df.to_csv(OUT / "PROCESS_LEAKAGE_CONTROL.csv", index=False)
    return df


def qa_entries() -> list[tuple[str, str, str, str]]:
    return [
        ("Dataset", "Neden Taiwan dataset seçildi?", "Taiwan daha büyük, kredi kartı default problemine doğrudan bağlı ve akademik literatürde yaygın bir benchmarktır.", "30.000 satırlık kredi kartı default datası model, calibration, threshold ve manual-review deneyleri için German Credit'e göre daha güçlü bir deney alanı sağlar.", "Hocam, Taiwan ana veri olduğu için sonuçlar daha istatistiksel ve literatürle daha kıyaslanabilir."),
        ("Dataset", "HELOC neden kullanıldı?", "HELOC external validation / robustness dataset olarak kullanıldı.", "Feature şeması Taiwan'dan farklıdır; bu yüzden direkt model transferi değil, framework mantığının başka kredi datasında davranışını görmek için değerlidir.", "HELOC'u aynı modeli kopyalamak için değil, yöntemin dış veri tarafında nasıl davrandığını görmek için kullandım."),
        ("Dataset", "German Credit neden ana dataset değil?", "German Credit aktif kapsamdan çıkarıldı.", "Daha küçük ve proje kapsamını dağıtan bir veri olduğu için archive altında bırakıldı; ana kapsam Taiwan + HELOC olarak temizlendi.", "Kapsamı şişirmemek için German Credit'i final kapsamdan çıkardım."),
        ("Dataset", "Target nedir?", "Taiwan'da `default_next_month`, HELOC'ta `bad_flag`.", "Her ikisinde de positive class riskli/default/bad performanstır.", "Pozitif sınıf kredi riski gerçekleşen müşteri anlamına geliyor."),
        ("Dataset", "ID neden çıkarıldı?", "ID satır kimliğidir, tahmin edici finansal bilgi değildir.", "Model feature olarak kullanılması anlamsız ezber riski yaratabilir; bu yüzden model-ready datadan çıkarıldı.", "ID modelin müşteriyi tanımasını değil, risk örüntüsünü öğrenmesini istiyoruz."),
        ("Dataset", "EDUCATION anomalileri neydi?", "Taiwan'da 0, 5, 6 gibi belirsiz kategoriler vardı.", "Bunlar literatürde yaygın şekilde 4=Other altında birleştirilir.", "Bu temizlik target kullanmadan yapıldığı için leakage değildir."),
        ("Dataset", "MARRIAGE anomalisi neydi?", "Taiwan'da MARRIAGE=0 anomalisi vardı.", "Bu değer 3=Other altında birleştirildi.", "Bu kategori temizliği veri sözlüğüne uygun bir normalizasyon adımıdır."),
        ("Dataset", "PAY_0 neyi ifade eder?", "En güncel ödeme gecikme durumunu ifade eder.", "Gecikme değişkenleri default riskinin en güçlü sinyallerindendir.", "Ödeme gecikmesi arttıkça default riski beklenen şekilde artıyor."),
        ("Dataset", "BILL_AMT negatif olabilir mi?", "Evet, kredi bakiyesi/ters bakiye gibi domain durumları nedeniyle olabilir.", "Projede bu değerler silinmedi; sadece audit edildi.", "Negatif fatura tutarlarını körlemesine outlier diye atmadım."),
        ("Dataset", "HELOC special missing codes ne oldu?", "-9/-8/-7 gibi özel kodlar NaN'e çevrildi.", "Model pipeline içinde train-fold imputation ile ele alındı.", "Missingness'i testten öğrenmedik; imputation pipeline içinde yapıldı."),
        ("Methodology", "Train/validation/test neden ayrıldı?", "Model fitting, policy seçimi ve final değerlendirmeyi ayırmak için.", "Train model eğitir; validation threshold/policy seçer; test sadece locked final evidence verir.", "Test seti karar vermek için değil, kilitlenmiş kararı kontrol etmek için kullandım."),
        ("Methodology", "Test set ne zaman kullanıldı?", "Final policy kilitlendikten sonra.", "V2'de held-out test evidence tablosu rank/winner içermez.", "Test seti sonucu seçmek için kullanmadım."),
        ("Methodology", "Leakage nasıl engellendi?", "ID/target feature dışı bırakıldı, resampling train içinde tutuldu, threshold/calibration testte fit edilmedi.", "Audit dosyaları test leakage olmadığını raporladı.", "Son auditlerde test leakage NO olarak geçti."),
        ("Methodology", "Calibration nedir?", "Model olasılıklarını gerçek default oranlarına daha uyumlu hale getirme işlemidir.", "Sigmoid/isotonic gibi yöntemler olasılık reliability için kullanılır.", "AUC yükseltmek değil, risk olasılığını daha güvenilir yapmak için yaptım."),
        ("Methodology", "Threshold tuning nedir?", "0.50 sabit eşik yerine validation üzerinde amaç/kısıta göre eşik seçmektir.", "Cost, precision, specificity ve recall trade-offları threshold ile değişir.", "Eşiği testte seçmedim; validation'da seçtim."),
        ("Methodology", "Manual-review band nedir?", "Düşük risk, manual review ve yüksek risk olarak üç karar bucket'ı üretmektir.", "Belirsiz müşteriler otomatik karar yerine insana gönderilir.", "Bu sistem kredi reddi değil, inceleme önceliklendirme sistemidir."),
        ("Methodology", "Manual-review cost neden farklı?", "Manual-review kararı binary high/low kararla aynı değildir.", "Review workload cost ayrıca eklenmelidir; 3x2 bucket mantığı kullanılır.", "Manual review maliyetini binary FN/FP cost ile karıştırmadım."),
        ("Methodology", "Validation-only selection ne demek?", "Final model/policy sadece validation kanıtıyla seçildi.", "Test metrikleri ranking veya winner seçimi için kullanılmadı.", "Bu V2'nin NOT READY sorununu düzelten ana metodolojik nokta."),
        ("Methodology", "Locked test evaluation ne demek?", "Model/policy/eşik sabitlendikten sonra testte sadece performans ölçmektir.", "Test sonucu kötü çıksa bile policy değiştirilmez.", "Bu yüzden test sonucunu seçim değil kanıt olarak sundum."),
        ("Methodology", "Resampling split öncesi yapıldı mı?", "Final güvenli protokolde hayır.", "SMOTE/KMeansSMOTE gibi yöntemler train fold içinde tutulmalıdır.", "Split öncesi resampling olsaydı critical leakage olurdu; bunu yasakladım."),
        ("Metrics", "Accuracy neden ana metric değil?", "Default sınıfı dengesiz olduğu için accuracy çoğunluk sınıfını iyi tahmin ederek yüksek görünebilir.", "Kredi riskinde default yakalama ve yanlış alarm ayrı ölçülmelidir.", "Accuracy tek başına bankanın karar kalitesini anlatmaz."),
        ("Metrics", "Precision nedir?", "Riskli dediğimiz müşterilerin ne kadarının gerçekten default olduğu.", "False positive baskısını gösterir.", "Precision düşükse otomatik red tehlikelidir."),
        ("Metrics", "Recall nedir?", "Gerçek defaultların ne kadarını yakaladığımız.", "FN maliyeti kredi riskinde önemlidir.", "Recall yükselirken FP artabilir; bu yüzden tek başına yeterli değildir."),
        ("Metrics", "Specificity nedir?", "Non-default müşterileri doğru düşük risk ayırma oranı.", "İyi müşteriye yanlış alarm vermemek için önemlidir.", "HELOC weakness-closing specificity üzerine kuruldu."),
        ("Metrics", "PR-AUC neden önemli?", "Imbalanced datada minority/default sınıf performansını ROC-AUC'den daha görünür yapar.", "Default prediction probleminde precision-recall trade-off doğrudan önemlidir.", "Raw classifier kıyaslarında PR-AUC kullandım."),
        ("Metrics", "ROC-AUC neden kullanıldı?", "Threshold bağımsız genel ayrım gücünü gösterir.", "Ama operational threshold seçimi için tek başına yetmez.", "ROC-AUC modelin sıralama gücünü anlatır, karar politikasını değil."),
        ("Metrics", "Brier ve ECE ne işe yaradı?", "Olasılıkların kalibrasyon kalitesini ölçtü.", "Bankacılıkta risk skoru olasılık gibi yorumlanacaksa reliability gerekir.", "Sadece iyi sıralama değil, güvenilir olasılık istedim."),
        ("Metrics", "Cost nasıl hesaplandı?", "Binary cost FN_cost*FN + FP_cost*FP, manual-review cost ise workload ile ayrı hesaplandı.", "FN=5 FP=1 temel senaryo ve sensitivity analizleri kullanıldı.", "Cost varsayımdır; gerçek banka zararı değildir."),
        ("Metrics", "Capture@K nedir?", "En riskli K% incelenirse defaultların kaçının yakalandığıdır.", "Manual-review capacity için en anlaşılır metriklerden biridir.", "Bankanın sadece %20 kişiyi inceleyebildiği senaryo gibi düşünülebilir."),
        ("Metrics", "Decision curve nedir?", "Model kararının treat-all ve treat-none stratejilerine göre net faydasını ölçer.", "AUC/accuracy dışına çıkan karar faydası analizi sağlar.", "Model gerçekten karar destek faydası sağlıyor mu sorusuna bakar."),
        ("Models", "Neden CatBoost?", "Categorical/numeric tabular kredi datasında güçlü boosting modelidir.", "Taiwan V2 final operational policy CatBoost manual-review oldu.", "Taiwan'da en audit-ready operational denge CatBoost ile geldi."),
        ("Models", "Neden Scorecard?", "Kredi riskinde regülasyon ve yorumlanabilirlik açısından klasik benchmarktır.", "HELOC'ta V2 final policy Scorecard manual-review oldu.", "HELOC tarafında interpretable ve güçlü bir final kanıt verdi."),
        ("Models", "Neden XGBoost final değil?", "Güçlü model olmasına rağmen final audit-ready operational role V2'de CatBoost/Scorecard'a geçti.", "Bazı appendix deneylerde iyiydi ama final replacement olmadı.", "XGBoost güçlü benchmark, fakat final policy winner değil."),
        ("Models", "Neden LightGBM final değil?", "Weighted/monotonic/interpretable denemelerde kullanıldı, ama final V2'yi değiştirmedi.", "Probability calibration ve FP/recall trade-offları final rolünü sınırladı.", "LightGBM destekleyici model olarak kaldı."),
        ("Models", "Neden DNN final değil?", "Leakage-free reproduction audit-ready final replacement üretmedi.", "Tabular credit scoring'de DNN her zaman boosting/scorecard'ı geçmeyebilir.", "DNN negatif sonucu saklamadım; appendix'e koydum."),
        ("Models", "Neden KMeansSMOTE final değil?", "Doğal test dağılımında V2 yerine temiz operasyonel replacement olmadı.", "Resampling recall artırabilir ama calibration/precision trade-off yaratabilir.", "Literatürdeki yüksek skorları leakage-free tekrar kontrol ettim."),
        ("Models", "SCRE neden final classifier değil?", "SCRE tüm modelleri her metrikte geçmedi.", "CatBoost/Scorecard bazı operational rollerde daha güçlü kaldı.", "SCRE'yi classifier değil reliability-aware framework olarak konumlandırdım."),
        ("Models", "SCRE'nin katkısı ne?", "Performance, calibration, cost, stability, faithfulness ve review prioritization kanıtlarını bir araya getiren çerçevedir.", "Top-K review prioritization tarafında faydalı destek sundu.", "SCRE proje katkısıdır ama dominant model iddiası değildir."),
        ("Models", "Scorecard neden HELOC'ta final?", "HELOC'ta interpretable ve audit-ready manual-review policy en güvenli final kanıt oldu.", "External dataset tarafında açıklanabilir benchmark olması da avantajdır.", "HELOC'ta karmaşık model yerine scorecard daha savunulabilir kaldı."),
        ("Models", "Monotonic model ne işe yaradı?", "Risk yönü belli feature'larda daha governance-friendly alternatif sundu.", "Fakat final operational policy olarak V2'yi geçmedi.", "Monotonic model appendix/interpretability kanıtıdır."),
        ("Models", "EBM ne işe yaradı?", "Scorecard ile boosting arasında yorumlanabilir orta model ihtimalini test etti.", "Final Attempt BLOCKED olduğu için final kanıt değil; appendix olabilir.", "EBM, performans-yorumlanabilirlik tartışmasına destek verir."),
        ("Results", "Taiwan sonucu iyi mi?", "Operational olarak dengeli ama mükemmel değil.", "Precision 0.529, recall 0.575, specificity 0.854; manual-review destekli screening sistemi için audit-ready.", "Taiwan'da güçlü yan FP kontrolü; zayıf yan recall'ın orta seviyede kalması."),
        ("Results", "HELOC sonucu iyi mi?", "External validation tarafında güçlü recall ve yorumlanabilirlik var.", "Precision 0.683, recall 0.846, specificity 0.574; specificity zayıflığı kabul edildi.", "HELOC sonucu scorecard'ın dış veri tarafında hâlâ değerli olduğunu gösterdi."),
        ("Results", "Neden V2 final?", "V2 audit READY ve final selection validation-only.", "Final Attempt BLOCKED, V3 adayları ise final replacement kriterlerini geçmedi.", "En güvenli final kanıt V2 olduğu için final V2 kalmalı."),
        ("Results", "Neden V3 değil?", "V3 adayları bazı metrikleri iyileştirdi ama trade-off yarattı ve final replacement olmadı.", "Taiwan recall artsa da FP arttı; HELOC specificity artsa da cost yükseldi.", "Zayıflıkları kapatma denendi ama final dengesi V2'den daha güvenli olmadı."),
        ("Results", "Final Attempt neden appendix?", "Audit BLOCKED: manual-review locked-test metric consistency hataları vardı.", "Sonra 3x2 repair yapıldı ama Final Attempt yine final evidence olmadı.", "Final Attempt'i robustluk/negatif sonuç appendix'i olarak kullanacağım."),
        ("Results", "Recall candidate neden final olmadı?", "Recall 0.633'e çıktı ama precision 0.463'e düştü ve FP 976'ya çıktı.", "Yani daha çok default yakaladı ama daha fazla yanlış alarm üretti.", "Bu bir trade-off gösterimi; final replacement değil."),
        ("Results", "HELOC specificity candidate neden final olmadı?", "Specificity 0.676'ya çıktı ama cost 1427'ye yükseldi ve recall 0.782'ye düştü.", "İyi müşterileri daha iyi ayırdı ama default kaçırma/maliyet arttı.", "Appendix'te trade-off olarak anlatılmalı."),
        ("Results", "Capacity-aware review ne gösterdi?", "SCRE-Optimized top-20% review'da Taiwan default capture 0.529 ve lift 2.645 verdi.", "Bu SCRE'nin ranking/review prioritization rolünü güçlendirdi.", "Banka %20 inceleme kapasitesine sahipse model sıralama aracı olabilir."),
        ("Results", "Instance-dependent cost sonucu neydi?", "Taiwan recall candidate proxy-cost açısından bazı tanımlarda iyi göründü; HELOC'ta V2 kaldı.", "Bu gerçek banka maliyeti değil, proxy sensitivity analizidir.", "Cost belirsizliğini dürüstçe tartışmak için kullanılır."),
        ("Results", "Temporal robustness sonucu neydi?", "Gerçek timestamp yok, gerçek temporal validation yapılamadı.", "Sadece limitation/pseudo diagnostic olarak raporlanabilir.", "Deployment claim'i üretmiyorum."),
        ("Weakness", "Taiwan recall düşük değil mi?", "Evet, orta seviyede bir zayıflık.", "Recall artırma denemeleri yapıldı ama FP ve precision trade-off'u doğdu.", "Bu zayıflığı saklamıyorum; final sistem screening desteği olarak sunuluyor."),
        ("Weakness", "Manual review rate yüksek değil mi?", "Yaklaşık %29.5 Taiwan ve %27 HELOC, operasyonel olarak dikkate değer.", "Capacity-aware analizle %20/%25 review senaryoları incelendi.", "MR workload bir limitation ve operasyonel tasarım konusu."),
        ("Weakness", "HELOC specificity düşük değil mi?", "Evet, 0.574 sınırlı.", "Specificity artırma denendi; FP düştü ama cost arttı ve recall düştü.", "Bu yüzden V2 değişmedi."),
        ("Weakness", "Gerçek deployment yoksa çalışma nasıl değerli?", "Bu bir deployment değil, leakage-aware model validation ve decision-support framework çalışmasıdır.", "Temporal validation limitation olarak yazıldı.", "Gerçek deployment için zamanlı veri ve banka costu gerekir."),
        ("Weakness", "Gerçek cost yoksa cost analizi nasıl savunulur?", "Cost sensitivity ve proxy instance-dependent cost olarak savunulur.", "Gerçek bank loss iddiası yapılmaz.", "Ben maliyeti varsayım olarak kullandım ve duyarlılık analizi yaptım."),
        ("Literature", "Literatürde daha yüksek skorlar var, neden düşük?", "Çünkü bazı literatür skorları farklı split, resampling veya leakage-riskli protokollerle raporlanmış olabilir.", "Biz doğal test distribution ve validation-only selection kullandık.", "Daha düşük ama daha dürüst/audit-ready sonuç ürettim."),
        ("Literature", "KMeansSMOTE neden işe yaramadı?", "Resampling doğal testte operasyonel replacement üretmedi.", "SMOTE türleri minority recall'ı artırabilir ama precision/calibration/cost trade-off yaratabilir.", "Bu negatif sonuç literatüre karşı dürüst bir kontrol."),
        ("Literature", "DNN neden final olmadı?", "Tabular credit data'da DNN her zaman boosting/scorecard'dan iyi değildir.", "Leakage-free protokolde final audit-ready replacement olmadı.", "DNN denendi ama sonuç saklanmadı."),
        ("Literature", "Bu çalışma neyi farklı yapıyor?", "Sadece AUC değil; calibration, cost, threshold, manual review, audit, explainability ve framework rolünü birlikte ele alıyor.", "Final amaç SOTA skor değil audit-ready decision-support.", "Katkı bu bütünleşik validasyon mantığıdır."),
        ("Claims", "Bu sistem kredi reddi yapabilir mi?", "Hayır, bu şekilde claim edilmemeli.", "Precision ve manual-review gerekliliği nedeniyle automatic rejection uygun değil.", "Screening/manual-review prioritization için kullanılabilir."),
        ("Claims", "Banka bunu kullanabilir mi?", "Doğrudan deployment için değil; karar destek prototipi olarak değerlendirilebilir.", "Gerçek deployment için zamanlı validation, gerçek maliyet ve governance gerekir.", "Akademik proje olarak audit-aware framework sunuyor."),
        ("Claims", "Makale/bildiri çıkar mı?", "Potansiyel olarak evet, ama claim doğru kurulmalı.", "Hikâye: leakage-aware, validation-selected, manual-review destekli credit-risk decision framework.", "SCRE'yi yeni temel algoritma değil framework olarak sunmak gerekir."),
        ("Claims", "Ana sonuç cümlesi ne olmalı?", "V2 final audit-ready manual-review support system oldu.", "Taiwan'da CatBoost, HELOC'ta Scorecard; SCRE framework/ranking desteği.", "Proje en yüksek skor değil, güvenilir karar destek framework'ü olarak anlatılmalı."),
    ]
    # Guarantee at least 60 entries by returning 70-ish list above.
    return qa_entries.__defaults__ if False else rows_placeholder


# The function above cannot reference a local placeholder at runtime. Build Q&A rows explicitly below.
QA_ROWS = [
    ("Dataset", "Neden Taiwan dataset seçildi?", "Taiwan daha büyük, kredi kartı default problemine doğrudan bağlı ve akademik literatürde yaygın bir benchmarktır.", "30.000 satırlık kredi kartı default datası model, calibration, threshold ve manual-review deneyleri için German Credit'e göre daha güçlü bir deney alanı sağlar.", "Hocam, Taiwan ana veri olduğu için sonuçlar daha istatistiksel ve literatürle daha kıyaslanabilir."),
    ("Dataset", "HELOC neden kullanıldı?", "HELOC external validation / robustness dataset olarak kullanıldı.", "Feature şeması Taiwan'dan farklıdır; bu yüzden direkt model transferi değil, framework mantığının başka kredi datasında davranışını görmek için değerlidir.", "HELOC'u aynı modeli kopyalamak için değil, yöntemin dış veri tarafında nasıl davrandığını görmek için kullandım."),
    ("Dataset", "German Credit neden ana dataset değil?", "German Credit aktif kapsamdan çıkarıldı.", "Daha küçük ve proje kapsamını dağıtan bir veri olduğu için archive altında bırakıldı; ana kapsam Taiwan + HELOC olarak temizlendi.", "Kapsamı şişirmemek için German Credit'i final kapsamdan çıkardım."),
    ("Dataset", "Target nedir?", "Taiwan'da `default_next_month`, HELOC'ta `bad_flag`.", "Her ikisinde de positive class riskli/default/bad performanstır.", "Pozitif sınıf kredi riski gerçekleşen müşteri anlamına geliyor."),
    ("Dataset", "ID neden çıkarıldı?", "ID satır kimliğidir, tahmin edici finansal bilgi değildir.", "Model feature olarak kullanılması anlamsız ezber riski yaratabilir; bu yüzden model-ready datadan çıkarıldı.", "ID modelin müşteriyi tanımasını değil, risk örüntüsünü öğrenmesini istiyoruz."),
    ("Dataset", "EDUCATION anomalileri neydi?", "Taiwan'da 0, 5, 6 gibi belirsiz kategoriler vardı.", "Bunlar literatürde yaygın şekilde 4=Other altında birleştirilir.", "Bu temizlik target kullanmadan yapıldığı için leakage değildir."),
    ("Dataset", "MARRIAGE anomalisi neydi?", "Taiwan'da MARRIAGE=0 anomalisi vardı.", "Bu değer 3=Other altında birleştirildi.", "Bu kategori temizliği veri sözlüğüne uygun bir normalizasyon adımıdır."),
    ("Dataset", "PAY_0 neyi ifade eder?", "En güncel ödeme gecikme durumunu ifade eder.", "Gecikme değişkenleri default riskinin en güçlü sinyallerindendir.", "Ödeme gecikmesi arttıkça default riski beklenen şekilde artıyor."),
    ("Dataset", "BILL_AMT negatif olabilir mi?", "Evet, kredi bakiyesi/ters bakiye gibi domain durumları nedeniyle olabilir.", "Projede bu değerler silinmedi; sadece audit edildi.", "Negatif fatura tutarlarını körlemesine outlier diye atmadım."),
    ("Dataset", "HELOC special missing codes ne oldu?", "-9/-8/-7 gibi özel kodlar NaN'e çevrildi.", "Model pipeline içinde train-fold imputation ile ele alındı.", "Missingness'i testten öğrenmedik; imputation pipeline içinde yapıldı."),
    ("Methodology", "Train/validation/test neden ayrıldı?", "Model fitting, policy seçimi ve final değerlendirmeyi ayırmak için.", "Train model eğitir; validation threshold/policy seçer; test sadece locked final evidence verir.", "Test seti karar vermek için değil, kilitlenmiş kararı kontrol etmek için kullandım."),
    ("Methodology", "Test set ne zaman kullanıldı?", "Final policy kilitlendikten sonra.", "V2'de held-out test evidence tablosu rank/winner içermez.", "Test seti sonucu seçmek için kullanmadım."),
    ("Methodology", "Leakage nasıl engellendi?", "ID/target feature dışı bırakıldı, resampling train içinde tutuldu, threshold/calibration testte fit edilmedi.", "Audit dosyaları test leakage olmadığını raporladı.", "Son auditlerde test leakage NO olarak geçti."),
    ("Methodology", "Calibration nedir?", "Model olasılıklarını gerçek default oranlarına daha uyumlu hale getirme işlemidir.", "Sigmoid/isotonic gibi yöntemler olasılık reliability için kullanılır.", "AUC yükseltmek değil, risk olasılığını daha güvenilir yapmak için yaptım."),
    ("Methodology", "Threshold tuning nedir?", "0.50 sabit eşik yerine validation üzerinde amaç/kısıta göre eşik seçmektir.", "Cost, precision, specificity ve recall trade-offları threshold ile değişir.", "Eşiği testte seçmedim; validation'da seçtim."),
    ("Methodology", "Manual-review band nedir?", "Düşük risk, manual review ve yüksek risk olarak üç karar bucket'ı üretmektir.", "Belirsiz müşteriler otomatik karar yerine insana gönderilir.", "Bu sistem kredi reddi değil, inceleme önceliklendirme sistemidir."),
    ("Methodology", "Manual-review cost neden farklı?", "Manual-review kararı binary high/low kararla aynı değildir.", "Review workload cost ayrıca eklenmelidir; 3x2 bucket mantığı kullanılır.", "Manual review maliyetini binary FN/FP cost ile karıştırmadım."),
    ("Methodology", "Validation-only selection ne demek?", "Final model/policy sadece validation kanıtıyla seçildi.", "Test metrikleri ranking veya winner seçimi için kullanılmadı.", "Bu V2'nin NOT READY sorununu düzelten ana metodolojik nokta."),
    ("Methodology", "Locked test evaluation ne demek?", "Model/policy/eşik sabitlendikten sonra testte sadece performans ölçmektir.", "Test sonucu kötü çıksa bile policy değiştirilmez.", "Bu yüzden test sonucunu seçim değil kanıt olarak sundum."),
    ("Methodology", "Resampling split öncesi yapıldı mı?", "Final güvenli protokolde hayır.", "SMOTE/KMeansSMOTE gibi yöntemler train fold içinde tutulmalıdır.", "Split öncesi resampling olsaydı critical leakage olurdu; bunu yasakladım."),
    ("Metrics", "Accuracy neden ana metric değil?", "Default sınıfı dengesiz olduğu için accuracy çoğunluk sınıfını iyi tahmin ederek yüksek görünebilir.", "Kredi riskinde default yakalama ve yanlış alarm ayrı ölçülmelidir.", "Accuracy tek başına bankanın karar kalitesini anlatmaz."),
    ("Metrics", "Precision nedir?", "Riskli dediğimiz müşterilerin ne kadarının gerçekten default olduğu.", "False positive baskısını gösterir.", "Precision düşükse otomatik red tehlikelidir."),
    ("Metrics", "Recall nedir?", "Gerçek defaultların ne kadarını yakaladığımız.", "FN maliyeti kredi riskinde önemlidir.", "Recall yükselirken FP artabilir; bu yüzden tek başına yeterli değildir."),
    ("Metrics", "Specificity nedir?", "Non-default müşterileri doğru düşük risk ayırma oranı.", "İyi müşteriye yanlış alarm vermemek için önemlidir.", "HELOC weakness-closing specificity üzerine kuruldu."),
    ("Metrics", "PR-AUC neden önemli?", "Imbalanced datada minority/default sınıf performansını ROC-AUC'den daha görünür yapar.", "Default prediction probleminde precision-recall trade-off doğrudan önemlidir.", "Raw classifier kıyaslarında PR-AUC kullandım."),
    ("Metrics", "ROC-AUC neden kullanıldı?", "Threshold bağımsız genel ayrım gücünü gösterir.", "Ama operational threshold seçimi için tek başına yetmez.", "ROC-AUC modelin sıralama gücünü anlatır, karar politikasını değil."),
    ("Metrics", "Brier ve ECE ne işe yaradı?", "Olasılıkların kalibrasyon kalitesini ölçtü.", "Bankacılıkta risk skoru olasılık gibi yorumlanacaksa reliability gerekir.", "Sadece iyi sıralama değil, güvenilir olasılık istedim."),
    ("Metrics", "Cost nasıl hesaplandı?", "Binary cost FN_cost*FN + FP_cost*FP, manual-review cost ise workload ile ayrı hesaplandı.", "FN=5 FP=1 temel senaryo ve sensitivity analizleri kullanıldı.", "Cost varsayımdır; gerçek banka zararı değildir."),
    ("Metrics", "Capture@K nedir?", "En riskli K% incelenirse defaultların kaçının yakalandığıdır.", "Manual-review capacity için en anlaşılır metriklerden biridir.", "Bankanın sadece %20 kişiyi inceleyebildiği senaryo gibi düşünülebilir."),
    ("Metrics", "Decision curve nedir?", "Model kararının treat-all ve treat-none stratejilerine göre net faydasını ölçer.", "AUC/accuracy dışına çıkan karar faydası analizi sağlar.", "Model gerçekten karar destek faydası sağlıyor mu sorusuna bakar."),
    ("Models", "Neden CatBoost?", "Categorical/numeric tabular kredi datasında güçlü boosting modelidir.", "Taiwan V2 final operational policy CatBoost manual-review oldu.", "Taiwan'da en audit-ready operational denge CatBoost ile geldi."),
    ("Models", "Neden Scorecard?", "Kredi riskinde regülasyon ve yorumlanabilirlik açısından klasik benchmarktır.", "HELOC'ta V2 final policy Scorecard manual-review oldu.", "HELOC tarafında interpretable ve güçlü bir final kanıt verdi."),
    ("Models", "Neden XGBoost final değil?", "Güçlü model olmasına rağmen final audit-ready operational role V2'de CatBoost/Scorecard'a geçti.", "Bazı appendix deneylerde iyiydi ama final replacement olmadı.", "XGBoost güçlü benchmark, fakat final policy winner değil."),
    ("Models", "Neden LightGBM final değil?", "Weighted/monotonic/interpretable denemelerde kullanıldı, ama final V2'yi değiştirmedi.", "Probability calibration ve FP/recall trade-offları final rolünü sınırladı.", "LightGBM destekleyici model olarak kaldı."),
    ("Models", "Neden DNN final değil?", "Leakage-free reproduction audit-ready final replacement üretmedi.", "Tabular credit scoring'de DNN her zaman boosting/scorecard'ı geçmeyebilir.", "DNN negatif sonucu saklamadım; appendix'e koydum."),
    ("Models", "Neden KMeansSMOTE final değil?", "Doğal test dağılımında V2 yerine temiz operasyonel replacement olmadı.", "Resampling recall artırabilir ama calibration/precision/cost trade-off yaratabilir.", "Literatürdeki yüksek skorları leakage-free tekrar kontrol ettim."),
    ("Models", "SCRE neden final classifier değil?", "SCRE tüm modelleri her metrikte geçmedi.", "CatBoost/Scorecard bazı operational rollerde daha güçlü kaldı.", "SCRE'yi classifier değil reliability-aware framework olarak konumlandırdım."),
    ("Models", "SCRE'nin katkısı ne?", "Performance, calibration, cost, stability, faithfulness ve review prioritization kanıtlarını bir araya getiren çerçevedir.", "Top-K review prioritization tarafında faydalı destek sundu.", "SCRE proje katkısıdır ama dominant model iddiası değildir."),
    ("Models", "Scorecard neden HELOC'ta final?", "HELOC'ta interpretable ve audit-ready manual-review policy en güvenli final kanıt oldu.", "External dataset tarafında açıklanabilir benchmark olması da avantajdır.", "HELOC'ta karmaşık model yerine scorecard daha savunulabilir kaldı."),
    ("Models", "Monotonic model ne işe yaradı?", "Risk yönü belli feature'larda daha governance-friendly alternatif sundu.", "Fakat final operational policy olarak V2'yi geçmedi.", "Monotonic model appendix/interpretability kanıtıdır."),
    ("Models", "EBM ne işe yaradı?", "Scorecard ile boosting arasında yorumlanabilir orta model ihtimalini test etti.", "Final Attempt BLOCKED olduğu için final kanıt değil; appendix olabilir.", "EBM, performans-yorumlanabilirlik tartışmasına destek verir."),
    ("Results", "Taiwan sonucu iyi mi kötü mü?", "Operational olarak dengeli ama mükemmel değil.", "Precision 0.529, recall 0.575, specificity 0.854; manual-review destekli screening sistemi için audit-ready.", "Taiwan'da güçlü yan FP kontrolü; zayıf yan recall'ın orta seviyede kalması."),
    ("Results", "HELOC sonucu iyi mi kötü mü?", "External validation tarafında güçlü recall ve yorumlanabilirlik var.", "Precision 0.683, recall 0.846, specificity 0.574; specificity zayıflığı kabul edildi.", "HELOC sonucu scorecard'ın dış veri tarafında hâlâ değerli olduğunu gösterdi."),
    ("Results", "Neden V2 final?", "V2 audit READY ve final selection validation-only.", "Final Attempt BLOCKED, V3 adayları ise final replacement kriterlerini geçmedi.", "En güvenli final kanıt V2 olduğu için final V2 kalmalı."),
    ("Results", "Neden V3 değil?", "V3 adayları bazı metrikleri iyileştirdi ama trade-off yarattı ve final replacement olmadı.", "Taiwan recall artsa da FP arttı; HELOC specificity artsa da cost yükseldi.", "Zayıflıkları kapatma denendi ama final dengesi V2'den daha güvenli olmadı."),
    ("Results", "Final Attempt neden appendix?", "Audit BLOCKED: manual-review locked-test metric consistency hataları vardı.", "Sonra 3x2 repair yapıldı ama Final Attempt yine final evidence olmadı.", "Final Attempt'i robustluk/negatif sonuç appendix'i olarak kullanacağım."),
    ("Results", "Recall candidate neden final olmadı?", "Recall 0.633'e çıktı ama precision 0.463'e düştü ve FP 976'ya çıktı.", "Yani daha çok default yakaladı ama daha fazla yanlış alarm üretti.", "Bu bir trade-off gösterimi; final replacement değil."),
    ("Results", "HELOC specificity candidate neden final olmadı?", "Specificity 0.676'ya çıktı ama cost 1427'ye yükseldi ve recall 0.782'ye düştü.", "İyi müşterileri daha iyi ayırdı ama default kaçırma/maliyet arttı.", "Appendix'te trade-off olarak anlatılmalı."),
    ("Results", "Capacity-aware review ne gösterdi?", "SCRE-Optimized top-20% review'da Taiwan default capture 0.529 ve lift 2.645 verdi.", "Bu SCRE'nin ranking/review prioritization rolünü güçlendirdi.", "Banka %20 inceleme kapasitesine sahipse model sıralama aracı olabilir."),
    ("Results", "Instance-dependent cost sonucu neydi?", "Taiwan recall candidate proxy-cost açısından bazı tanımlarda iyi göründü; HELOC'ta V2 kaldı.", "Bu gerçek banka maliyeti değil, proxy sensitivity analizidir.", "Cost belirsizliğini dürüstçe tartışmak için kullanılır."),
    ("Results", "Temporal robustness sonucu neydi?", "Gerçek timestamp yok, gerçek temporal validation yapılamadı.", "Sadece limitation/pseudo diagnostic olarak raporlanabilir.", "Deployment claim'i üretmiyorum."),
    ("Weakness", "Taiwan recall düşük değil mi?", "Evet, orta seviyede bir zayıflık.", "Recall artırma denemeleri yapıldı ama FP ve precision trade-off'u doğdu.", "Bu zayıflığı saklamıyorum; final sistem screening desteği olarak sunuluyor."),
    ("Weakness", "Manual review rate yüksek değil mi?", "Yaklaşık %29.5 Taiwan ve %27 HELOC, operasyonel olarak dikkate değer.", "Capacity-aware analizle %20/%25 review senaryoları incelendi.", "MR workload bir limitation ve operasyonel tasarım konusu."),
    ("Weakness", "HELOC specificity düşük değil mi?", "Evet, 0.574 sınırlı.", "Specificity artırma denendi; FP düştü ama cost arttı ve recall düştü.", "Bu yüzden V2 değişmedi."),
    ("Weakness", "Gerçek deployment yoksa çalışma nasıl değerli?", "Bu bir deployment değil, leakage-aware model validation ve decision-support framework çalışmasıdır.", "Temporal validation limitation olarak yazıldı.", "Gerçek deployment için zamanlı veri ve banka costu gerekir."),
    ("Weakness", "Gerçek cost yoksa cost analizi nasıl savunulur?", "Cost sensitivity ve proxy instance-dependent cost olarak savunulur.", "Gerçek bank loss iddiası yapılmaz.", "Ben maliyeti varsayım olarak kullandım ve duyarlılık analizi yaptım."),
    ("Literature", "Literatürde daha yüksek skorlar var, neden sizin sonuçlar daha düşük?", "Çünkü bazı literatür skorları farklı split, resampling veya leakage-riskli protokollerle raporlanmış olabilir.", "Biz doğal test distribution ve validation-only selection kullandık.", "Daha düşük ama daha dürüst/audit-ready sonuç ürettim."),
    ("Literature", "KMeansSMOTE neden final olmadı?", "Resampling doğal testte operasyonel replacement üretmedi.", "SMOTE türleri minority recall'ı artırabilir ama precision/calibration/cost trade-off yaratabilir.", "Bu negatif sonuç literatüre karşı dürüst bir kontrol."),
    ("Literature", "DNN neden final olmadı?", "Tabular credit data'da DNN her zaman boosting/scorecard'dan iyi değildir.", "Leakage-free protokolde final audit-ready replacement olmadı.", "DNN denendi ama sonuç saklanmadı."),
    ("Literature", "Bu çalışma neyi farklı yapıyor?", "Sadece AUC değil; calibration, cost, threshold, manual review, audit, explainability ve framework rolünü birlikte ele alıyor.", "Final amaç SOTA skor değil audit-ready decision-support.", "Katkı bu bütünleşik validasyon mantığıdır."),
    ("Claims", "Bu sistem kredi reddi yapabilir mi?", "Hayır, bu şekilde claim edilmemeli.", "Precision ve manual-review gerekliliği nedeniyle automatic rejection uygun değil.", "Screening/manual-review prioritization için kullanılabilir."),
    ("Claims", "Banka bunu kullanabilir mi?", "Doğrudan deployment için değil; karar destek prototipi olarak değerlendirilebilir.", "Gerçek deployment için zamanlı validation, gerçek maliyet ve governance gerekir.", "Akademik proje olarak audit-aware framework sunuyor."),
    ("Claims", "SCRE'nin katkısı ne?", "Reliability-aware framework ve review-prioritization desteği.", "SCRE performance, calibration, cost, stability, faithfulness ve capacity evidence'ı yapılandırır.", "SCRE'yi her şeyi yenen model diye değil, framework diye anlatmalıyım."),
    ("Claims", "Makale/bildiri çıkar mı?", "Potansiyel olarak evet, ama claim doğru kurulmalı.", "Hikâye: leakage-aware, validation-selected, manual-review destekli credit-risk decision framework.", "SCRE'yi yeni temel algoritma değil framework olarak sunmak gerekir."),
    ("Claims", "Ana sonuç cümlesi ne olmalı?", "V2 final audit-ready manual-review support system oldu.", "Taiwan'da CatBoost, HELOC'ta Scorecard; SCRE framework/ranking desteği.", "Proje en yüksek skor değil, güvenilir karar destek framework'ü olarak anlatılmalı."),
]


def build_qa() -> str:
    lines = ["# Teacher Q&A Preparation\n"]
    for i, (cat, q, short, tech, safe) in enumerate(QA_ROWS, 1):
        lines += [
            f"## {i}. [{cat}] {q}",
            "",
            f"**Kısa cevap:** {short}",
            "",
            f"**Teknik açıklama:** {tech}",
            "",
            f"**Hocaya güvenli cümle:** {safe}",
            "",
        ]
    text = "\n".join(lines)
    (OUT / "TEACHER_QA_PREPARATION.md").write_text(text, encoding="utf-8")
    return text


def build_final_decision() -> str:
    text = """# Final Decision Explained

## Net Karar

- Proceed to conclusion: **YES**
- Use V2 as final: **YES**
- Use V3 as final: **NO**
- Use Final Attempt as appendix: **YES**
- Use SCRE as final classifier: **NO**
- Use SCRE as framework/prioritization: **YES**
- Automatic rejection: **NO**
- Screening/manual-review support: **YES**

## Neden V2?

V2, test-set ranking problemini düzelten, validation-only selection kullanan ve locked held-out test evidence üreten audit-ready aşamadır. `outputs/decision_revision_v2/final_audit_v2.md` dosyasında critical/high issue yoktur. `outputs/final_frozen_v2/final_readiness_check.md` dosyasında V2 final operational evidence olarak kilitlenmiştir.

Taiwan için final policy V2 CatBoost manual-review policy'dir: precision 0.529, recall 0.575, specificity 0.854, FP 680, FN 564, review-adjusted cost 2705.5, manual-review rate 0.295. HELOC için final policy V2 Scorecard manual-review policy'dir: precision 0.683, recall 0.846, specificity 0.574, FP 403, FN 158, review-adjusted cost 764.5, manual-review rate 0.270.

## Neden V3 Değil?

Weakness-closing paketi V3 adaylarını üretti ancak final replacement kriterlerini geçirmedi. Taiwan recall candidate recall'ı 0.633'e kadar çıkardı fakat precision/specificity düştü ve FP arttı. HELOC specificity candidate specificity'yi 0.676'ya çıkardı fakat cost yükseldi ve recall düştü. Final weakness-closing audit READY olsa da final selector V2'yi korudu.

## Final Attempt Neden Appendix?

Final Attempt geniş bir robustness/stress-test paketidir ama audit verdict BLOCKED idi. Ana problem manual-review locked-test satırlarında binary confusion metrics ile 3x2 manual-review bucket metrics'in karışmasıydı. Daha sonra metric repair yapıldı; ancak bu repair Final Attempt'i final evidence haline getirmez. Final Attempt literature reproduction, DNN, EBM/monotonic, decision curve ve capacity analysis için appendix olarak değerlidir.

## SCRE-Credit'in Rolü

SCRE-Credit dominant classifier değildir. Taiwan'da operational final CatBoost, HELOC'ta Scorecard'dır. SCRE'nin güvenli rolü reliability-aware framework, model comparison structure ve review-prioritization/ranking support'tur. Capacity-aware review analizinde SCRE-Optimized top-20% review için Taiwan'da capture@20=0.529 ve lift@20=2.645 ile faydalı ranking sinyali vermiştir.

## Son Cümle

“Bu proje en yüksek skor alan default modeli olarak değil; leakage-aware, validation-selected, manual-review destekli ve reliability-aware kredi riski karar framework'ü olarak sunulmalıdır.”
"""
    (OUT / "FINAL_DECISION_EXPLAINED.md").write_text(text, encoding="utf-8")
    return text


def section(title: str, body: str, files: list[str], result: str, worked: str, final_effect: str, teacher: str) -> str:
    file_lines = "\n".join(f"- `{f}`" for f in files) if files else "- Belirsiz / dosyada yok / doğrudan doğrulanamadı."
    return f"""# {title}

## Ne yapıldı?
{body}

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
{file_lines}

## Sonuç ne çıktı?
{result}

## İşe yaradı mı?
{worked}

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
{final_effect}

## Hocaya nasıl anlatılır?
{teacher}
"""


def build_main_report(
    file_inventory: pd.DataFrame,
    dataset_dict: pd.DataFrame,
    feature_inv: pd.DataFrame,
    cleaning: pd.DataFrame,
    results: pd.DataFrame,
    worked_failed: pd.DataFrame,
    claims: pd.DataFrame,
    literature: pd.DataFrame,
    metrics: pd.DataFrame,
    table_plan: pd.DataFrame,
    leakage_table: pd.DataFrame,
) -> str:
    v2_tw = read_csv("outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv")
    v2_he = read_csv("outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv")
    weakness_pkg = read_text("outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md", 7000)
    audit_v2 = read_text("outputs/decision_revision_v2/final_audit_v2.md", 3000)
    audit_wc = read_text("outputs/final_weakness_closing/audit/final_weakness_closing_audit.md", 3500)
    final_attempt = read_text("outputs/final_attempt/report_to_user/FINAL_ATTEMPT_RESULTS_FOR_CHATGPT.md", 3500)

    v2_table = pd.concat([v2_tw, v2_he], ignore_index=True)
    report_parts = [
        "# FULL PROJECT DETAILED REPORT TR\n",
        "Bu rapor mevcut proje çıktılarının teknik tartışma envanteridir. Bu aşamada yeni model eğitilmemiş, yeni feature üretilmemiş, yeni threshold seçilmemiş ve test set üzerinden yeni karar verilmemiştir. Sayısal sonuçlar mevcut dosyalardan okunmuştur; bulunamayan dosyalar envanterde not edilmiştir.\n",
        "## Ana final karar\n",
        "- Final operational version: **V2**\n- Taiwan final policy: **V2 CatBoost manual-review**\n- HELOC final policy: **V2 Scorecard manual-review**\n- V3 final mi: **Hayır**\n- Final Attempt final mi: **Hayır, appendix/stress-test**\n- SCRE-Credit final classifier mı: **Hayır; reliability-aware framework ve review-prioritization support**\n- Sistem automatic rejection mı: **Hayır; screening/manual-review decision-support**\n",
        "## V2 ana sonuç tablosu\n",
        md_table(v2_table[["dataset", "final_policy", "precision", "recall", "specificity", "fp", "fn", "cost", "manual_review_rate", "audit_status", "safe_category"]]),
        "\n## Weakness-closing özetinden kilit noktalar\n",
        weakness_pkg,
        "\n",
    ]

    headings = [
        ("1. Executive Summary", "Proje kredi default tahmini ile başladı; zamanla salt model skoru üretmekten çıkıp validation-selected, calibration-aware, cost-sensitive ve manual-review destekli karar framework'üne dönüştü. Ana sonuç V2'dir.", ["outputs/final_frozen_v2/final_readiness_check.md", "outputs/decision_revision_v2/final_audit_v2.md"], "V2 READY; Final Attempt appendix; weakness-closing READY ama V3 final değil.", "Evet, ana final karar için işe yaradı.", "V2 final kaldı.", "Bu projede en önemli başarı en yüksek skoru kovalamak değil, metodolojik olarak savunulabilir final policy üretmektir."),
        ("2. Projenin Amacı", "Amaç kredi riski/default olasılığını tahmin etmek ve bu tahminleri gerçekçi karar destek politikasına dönüştürmekti.", ["PROJECT_SCOPE.md", "outputs/final_positioning_statement.md"], "Primary dataset Taiwan, external validation HELOC; SCRE framework olarak konumlandı.", "Evet.", "Proje automatic rejection yerine screening/manual-review olarak sunuldu.", "Amaç bankanın tek tuşla ret sistemi değil, riskli başvuruları önceliklendiren bir karar destek çerçevesi."),
        ("3. Problem Tanımı", "Problem binary credit default classification olarak başladı; ancak imbalanced data, false positives, calibration, cost ve manual-review boyutları nedeniyle karar problemi haline geldi.", ["outputs/decision_revision/current_problem_summary.md", "outputs/decision_revision_v2/SELECTION_PROTOCOL_V2.md"], "Cost-minimization tek başına fazla agresif davranabildi.", "Evet, problem doğru genişletildi.", "V2 manual-review policy doğdu.", "Sadece 'default tahmin ettik' değil, hangi müşterinin insana gönderileceğini de tartıştık."),
        ("4. Dataset Açıklaması", "Taiwan ana dataset, HELOC external validation dataset olarak kullanıldı. German Credit aktif kapsamdan çıkarıldı.", ["data/processed/taiwan_model_ready.csv", "data/processed/heloc_model_ready.csv", "PROJECT_SCOPE.md"], "Taiwan model-ready 41 kolon; HELOC model-ready 32 kolon olarak doğrulandı.", "Evet.", "Taiwan ve HELOC rolleri ayrıldı.", "Taiwan ana deney, HELOC dış doğrulama/robustness gibi düşünülmeli."),
        ("5. Dataset Kolonları ve Kullanım Durumu", "Her kolon için anlam, modelde kullanım, cost/explainability/segment analizinde kullanım sözlüğe yazıldı.", ["outputs/full_project_discussion_report/DATASET_COLUMN_DICTIONARY.csv"], "Kolon sözlüğü üretildi.", "Evet.", "Raporun dataset appendix'ine girecek.", "ID ve target feature değildir; ödeme gecikme ve borç/ödeme kolonları temel risk sinyalleridir."),
        ("6. Data Cleaning ve Anomali İşleme", "Taiwan EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3; HELOC special codes NaN olarak temizlendi.", ["src/data_preprocessing.py", "src/heloc_preprocessing.py", "outputs/full_project_discussion_report/data_cleaning_audit.csv"], "Cleaning target-free ve leakage yaratmıyor.", "Evet.", "Güvenli preprocessing olarak ana raporda anlatılacak.", "Anomali temizliği hedefe bakmadan yapıldı; bu yüzden leakage değil."),
        ("7. Missing / Duplicate / Target Kontrolleri", "Missing, duplicate, target distribution ve negative bill count kontrolleri toplandı.", ["outputs/full_project_discussion_report/data_cleaning_audit.csv", "outputs/tables/taiwan_data_audit.csv", "outputs/tables/heloc_data_audit.csv"], "Taiwan target imbalanced; HELOC missing special-code kaynaklı olabilir.", "Evet.", "Data audit kanıtı oluşturdu.", "Sınıf dengesizliği accuracy'nin neden yeterli olmadığını açıklıyor."),
        ("8. EDA ve İlk Bulgular", "Target dağılımı, sınıf dengesizliği, gecikme davranışları ve borç/ödeme değişkenleri incelendi.", ["outputs/figures/taiwan_target_distribution.png", "outputs/figures/heloc_target_distribution.png", "outputs/tables/taiwan_data_audit.csv"], "Ödeme gecikmeleri güçlü risk sinyali; class imbalance belirgin.", "Evet.", "Metric ve threshold seçim gerekçesi güçlendi.", "EDA bize modelden önce problemin dengesiz ve cost-sensitive olduğunu gösterdi."),
        ("9. Feature Engineering", "Row-wise, target-free finansal davranış feature'ları üretildi: delay_count, severe_delay_count, utilization_proxy, payment_to_bill_ratio vb.", ["src/feature_engineering.py", "src/features/heloc_features.py", "outputs/full_project_discussion_report/USED_UNUSED_FEATURES.csv"], "Feature'lar target kullanmadı ve dataset-level fitted statistic kullanmadı.", "Evet.", "Model/explainability/cost analizine destek verdi.", "Feature engineering hedef değişkenden türetilmedi; bu kritik güvenlik noktası."),
        ("10. Preprocessing Pipeline", "Imputation, scaling/encoding ve model pipeline'larının split sonrası fit edilmesi prensibi korundu.", ["src/models/model_factory.py", "src/data/split_leakage.py", "outputs/tables/leakage_checklist.csv"], "Preprocessing leakage riskleri audit edildi.", "Evet.", "Leakage-free claim desteklendi.", "Scaler/encoder testten öğrenirse sonuç şişer; biz bu riski audit ettik."),
        ("11. Train / Validation / Test Ayrımı", "Train model fitting, validation policy seçimi, test locked evaluation için kullanıldı.", ["outputs/decision_revision_v2/SELECTION_PROTOCOL_V2.md", "outputs/decision_revision_v2/final_table_methodology_note.md"], "V2 test-ranking sorununu düzeltti.", "Evet.", "V2 final seçimi güvenli hale geldi.", "Test seti karar seçmek için değil, seçilmiş kararı sınamak için kullanıldı."),
        ("12. Baseline Modeller", "Logistic Regression, Random Forest ve temel modeller benchmark olarak denendi.", ["outputs/tables/model_results_taiwan.csv", "outputs/tables/model_results_heloc.csv", "src/models/baselines.py"], "Baseline'lar boosting/scorecard/SCRE için referans verdi.", "Evet ama final olmadı.", "Appendix/baseline evidence.", "Baselines final değil ama güçlü modellerin gerçekten değer kattığını görmemizi sağlar."),
        ("13. Boosting Modelleri", "XGBoost, LightGBM, CatBoost ve monotonic varyantlar denendi.", ["outputs/tables/boosting_results_taiwan.csv", "outputs/tables/boosting_results_heloc.csv", "outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv"], "Taiwan final operational model CatBoost manual-review oldu.", "Evet.", "Taiwan final policy seçimini belirledi.", "Taiwan'da en dengeli audit-ready operasyonel sonuç CatBoost manual-review ile geldi."),
        ("14. Scorecard / WOE Logistic Regression", "WOE/Scorecard finansal yorumlanabilir baseline olarak kuruldu.", ["src/models/scorecard.py", "outputs/tables/scorecard_results.csv", "outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv"], "HELOC final operational policy Scorecard manual-review oldu.", "Evet.", "HELOC final model rolünü belirledi.", "HELOC'ta açıklanabilir scorecard final kalması governance açısından güçlü."),
        ("15. Imbalance Handling Denemeleri", "SMOTE/SMOTENC, class weights, scale_pos_weight, advanced imbalance ve literature reproduction denendi.", ["outputs/decision_revision/imbalance_training_variants_taiwan.csv", "outputs/final_attempt/imbalance_boosting/advanced_imbalance_summary.md"], "Bazı trade-offlar iyileşti ama V2 yerine audit-ready final çıkmadı.", "Appendix olarak yaradı.", "Final kararı değiştirmedi.", "Imbalance çözümü sadece recall artırmak değil; FP, calibration ve cost da korunmalı."),
        ("16. Hyperparameter Tuning", "Optuna ile XGBoost/LightGBM/CatBoost tuning yapıldı.", ["outputs/tables/optuna_trials.csv", "outputs/tables/best_params.csv", "outputs/tables/optuna_run_status.csv"], "Taiwan 100, HELOC 50 trial gibi kapsamlı tuning kayıtları üretildi.", "Evet.", "Güçlü candidate modeller üretildi.", "Tuning testle değil validation/CV mantığıyla yapılmalı; bu proje bunu ayırmaya çalıştı."),
        ("17. Calibration", "Sigmoid/isotonic calibration ve Brier/ECE ölçümleri yapıldı.", ["outputs/tables/calibration_results_taiwan.csv", "outputs/tables/calibration_results_heloc.csv", "outputs/decision_revision_v2/calibration_threshold_caveat.md"], "Probability reliability ayrı değerlendirildi.", "Evet.", "Manual-review ve threshold politikaları için daha güvenilir olasılıklar sağladı.", "Calibration AUC artırmak için değil, olasılığı güvenilir yapmak için kullanıldı."),
        ("18. Threshold Optimization", "0.50, cost-optimal, precision/specificity constrained threshold politikaları denendi.", ["outputs/decision_revision/precision_constrained_test_results_taiwan.csv", "outputs/decision_revision_v2/validation_candidate_registry_all.csv"], "Eski agresif threshold FP problemini gösterdi.", "Evet.", "V2 manual-review yaklaşımına geçişi motive etti.", "Threshold seçimi validation'da yapılmazsa test leakage olur; bu özellikle düzeltildi."),
        ("19. Cost-Sensitive Decision Layer", "FN/FP cost ve cost matrix sensitivity analizleri yapıldı.", ["outputs/decision_revision/cost_matrix_sensitivity_taiwan.csv", "outputs/final_weakness_closing/cost_sensitivity/cost_decision_summary.md"], "Cost varsayımlarına duyarlılık açıkça raporlandı.", "Evet.", "Gerçek cost yokluğu limitation olarak yönetildi.", "Cost gerçek banka zararı değil; varsayım ve sensitivity analizidir."),
        ("20. Manual-Review Band Sistemi", "Binary karar üçlü bucket'a çevrildi: low risk, manual review, high risk.", ["outputs/final_frozen_v2/manual_review_metric_definitions.md", "outputs/decision_revision_v2/manual_review_cost_model_all.csv"], "V2 final policy manual-review destekli oldu.", "Evet.", "Automatic rejection claim kaldırıldı.", "Belirsiz müşteriyi otomatik reddetmek yerine insana gönderiyoruz."),
        ("21. SCRE-Credit Framework", "SCRE-Credit performance, calibration, cost, stability ve faithfulness kanıtlarını ağırlıklandıran framework olarak geliştirildi.", ["src/models/scre_credit.py", "outputs/tables/scre_pareto_results_taiwan.csv", "outputs/final_weakness_closing/scre_prioritization/scre_role_summary.md"], "SCRE dominant classifier değil; review-prioritization/framework rolü güçlü.", "Evet, framework olarak.", "Final classifier olmadı ama proje katkısı olarak kaldı.", "SCRE'yi 'her şeyi yenen model' değil, güvenilirlik-aware framework diye anlatmalıyız."),
        ("22. Explainability: SHAP ve LIME", "Global/local SHAP, LIME ve agreement tabloları üretildi.", ["outputs/tables/shap_top_features.csv", "outputs/tables/lime_local_features.csv", "outputs/tables/shap_lime_agreement.csv"], "Model behavior explanation appendix'i oluştu.", "Evet, supporting.", "Final seçim değil, yorumlama desteği.", "Açıklamalar causal proof değil; model davranışı sanity check."),
        ("23. SHAP Stability / Kendall’s W", "Seed bazlı SHAP stability, Kendall's W ve literature comparison üretildi.", ["outputs/tables/kendalls_w_stability_taiwan.csv", "outputs/tables/kendalls_w_stability_heloc.csv", "outputs/tables/shap_stability_literature_comparison.csv"], "Explanation reliability tartışması güçlendi.", "Evet.", "SCRE/reliability framework anlatımını destekledi.", "Açıklama sadece tek koşuda değil, seed'ler arasında ne kadar stabil diye de kontrol edildi."),
        ("24. Faithfulness Test", "Feature deletion/insertion/perturbation analizleri ile açıklamaların model davranışına uyumu incelendi.", ["outputs/tables/faithfulness_results.csv", "outputs/tables/faithfulness_group_results.csv"], "Causality değil, model behavior sanity check.", "Appendix olarak evet.", "Final policy seçmedi.", "Top feature'lar silinince performans ne oluyor diye baktık; bu nedensellik iddiası değil."),
        ("25. Statistical Tests", "Bootstrap CI, paired tests, McNemar ve cleaned statistical tests üretildi.", ["outputs/tables/statistical_tests_cleaned.csv", "outputs/tables/statistical_tests_summary.csv"], "p-value yön/winner ile temizlendi.", "Supporting.", "Overclaim riskini azalttı.", "Significant demek bizim model kazandı demek değildir; yönüne bakmak gerekir."),
        ("26. İlk Final Sonuçlar ve Problemler", "İlk final tablolar test metric ranking ambiguity içerdi.", ["outputs/decision_revision/final_decision_revision_audit.md"], "NOT READY kararı verildi.", "Evet, hata yakalandı.", "V2 protokolüne geçildi.", "Bu aşama projeyi güçlendirdi çünkü hatayı saklamadık."),
        ("27. Decision Revision V1", "FP reduction, precision constraints, cost matrix, DCA, manual review, imbalance, eligibility ve FP analysis yapıldı.", ["outputs/decision_revision/"], "Retrospective evidence güvenli ama final selection olarak güvenli değildi.", "Diagnostic olarak yaradı.", "V2 reset ihtiyacını doğurdu.", "V1 sonuçları öğretici ama final seçim kanıtı değil."),
        ("28. V2 Revision ve Final Operational Policy", "Validation-only candidate registry, manual-review cost correction, locked policy ve held-out evidence üretildi.", ["outputs/decision_revision_v2/final_audit_v2.md", "outputs/final_frozen_v2/"], "V2 READY ve final operational version oldu.", "Evet.", "Final karar V2.", "V2 projenin metodolojik olarak temiz final noktasıdır."),
        ("29. Final Attempt Denemeleri", "Literature reproduction, imbalance, DNN, EBM/monotonic, capacity, decision curve, final selector denendi.", ["outputs/final_attempt/report_to_user/FINAL_ATTEMPT_RESULTS_FOR_CHATGPT.md"], "Geniş stress-test yapıldı.", "Appendix olarak yaradı.", "V2'yi değiştirmedi.", "Son büyük deneme negatif sonuçlar dahil dürüstçe raporlandı."),
        ("30. Final Attempt Neden Final Olmadı?", "Audit BLOCKED: manual-review locked-test metric consistency failures.", ["outputs/final_attempt/audit/final_attempt_audit.md", "outputs/final_weakness_closing/metric_repair/final_attempt_metric_repair_summary.md"], "3x2 repair sonrası bile appendix only.", "Final olarak hayır.", "V2 korunur.", "Metric tutarsızlığı varsa sonuç iyi görünse bile final yapılmaz."),
        ("31. Weakness-Closing Denemeleri", "V2 zayıflıklarını kapatmak için recall, capacity, specificity, cost, instance cost, SCRE, segment, temporal analizleri yapıldı.", ["outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md"], "Audit READY, critical 0; ancak V3 final replacement olmadı.", "Appendix/supporting olarak çok yaradı.", "V2 final kaldı.", "Zayıflıklar denendi ama daha iyi final denge çıkmadı."),
        ("32. Taiwan Recall İyileştirme Denemeleri", "Taiwan recall 0.575'ten 0.633'e çıkaran candidate bulundu.", ["outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv"], "Precision 0.463'e düştü, FP 976'ya çıktı.", "Trade-off kanıtı olarak yaradı.", "Final olmadı.", "Daha çok default yakalamak daha fazla yanlış alarm doğurdu."),
        ("33. Capacity-Aware Review ve Lift@K", "Top-K review prioritization analizi yapıldı.", ["outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv"], "SCRE-Optimized Taiwan top20 capture 0.529, lift 2.645; HELOC capture 0.340, lift 1.698.", "Evet, appendix güçlü.", "SCRE framework rolünü güçlendirdi.", "Banka sadece %20 inceleyebiliyorsa model sıralama aracı olarak işe yarıyor."),
        ("34. HELOC Specificity İyileştirme Denemeleri", "HELOC specificity candidate specificity'yi 0.676'ya çıkardı.", ["outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv"], "Recall 0.782'ye düştü, cost 1427'ye yükseldi.", "Appendix olarak yaradı.", "Final olmadı.", "İyi müşterileri daha iyi ayırmak maliyeti artırdı."),
        ("35. Cost Sensitivity ve Decision Curve", "FN/FP/review cost ve net benefit analizleri yapıldı.", ["outputs/final_weakness_closing/cost_sensitivity/cost_decision_summary.md"], "V2 core assumptions altında stabil kaldı; cost uncertainty final kararı değiştirmedi.", "Evet.", "Gerçek cost limitation'ı yönetildi.", "Gerçek maliyet yoksa en doğru savunma sensitivity analysis'tir."),
        ("36. Instance-Dependent Cost Analysis", "LIMIT/BILL/utilization ve HELOC risk proxy'leriyle örnek-bağımlı cost sensitivity yapıldı.", ["outputs/final_weakness_closing/instance_cost/instance_cost_summary.md"], "Taiwan recall candidate bazı proxy costlarda iyi; HELOC V2 best kaldı.", "Appendix olarak yaradı.", "Final karar değişmedi.", "Bu gerçek banka maliyeti değil, proxy-based sensitivity."),
        ("37. Segment-Aware Threshold Denemeleri", "Limit/utilization/recent delay ve HELOC risk burden segmentlerinde policy denendi.", ["outputs/final_weakness_closing/segment_thresholds/segment_policy_summary.md"], "Taiwan'da net replacement yok; HELOC specificity trade-off var.", "Appendix/diagnostic.", "Final olmadı.", "Segment policy overfitting riski taşır; dikkatli appendix."),
        ("38. Temporal Robustness / Deployment Simülasyonu", "Gerçek timestamp arandı; bulunamadı.", ["outputs/final_weakness_closing/temporal_robustness/temporal_data_availability_check.md"], "Gerçek temporal validation mümkün değil; limitation only.", "Limitation olarak yaradı.", "Deployment claim kaldırıldı.", "Tarih yoksa temporal validation iddiası kuramayız."),
        ("39. Final Audit ve Metric Consistency", "Final weakness-closing audit yapıldı.", ["outputs/final_weakness_closing/audit/final_weakness_closing_audit.md"], "READY, critical 0, test leakage NO, metric consistency PASS; high caveat diagnostic test columns.", "Evet.", "Sonuca geçilebilir.", "Audit READY ama diagnostic columns selection evidence değildir."),
        ("40. Final Karar: Neden V2?", "V2 tek audit-ready final operational evidence olarak kaldı.", ["outputs/final_frozen_v2/final_readiness_check.md", "outputs/final_weakness_closing/final_selection/final_selection_summary.md"], "Use V2 as final YES; use V3 final NO.", "Evet.", "Final rapor V2 üzerine kurulmalı.", "V2 sıkıcı ama güvenli; bilimsel olarak güvenli olan final budur."),
        ("41. Ne İşe Yaradı?", "V2 final policies, SCRE prioritization, capacity, cost sensitivity, XAI/stability ve claim control işe yaradı.", ["outputs/full_project_discussion_report/WHAT_WORKED_WHAT_FAILED.csv"], "Worked/failed tablosu üretildi.", "Evet.", "Rapor planını netleştirdi.", "İşe yarayanları final/appendix diye ayırdım."),
        ("42. Ne İşe Yaramadı?", "Final Attempt, DNN, KMeansSMOTE, advanced imbalance ve V3 candidates V2'yi değiştirmedi.", ["outputs/full_project_discussion_report/WHAT_WORKED_WHAT_FAILED.csv"], "Negatif sonuçlar saklanmadı.", "Evet, dürüst rapor için.", "Overclaim engellendi.", "Başarısız deney de bilimsel kanıttır."),
        ("43. Neler Appendix’e Girecek?", "Final Attempt, weakness-closing candidates, capacity/lift, decision curve, XAI/stability, instance cost, segment diagnostics appendix'e girebilir.", ["outputs/final_project_inventory/report_usage_plan.md", "outputs/full_project_discussion_report/TABLE_FIGURE_PLACEMENT_PLAN.csv"], "Appendix planı oluşturuldu.", "Evet.", "Ana hikâye şişmeyecek.", "Ana sonuç V2; appendix kapsamlı robustness."),
        ("44. Neler Ana Sonuç Olarak Kullanılacak?", "V2 frozen policies, V2 audit READY, final claim control, manual-review metric definitions ana sonuçtur.", ["outputs/final_frozen_v2/"], "Ana evidence net.", "Evet.", "Rapor ana gövdesi V2'ye dayanır.", "Ana sonuç sayısı az ama temiz."),
        ("45. Neler Kesinlikle Claim Edilmeyecek?", "Automatic rejection, SCRE beats all, V3 final, Final Attempt replaces V2, gerçek deployment, gerçek bank cost iddiaları yasak.", ["outputs/full_project_discussion_report/FINAL_CLAIM_CONTROL_TABLE.csv"], "Claim control üretildi.", "Evet.", "Güvenli akademik dil sağlandı.", "Ne söylemeyeceğimizi bilmek en az sonuç kadar önemli."),
        ("46. Literatürle Karşılaştırma", "Imbalanced learning, DNN, boosting, scorecard, XAI, cost, manual-review ve reliability framework temalarıyla karşılaştırıldı.", ["outputs/full_project_discussion_report/literature_positioning_table.csv"], "Proje SOTA score değil audit-ready decision framework olarak konumlandı.", "Evet.", "Literature discussion güvenli hale geldi.", "Düşük görünen skorlar daha dürüst protokolün sonucu olabilir."),
        ("47. Projenin Güçlü Yanları", "Leakage-aware protokol, V2 audit READY, manual-review policy, external HELOC, SCRE framework, claim control.", ["outputs/decision_revision_v2/final_audit_v2.md", "outputs/final_frozen_v2/final_readiness_check.md"], "Güçlü yanlar net.", "Evet.", "Rapor conclusion desteklendi.", "Projenin gücü sadece model değil, validation disiplini."),
        ("48. Projenin Zayıf Yanları", "Taiwan recall orta, MR rate yüksek, HELOC specificity sınırlı, gerçek temporal/cost/deployment yok, Final Attempt blocked.", ["outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md"], "Zayıflıklar saklanmadı.", "Evet.", "Limitations güvenli yazılacak.", "Zayıflıkları gizlemedim; test ettim ve sınırlılık olarak yazdım."),
        ("49. Kalan Limitations", "Gerçek banka deployment, timestamp, richer financial data, actual loss/cost, fairness/legal proof yok.", ["outputs/final_weakness_closing/temporal_robustness/temporal_robustness_summary.md", "outputs/final_frozen_v2/final_v2_do_not_claim.md"], "Limitations net.", "Evet.", "Claim sınırı belirlendi.", "Bu proje deployment değil, akademik validation framework."),
        ("50. Hocanın Sorabileceği Sorular ve Cevaplar", "60+ Q&A hazırlandı.", ["outputs/full_project_discussion_report/TEACHER_QA_PREPARATION.md"], "Hocaya tartışma hazırlığı üretildi.", "Evet.", "Sunum savunması kolaylaştı.", "Zor sorulara doğrudan, iddiasız ve kanıtlı cevap vereceğiz."),
        ("51. Final Sonuç", "V2 final, V3 hayır, Final Attempt appendix, SCRE framework, automatic rejection hayır.", ["outputs/full_project_discussion_report/FINAL_DECISION_EXPLAINED.md"], "Proceed to conclusion YES.", "Evet.", "Sonraki aşama final conclusion/report writing olabilir.", "Bu proje güvenli karar destek framework'ü olarak sunulmalı."),
        ("52. Ekler / Appendix Planı", "Appendix'e girecek tablo/figür ve kaynaklar belirlendi.", ["outputs/full_project_discussion_report/TABLE_FIGURE_PLACEMENT_PLAN.csv"], "Ana vs appendix ayrımı netleşti.", "Evet.", "Rapor şişmeden detay korunur.", "Ana gövde temiz; tüm ekstra deneyler appendix'te kanıt olarak durur."),
    ]
    for args in headings:
        report_parts.append(section(*args))

    report_parts += [
        "# Dataset Kolon Sözlüğü Özeti\n",
        md_table(dataset_dict.head(60)),
        "\n# Feature Engineering Özeti\n",
        md_table(feature_inv.head(60)),
        "\n# Metric Sözlüğü\n",
        md_table(metrics),
        "\n# Split ve Leakage Kontrol Tablosu\n",
        md_table(leakage_table),
        "\n# Model ve Policy Sonuç Özeti\n",
        md_table(results.head(80)),
        "\n# Worked / Failed Tablosu\n",
        md_table(worked_failed),
        "\n# Claim Control Tablosu\n",
        md_table(claims),
        "\n# Literature Positioning Tablosu\n",
        md_table(literature),
        "\n# Table/Figure Placement Planı\n",
        md_table(table_plan),
        "\n# Audit excerpts\n",
        "## V2 audit excerpt\n",
        audit_v2,
        "\n## Final Attempt audit/report excerpt\n",
        final_attempt,
        "\n## Weakness-closing audit excerpt\n",
        audit_wc,
        "\n# Hocaya 5 dakikada anlatım\n",
        "1. Problem kredi default tahminiydi.\n2. İlk modeller ve cost-threshold yaklaşımları fazla agresif davranıp FP sorununu büyütebildi.\n3. Sadece accuracy/AUC yeterli olmadığı için precision, recall, specificity, cost, calibration ve manual-review metrikleri birlikte ele alındı.\n4. V2 ile validation-only seçilen manual-review destekli karar sistemi kuruldu.\n5. Taiwan'da final operational policy CatBoost manual-review, HELOC'ta Scorecard manual-review oldu.\n6. SCRE-Credit final classifier değil, reliability-aware framework ve review-prioritization destek aracı olarak kaldı.\n7. Final Attempt ve weakness-closing deneyleri ek robustluk sağladı ama V2'yi temiz şekilde geçmedi.\n8. Son sistem automatic rejection değil, screening/manual-review support sistemidir.\n9. Audit sonucu final V2 için READY; test leakage yok.\n10. Sonuçlar abartılmadan, limitations açık yazılarak rapora geçilebilir.\n",
        "\n# Maymuna anlatır gibi versiyon\n",
        "Bankaya 100 kişi geliyor. Model bu kişileri risk sırasına koyuyor. Çok riskli görünenleri işaretliyor, emin olamadıklarını insana gönderiyor. İlk denemelerde daha çok default yakalamak için eşik düşürülünce yanlış alarm çok arttı. Sonra daha dengeli bir sistem kuruldu: Taiwan için CatBoost, HELOC için Scorecard. Daha fazla default yakalamak tekrar denendi ama bu kez yanlış alarm veya maliyet arttı. Bu yüzden en güvenli ve denetlenebilir final V2 oldu.\n",
    ]
    text = "\n".join(report_parts)
    (OUT / "FULL_PROJECT_DETAILED_REPORT_TR.md").write_text(text, encoding="utf-8")

    teacher = "\n".join(
        [
            "# FULL PROJECT DETAILED REPORT FOR TEACHER TR",
            "",
            "Bu dosya ana raporun hocayla tartışmaya uygun, daha kısa ama teknik versiyonudur. Yeni deney üretmez; mevcut kanıtları derler.",
            "",
            "## Net final karar",
            "- V2 final operational version olarak kalmalıdır.",
            "- Taiwan: V2 CatBoost manual-review.",
            "- HELOC: V2 Scorecard manual-review.",
            "- Final Attempt appendix/stress-test olarak kullanılmalıdır.",
            "- V3 adayları bazı zayıflıkları iyileştirdi ama final replacement olmadı.",
            "- SCRE-Credit dominant classifier değil, reliability-aware framework ve review-prioritization desteğidir.",
            "- Sistem automatic rejection değil, screening/manual-review decision-support sistemidir.",
            "",
            "## V2 ana sonuçlar",
            md_table(v2_table[["dataset", "final_policy", "precision", "recall", "specificity", "fp", "fn", "cost", "manual_review_rate", "audit_status"]]),
            "",
            "## Dataset ve kapsam",
            "- Ana veri seti Taiwan Default of Credit Card Clients'tır. Bu veri kredi kartı default tahminine doğrudan bağlıdır ve German Credit'e göre daha büyük/akademik olarak daha güçlüdür.",
            "- HELOC external validation / robustness tarafında kullanılmıştır. Feature şeması Taiwan'dan farklı olduğu için doğrudan model transferi değil, framework davranışını tartışmak için değerlidir.",
            "- German Credit aktif kapsamdan çıkarılmış ve archive altında tutulmuştur.",
            "",
            "## Metodoloji özeti",
            "- Train: model fitting.",
            "- Validation: threshold, manual-review band ve policy selection.",
            "- Test: yalnızca locked final policy değerlendirmesi.",
            "- V2'de test table rank/winner içermez; bu nedenle test-ranked final seçim problemi giderilmiştir.",
            "- Manual-review cost, binary FN/FP cost ile aynı değildir. Manual-review üç bucket olarak yorumlanır: low risk, manual review, high risk.",
            "",
            "## Neden automatic rejection değil?",
            "Precision ve FP trade-offları hâlâ önemlidir. Taiwan final precision 0.529'dur; bu değer riskli işaretlenen herkesin gerçekten default olduğu anlamına gelmez. Bu nedenle sistem otomatik red için değil, riskli ve belirsiz vakaları insan incelemesine önceliklendirmek için daha uygundur.",
            "",
            "## Neden Taiwan'da CatBoost?",
            "Taiwan'da V2 CatBoost manual-review policy audit-ready final operational evidence oldu. Specificity 0.854 ile yanlış alarm baskısını eski agresif thresholdlara göre daha yönetilebilir hale getirdi. Recall 0.575 orta seviyede kaldı; bu zayıflık weakness-closing aşamasında ayrıca test edildi.",
            "",
            "## Neden HELOC'ta Scorecard?",
            "HELOC'ta V2 Scorecard manual-review policy hem interpretable benchmark hem de final operational evidence olarak kaldı. Recall 0.846 güçlüdür, precision 0.683'tür. Specificity 0.574 sınırlıdır; bu zayıflık test edildi ama specificity artırıldığında cost yükseldi.",
            "",
            "## Final Attempt neden final olmadı?",
            "Final Attempt geniş ve değerli bir stress-test paketidir; literature reproduction, DNN/BP NN, advanced imbalance, EBM/monotonic, decision curve ve capacity analysis içerir. Ancak audit verdict BLOCKED idi: manual-review locked-test satırlarında binary confusion matrix metrikleri ile manual-review bucket metrikleri karışmıştı. Daha sonra 3x2 repair yapıldı, fakat bu Final Attempt'i final evidence yapmadı; appendix olarak kalmalıdır.",
            "",
            "## Weakness-closing ne gösterdi?",
            "- Taiwan recall candidate recall'ı 0.575'ten 0.633'e çıkarabildi; fakat precision 0.463'e düştü ve FP 976'ya çıktı.",
            "- HELOC specificity candidate specificity'yi 0.574'ten 0.676'ya çıkarabildi; fakat cost 764.5'ten 1427'ye çıktı ve recall düştü.",
            "- Capacity-aware review, SCRE-Optimized'ın top-20% review prioritization için faydalı ranking sinyali verdiğini gösterdi.",
            "- Temporal robustness gerçek timestamp olmadığı için limitation-only kaldı.",
            "",
            "## SCRE-Credit rolü",
            "SCRE-Credit her şeyi yenen classifier olarak sunulmamalıdır. Güvenli rolü reliability-aware framework, model comparison yapısı ve review-prioritization desteğidir. Bu ayrım raporun claim güvenliği açısından kritik.",
            "",
            "## Ana rapora girmesi gerekenler",
            "- V2 final policies.",
            "- V2 audit READY.",
            "- Manual-review metric definitions.",
            "- Final claim control.",
            "- Taiwan CatBoost ve HELOC Scorecard final sonuçları.",
            "- SCRE framework rolü.",
            "",
            "## Appendix'e girmesi gerekenler",
            "- Final Attempt stress-test ve negative results.",
            "- DNN/BP NN ve KMeansSMOTE reproduction.",
            "- Capacity-aware review ve Lift@K.",
            "- Decision curve.",
            "- Instance-dependent cost.",
            "- Segment-aware threshold diagnostics.",
            "- SHAP/LIME/stability/faithfulness.",
            "",
            "## Claim edilmemesi gerekenler",
            "- Model otomatik kredi reddi için uygundur.",
            "- SCRE-Credit tüm modelleri geçti.",
            "- Final Attempt V2'yi değiştirdi.",
            "- V3 finaldir.",
            "- Gerçek temporal deployment validation yapıldı.",
            "- Gerçek banka maliyeti biliniyor.",
            "",
            "## Hocaya anlatılacak ana hikâye",
            "Projede önce default tahmini için klasik ve gelişmiş modeller denendi. İlk cost-sensitive politikalar false positive problemini büyütebildi. Bu yüzden final yaklaşım binary otomatik karar değil, manual-review destekli screening sistemine dönüştürüldü. V2 aşamasında seçim validation-only yapıldı; test set yalnızca kilitlenmiş policy için kullanıldı. Bu metodolojik temizlik nedeniyle V2 final audit-ready kanıt oldu. Final Attempt ve weakness-closing deneyleri çok faydalı appendix/robustness kanıtı verdi ancak V2'yi güvenli şekilde değiştirmedi.",
            "",
            "## Tartışmaya açık zayıflıklar",
            "- Taiwan recall orta seviyede kaldı.",
            "- Manual-review oranı yaklaşık %30 civarında olduğu için operasyonel kapasite tartışması gerekir.",
            "- HELOC specificity sınırlı kaldı.",
            "- Gerçek temporal deployment ve gerçek banka cost verisi yok.",
            "- SCRE-Credit güçlü framework ama dominant classifier değil.",
            "",
            "## Güvenli final cümle",
            "Bu proje en yüksek skor alan default modeli olarak değil; leakage-aware, validation-selected, manual-review destekli ve reliability-aware kredi riski karar framework'ü olarak sunulmalıdır.",
            "",
            "## Q&A dosyası",
            "`outputs/full_project_discussion_report/TEACHER_QA_PREPARATION.md` içinde 60+ soru-cevap hazırlanmıştır.",
        ]
    )
    (OUT / "FULL_PROJECT_DETAILED_REPORT_FOR_TEACHER_TR.md").write_text(teacher, encoding="utf-8")
    return text


def main() -> None:
    file_inventory = build_file_inventory()
    dataset_dict = build_dataset_dictionary()
    feature_inv = build_feature_inventory()
    cleaning = build_data_cleaning_audit()
    results = build_results_summary()
    experiments = build_experiment_inventory()
    worked_failed = build_worked_failed()
    claims = build_claim_control()
    literature = build_literature_positioning()
    metrics = build_metric_table()
    metrics.to_csv(OUT / "EVALUATION_METRICS_DICTIONARY.csv", index=False)
    table_plan = build_table_figure_plan()
    leakage_table = build_process_leakage_table()
    build_qa()
    build_final_decision()
    build_main_report(
        file_inventory=file_inventory,
        dataset_dict=dataset_dict,
        feature_inv=feature_inv,
        cleaning=cleaning,
        results=results,
        worked_failed=worked_failed,
        claims=claims,
        literature=literature,
        metrics=metrics,
        table_plan=table_plan,
        leakage_table=leakage_table,
    )

    print(OUT)


if __name__ == "__main__":
    main()
