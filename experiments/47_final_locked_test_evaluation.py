"""Held-out evidence for final-attempt locked policies.

The final policy selection was already locked by
``experiments/46_final_multiobjective_selector.py``. This script only collects
held-out evidence for those locked systems and V2 references. It does not rank,
select, retune, recalibrate, or change any policy based on the test set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCKED_JSON = (
    ROOT
    / "outputs"
    / "final_attempt"
    / "final_selection"
    / "final_selected_locked_policies.json"
)
OUT_DIR = ROOT / "outputs" / "final_attempt" / "locked_test"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file or return an empty DataFrame."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_float(value: Any, default: float = np.nan) -> float:
    """Convert a value to float while preserving NaN for missing data."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "na", "none"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(value: Any) -> bool:
    """Return True if value can be interpreted as a finite float."""
    return np.isfinite(as_float(value))


def first_existing(row: pd.Series, *names: str) -> Any:
    """Return the first non-null value among possible column names."""
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return np.nan


def safe_div(num: float, den: float) -> float:
    """Safely divide two floats."""
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return np.nan
    return num / den


def load_locked_policy() -> dict[str, Any]:
    """Load the validation-locked policy definition."""
    if not LOCKED_JSON.exists():
        raise FileNotFoundError(f"Locked policy JSON not found: {LOCKED_JSON}")
    with LOCKED_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("test_set_used_for_selection") is not False:
        raise RuntimeError("Locked policy JSON does not assert validation-only selection.")
    return data


def v2_reference(dataset: str) -> pd.Series:
    """Return the V2 locked-test reference row for a dataset."""
    path = ROOT / "outputs" / "decision_revision_v2" / f"locked_test_evaluation_{dataset}.csv"
    df = read_csv(path)
    if df.empty:
        raise FileNotFoundError(f"Missing V2 locked-test reference: {path}")
    return df.iloc[0]


def estimate_review_count(row: pd.Series) -> float:
    """Estimate manual review count from explicit count or rate."""
    count = as_float(first_existing(row, "test_manual_review_count"))
    if np.isfinite(count):
        return count
    rate = as_float(first_existing(row, "test_manual_review_rate"))
    if np.isfinite(rate):
        total = as_float(first_existing(row, "test_rows"))
        if np.isfinite(total):
            return rate * total
        tn = as_float(first_existing(row, "test_tn"))
        fp = as_float(first_existing(row, "test_fp"))
        fn = as_float(first_existing(row, "test_fn"))
        tp = as_float(first_existing(row, "test_tp"))
        observed = sum(v for v in [tn, fp, fn, tp] if np.isfinite(v))
        if observed > 0 and rate < 1:
            return observed * rate / (1.0 - rate)
    return np.nan


def standard_metrics_from_row(row: pd.Series) -> dict[str, float]:
    """Normalize test metrics from heterogeneous locked-test output rows."""
    tn = as_float(first_existing(row, "test_tn"))
    fp = as_float(first_existing(row, "test_fp"))
    fn = as_float(first_existing(row, "test_fn"))
    tp = as_float(first_existing(row, "test_tp"))
    total_auto = tn + fp + fn + tp
    precision = first_finite(as_float(first_existing(row, "test_precision")), safe_div(tp, tp + fp))
    recall = first_finite(as_float(first_existing(row, "test_recall")), safe_div(tp, tp + fn))
    specificity = first_finite(as_float(first_existing(row, "test_specificity")), safe_div(tn, tn + fp))
    f1 = first_finite(
        as_float(first_existing(row, "test_f1")),
        safe_div(2 * precision * recall, precision + recall),
    )
    accuracy = first_finite(
        as_float(first_existing(row, "test_accuracy")),
        safe_div(tp + tn, total_auto),
    )
    fn_cost = as_float(first_existing(row, "fn_cost"), 5.0)
    fp_cost = as_float(first_existing(row, "fp_cost"), 1.0)
    residual_cost = fp_cost * fp + fn_cost * fn if np.isfinite(fp + fn) else np.nan
    review_adjusted = first_finite(
        as_float(first_existing(row, "test_review_adjusted_cost")),
        as_float(first_existing(row, "test_manual_review_adjusted_cost")),
        as_float(first_existing(row, "test_operational_cost")),
        np.nan,
    )
    return {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_specificity": specificity,
        "test_f1": f1,
        "test_pr_auc": as_float(first_existing(row, "test_pr_auc")),
        "test_roc_auc": as_float(first_existing(row, "test_roc_auc")),
        "test_brier": as_float(first_existing(row, "test_brier")),
        "test_ece": as_float(first_existing(row, "test_ece")),
        "test_tn": tn,
        "test_fp": fp,
        "test_fn": fn,
        "test_tp": tp,
        "test_expected_cost_binary": residual_cost,
        "test_review_adjusted_cost": review_adjusted,
        "test_manual_review_rate": as_float(first_existing(row, "test_manual_review_rate")),
        "test_auto_decision_rate": as_float(first_existing(row, "test_auto_decision_rate")),
        "test_high_risk_precision": first_finite(
            as_float(first_existing(row, "test_high_risk_precision")),
            as_float(first_existing(row, "test_high_risk_default_rate")),
            precision,
        ),
        "test_high_risk_recall": first_finite(
            as_float(first_existing(row, "test_high_risk_recall")), recall
        ),
        "test_low_risk_default_rate": as_float(
            first_existing(row, "test_low_risk_default_rate")
        ),
    }


def first_finite(*values: float) -> float:
    """Return the first finite float in a sequence."""
    for value in values:
        if np.isfinite(value):
            return value
    return np.nan


def capacity_row(dataset: str, model_label: str) -> pd.Series | None:
    """Fetch held-out diagnostic capacity metrics for a model label."""
    path = ROOT / "outputs" / "final_attempt" / "capacity_review" / "capacity_model_comparison.csv"
    df = read_csv(path)
    if df.empty:
        return None
    df = df[
        df["dataset"].astype(str).str.lower().eq(dataset)
        & df["split"].astype(str).str.lower().eq("heldout_test_diagnostic")
    ].copy()
    if df.empty:
        return None
    label = model_label.lower()
    mapping = {
        "scre-optimized": "SCRE-Optimized probability ranking",
        "scre-optimized probability ranking": "SCRE-Optimized probability ranking",
        "monotonic lightgbm": "Best EBM/monotonic model: Monotonic LightGBM",
        "scorecard": "Best Scorecard probability ranking",
        "best scorecard": "V2 final locked policy ranking",
        "v2": "V2 final locked policy ranking",
    }
    target = None
    for key, mapped in mapping.items():
        if key in label:
            target = mapped
            break
    if target is None:
        return None
    matched = df[df["model"].astype(str).eq(target)]
    return None if matched.empty else matched.iloc[0]


def decision_curve_summary(dataset: str, model_label: str) -> str:
    """Return compact held-out decision-curve evidence for a model label."""
    path = ROOT / "outputs" / "final_attempt" / "decision_curve" / "decision_curve_best_ranges.csv"
    df = read_csv(path)
    if df.empty:
        return "not available"
    df = df[
        df["dataset"].astype(str).str.lower().eq(dataset)
        & df["split"].astype(str).str.lower().eq("heldout_test_diagnostic")
    ].copy()
    label = model_label.lower()
    mapping = {
        "scre-optimized": "SCRE-Optimized probability ranking",
        "scre-optimized probability ranking": "SCRE-Optimized probability ranking",
        "monotonic lightgbm": "Best EBM/monotonic model: Monotonic LightGBM",
        "scorecard": "Best Scorecard probability ranking",
        "best scorecard": "V2 final locked policy ranking",
        "v2": "V2 final locked policy ranking",
    }
    target = None
    for key, mapped in mapping.items():
        if key in label:
            target = mapped
            break
    if target is None:
        return "not available for this exact locked system"
    matched = df[df["model"].astype(str).eq(target)]
    if matched.empty:
        return "not available"
    row = matched.iloc[0]
    return (
        f"useful_range={row.get('useful_threshold_range')}; "
        f"max_nb={as_float(row.get('max_net_benefit')):.6f}; "
        f"best_range={row.get('best_model_at_threshold_range')}"
    )


def base_output_row(
    *,
    dataset: str,
    system_name: str,
    model: str,
    policy_type: str,
    threshold: Any,
    t_low: Any,
    t_high: Any,
    calibration_type: Any,
    source: str,
    comment: str,
) -> dict[str, Any]:
    """Return the common output row skeleton."""
    return {
        "dataset": dataset,
        "system_name": system_name,
        "model": model,
        "policy_type": policy_type,
        "threshold": as_float(threshold),
        "t_low": as_float(t_low),
        "t_high": as_float(t_high),
        "calibration_type": calibration_type,
        "source_evidence_file": source,
        "test_accuracy": np.nan,
        "test_precision": np.nan,
        "test_recall": np.nan,
        "test_specificity": np.nan,
        "test_f1": np.nan,
        "test_pr_auc": np.nan,
        "test_roc_auc": np.nan,
        "test_brier": np.nan,
        "test_ece": np.nan,
        "test_tn": np.nan,
        "test_fp": np.nan,
        "test_fn": np.nan,
        "test_tp": np.nan,
        "test_expected_cost_binary": np.nan,
        "test_review_adjusted_cost": np.nan,
        "test_manual_review_rate": np.nan,
        "test_auto_decision_rate": np.nan,
        "test_high_risk_precision": np.nan,
        "test_high_risk_recall": np.nan,
        "test_low_risk_default_rate": np.nan,
        "test_default_capture_at_10": np.nan,
        "test_default_capture_at_20": np.nan,
        "test_default_capture_at_30": np.nan,
        "test_precision_at_20": np.nan,
        "test_lift_at_20": np.nan,
        "test_net_benefit_summary": np.nan,
        "comparison_to_v2": np.nan,
        "comment": comment,
        "policy_changed_after_test": False,
        "test_set_used_for_selection": False,
    }


def apply_metrics(out: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    """Apply normalized test metrics from a source row to output."""
    out.update(standard_metrics_from_row(row))
    return out


def apply_capacity(out: dict[str, Any], cap: pd.Series | None) -> dict[str, Any]:
    """Apply held-out capacity metrics to output when available."""
    if cap is None:
        return out
    out["test_default_capture_at_10"] = as_float(cap.get("top_10_capture"))
    out["test_default_capture_at_20"] = as_float(cap.get("top_20_capture"))
    out["test_default_capture_at_30"] = as_float(cap.get("top_30_capture"))
    out["test_precision_at_20"] = as_float(cap.get("precision_at_20"))
    out["test_lift_at_20"] = as_float(cap.get("lift_at_20"))
    return out


def scre_test_row(dataset: str) -> pd.Series:
    """Return selected SCRE-Optimized test row for a dataset."""
    path = ROOT / "outputs" / "tables" / f"scre_optimized_results_{dataset}.csv"
    df = read_csv(path)
    if df.empty:
        raise FileNotFoundError(path)
    df = df[
        df["split"].astype(str).str.lower().eq("test")
        & df["selected_final_setting"].astype(bool)
    ]
    if df.empty:
        raise RuntimeError(f"Selected SCRE test row not found for {dataset}")
    return df.iloc[0]


def row_from_scre(dataset: str) -> dict[str, Any]:
    """Build the SCRE research-framework test evidence row."""
    source = f"outputs/tables/scre_optimized_results_{dataset}.csv"
    r = scre_test_row(dataset)
    out = base_output_row(
        dataset=dataset,
        system_name="Research framework",
        model="SCRE-Optimized",
        policy_type="SCRE_binary_cost_policy",
        threshold=r.get("threshold"),
        t_low=np.nan,
        t_high=np.nan,
        calibration_type=r.get("calibration_type"),
        source=source,
        comment="Binary SCRE test evidence; not selected after seeing test results.",
    )
    mapped = pd.Series(
        {
            "test_accuracy": r.get("accuracy"),
            "test_precision": r.get("precision"),
            "test_recall": r.get("recall"),
            "test_specificity": r.get("specificity"),
            "test_f1": r.get("f1"),
            "test_pr_auc": r.get("pr_auc"),
            "test_roc_auc": r.get("roc_auc"),
            "test_brier": r.get("brier_score"),
            "test_ece": r.get("ece"),
            "test_tn": r.get("tn"),
            "test_fp": r.get("fp"),
            "test_fn": r.get("fn"),
            "test_tp": r.get("tp"),
            "fn_cost": r.get("fn_cost"),
            "fp_cost": r.get("fp_cost"),
        }
    )
    apply_metrics(out, mapped)
    out["test_review_adjusted_cost"] = np.nan
    out["test_net_benefit_summary"] = decision_curve_summary(dataset, "SCRE-Optimized")
    apply_capacity(out, capacity_row(dataset, "SCRE-Optimized"))
    return out


def row_from_advanced_taiwan() -> dict[str, Any]:
    """Build Taiwan final operational XGBoost test evidence row."""
    rel = "outputs/final_attempt/imbalance_boosting/advanced_imbalance_locked_test_taiwan.csv"
    df = read_csv(ROOT / rel)
    if df.empty:
        raise FileNotFoundError(rel)
    r = df.iloc[0]
    out = base_output_row(
        dataset="taiwan",
        system_name="Final operational screening",
        model=f"{r.get('model_family')} {r.get('variant_name')}",
        policy_type=r.get("policy_type"),
        threshold=r.get("threshold"),
        t_low=r.get("t_low"),
        t_high=r.get("t_high"),
        calibration_type=r.get("calibration_type"),
        source=rel,
        comment=(
            "Locked Taiwan final operational policy. Manual-review cost is "
            "review-adjusted; binary cost is automation residual only."
        ),
    )
    apply_metrics(out, r)
    out["test_net_benefit_summary"] = decision_curve_summary("taiwan", out["model"])
    apply_capacity(out, capacity_row("taiwan", out["model"]))
    return out


def row_from_v2_heloc() -> dict[str, Any]:
    """Build HELOC final operational Scorecard test evidence row."""
    rel = "outputs/decision_revision_v2/locked_test_evaluation_heloc.csv"
    r = v2_reference("heloc")
    out = base_output_row(
        dataset="heloc",
        system_name="Final operational screening",
        model=str(r.get("selected_model")),
        policy_type=str(r.get("selected_policy_type")),
        threshold=r.get("threshold"),
        t_low=r.get("t_low"),
        t_high=r.get("t_high"),
        calibration_type=r.get("calibration_type"),
        source=rel,
        comment="Locked HELOC final operational policy inherited from V2 Scorecard.",
    )
    apply_metrics(out, r)
    out["test_net_benefit_summary"] = str(r.get("test_net_benefit_summary", "not available"))
    apply_capacity(out, capacity_row("heloc", "V2"))
    return out


def row_from_interpretable(dataset: str) -> dict[str, Any]:
    """Build interpretable benchmark test evidence row."""
    rel = f"outputs/final_attempt/interpretable_models/interpretable_models_locked_test_{dataset}.csv"
    df = read_csv(ROOT / rel)
    if df.empty:
        raise FileNotFoundError(rel)
    if dataset == "taiwan":
        df = df[df["model"].astype(str).eq("Monotonic LightGBM conservative")]
    else:
        df = df[df["model"].astype(str).eq("WOE scorecard unweighted")]
    if df.empty:
        raise RuntimeError(f"Interpretable locked row missing for {dataset}")
    r = df.iloc[0]
    out = base_output_row(
        dataset=dataset,
        system_name="Interpretable benchmark",
        model=str(r.get("model")),
        policy_type=str(r.get("policy_type")),
        threshold=r.get("threshold"),
        t_low=r.get("t_low"),
        t_high=r.get("t_high"),
        calibration_type=r.get("calibration_type"),
        source=rel,
        comment="Interpretable locked benchmark; evidence only, no test ranking.",
    )
    apply_metrics(out, r)
    out["test_net_benefit_summary"] = decision_curve_summary(dataset, out["model"])
    apply_capacity(out, capacity_row(dataset, out["model"]))
    return out


def row_from_capacity(dataset: str) -> dict[str, Any]:
    """Build SCRE capacity-review prioritization held-out evidence row."""
    model = "SCRE-Optimized probability ranking"
    out = base_output_row(
        dataset=dataset,
        system_name="Review prioritization",
        model=model,
        policy_type="top_k_review_prioritization",
        threshold=np.nan,
        t_low=np.nan,
        t_high=np.nan,
        calibration_type="probability_ranking",
        source="outputs/final_attempt/capacity_review/capacity_model_comparison.csv",
        comment=(
            "Top-k review prioritization evidence. This is not a binary or "
            "manual-review threshold policy."
        ),
    )
    apply_capacity(out, capacity_row(dataset, model))
    out["test_net_benefit_summary"] = decision_curve_summary(dataset, model)
    return out


def add_v2_deltas(rows: list[dict[str, Any]], dataset: str) -> None:
    """Add deltas versus the V2 final locked reference for a dataset."""
    ref = v2_reference(dataset)
    ref_metrics = standard_metrics_from_row(ref)
    ref_capture = capacity_row(dataset, "V2")
    ref_capture20 = as_float(ref_capture.get("top_20_capture")) if ref_capture is not None else np.nan
    for out in rows:
        if out["dataset"] != dataset:
            continue
        out["delta_precision"] = as_float(out["test_precision"]) - ref_metrics["test_precision"]
        out["delta_recall"] = as_float(out["test_recall"]) - ref_metrics["test_recall"]
        out["delta_specificity"] = as_float(out["test_specificity"]) - ref_metrics["test_specificity"]
        out["delta_fp"] = as_float(out["test_fp"]) - ref_metrics["test_fp"]
        out["delta_fn"] = as_float(out["test_fn"]) - ref_metrics["test_fn"]
        out["delta_cost"] = first_finite(
            as_float(out["test_review_adjusted_cost"]),
            as_float(out["test_expected_cost_binary"]),
        ) - first_finite(
            ref_metrics["test_review_adjusted_cost"],
            ref_metrics["test_expected_cost_binary"],
        )
        out["delta_mr_rate"] = (
            as_float(out["test_manual_review_rate"]) - ref_metrics["test_manual_review_rate"]
        )
        out["delta_capture_at_20"] = as_float(out["test_default_capture_at_20"]) - ref_capture20
        out["comparison_to_v2"] = comparison_text(out)


def comparison_text(row: dict[str, Any]) -> str:
    """Create compact comparison text against V2."""
    parts: list[str] = []
    if finite(row.get("delta_precision")):
        parts.append(f"precision {signed(row['delta_precision'])}")
    if finite(row.get("delta_recall")):
        parts.append(f"recall {signed(row['delta_recall'])}")
    if finite(row.get("delta_specificity")):
        parts.append(f"specificity {signed(row['delta_specificity'])}")
    if finite(row.get("delta_cost")):
        parts.append(f"cost {signed(row['delta_cost'], digits=1)}")
    if finite(row.get("delta_capture_at_20")):
        parts.append(f"capture@20 {signed(row['delta_capture_at_20'])}")
    if not parts:
        return "No comparable V2 metric for this row."
    return "vs V2: " + ", ".join(parts)


def signed(value: Any, digits: int = 4) -> str:
    """Format signed metric deltas."""
    number = as_float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:+.{digits}f}"


def build_outputs() -> pd.DataFrame:
    """Build held-out evidence rows for all final locked systems."""
    load_locked_policy()
    rows = [
        row_from_advanced_taiwan(),
        row_from_capacity("taiwan"),
        row_from_interpretable("taiwan"),
        row_from_scre("taiwan"),
        row_from_v2_heloc(),
        row_from_capacity("heloc"),
        row_from_interpretable("heloc"),
        row_from_scre("heloc"),
    ]
    add_v2_deltas(rows, "taiwan")
    add_v2_deltas(rows, "heloc")
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame) -> None:
    """Write Markdown summary for held-out evidence."""
    taiwan = df[df["dataset"].eq("taiwan")].copy()
    heloc = df[df["dataset"].eq("heloc")].copy()
    tw_op = taiwan[taiwan["system_name"].eq("Final operational screening")].iloc[0]
    he_op = heloc[heloc["system_name"].eq("Final operational screening")].iloc[0]
    lines = [
        "# Final Locked Test Evaluation Summary",
        "",
        "This file reports held-out evidence only. The final policies were locked in `final_selected_locked_policies.json` before this evaluation. No threshold, model, calibration, or policy was changed after seeing test results.",
        "",
        "## 1. Yeni final system V2'den iyi mi?",
        "",
        "- Taiwan: not uniformly. The new XGBoost manual-review policy has slightly higher recall, but lower precision/specificity and higher review-adjusted cost than the V2 CatBoost manual-review reference.",
        "- HELOC: unchanged operationally; the final operational policy remains the V2 Scorecard manual-review policy.",
        "",
        "## 2. Hangi metrikte iyi?",
        "",
        f"- Taiwan operational XGBoost improves recall by {signed(tw_op['delta_recall'])} versus V2.",
        f"- SCRE-Optimized review prioritization improves Taiwan capture@20 by {signed(taiwan[taiwan['system_name'].eq('Review prioritization')].iloc[0]['delta_capture_at_20'])} and HELOC capture@20 by {signed(heloc[heloc['system_name'].eq('Review prioritization')].iloc[0]['delta_capture_at_20'])}.",
        "",
        "## 3. Hangi metrikte kotu?",
        "",
        f"- Taiwan operational XGBoost has precision delta {signed(tw_op['delta_precision'])}, specificity delta {signed(tw_op['delta_specificity'])}, and review-adjusted cost delta {signed(tw_op['delta_cost'], digits=1)} versus V2.",
        "- HELOC operational system is the same as V2, so there is no operational degradation from V2 on the locked final policy.",
        "",
        "## 4. Skor artisi mi, operational denge mi saglandi?",
        "",
        "- The strongest improvement is not a simple score increase. It is a clearer separation of roles: XGBoost/Scorecard for operational screening, SCRE-Optimized for review prioritization, and interpretable benchmarks for governance.",
        "",
        "## 5. Automatic rejection uygun mu?",
        "",
        "- No. Precision remains limited and manual-review policies explicitly route uncertain cases to review. These results should not be framed as automatic credit rejection.",
        "",
        "## 6. Screening/manual review uygun mu?",
        "",
        "- Yes, conditionally. The evidence supports screening and review prioritization, especially using manual-review bands and top-k review capacity analysis.",
        "",
        "## Held-out Operational Rows",
        "",
        "| Dataset | System | Precision | Recall | Specificity | FP | FN | Review Cost | MR Rate | Capture@20 | Comment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df[df["system_name"].isin(["Final operational screening", "Review prioritization"])].iterrows():
        lines.append(
            f"| {r['dataset']} | {r['system_name']} / {r['model']} | "
            f"{fmt(r['test_precision'])} | {fmt(r['test_recall'])} | "
            f"{fmt(r['test_specificity'])} | {fmt(r['test_fp'], 0)} | "
            f"{fmt(r['test_fn'], 0)} | {fmt(r['test_review_adjusted_cost'], 1)} | "
            f"{fmt(r['test_manual_review_rate'])} | {fmt(r['test_default_capture_at_20'])} | "
            f"{r['comparison_to_v2']} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Status",
            "",
            "- Test set used for selection: NO.",
            "- Test set used for threshold changes: NO.",
            "- Test set used for calibration changes: NO.",
            "- Test ranking produced: NO.",
        ]
    )
    (OUT_DIR / "final_locked_test_evaluation_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def fmt(value: Any, digits: int = 4) -> str:
    """Format numeric cells for Markdown."""
    number = as_float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def main() -> None:
    """Run held-out evidence collection for locked final systems."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_outputs()
    ordered_cols = [
        "dataset",
        "system_name",
        "model",
        "policy_type",
        "threshold",
        "t_low",
        "t_high",
        "calibration_type",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_f1",
        "test_pr_auc",
        "test_roc_auc",
        "test_brier",
        "test_ece",
        "test_tn",
        "test_fp",
        "test_fn",
        "test_tp",
        "test_expected_cost_binary",
        "test_review_adjusted_cost",
        "test_manual_review_rate",
        "test_auto_decision_rate",
        "test_high_risk_precision",
        "test_high_risk_recall",
        "test_low_risk_default_rate",
        "test_default_capture_at_10",
        "test_default_capture_at_20",
        "test_default_capture_at_30",
        "test_precision_at_20",
        "test_lift_at_20",
        "test_net_benefit_summary",
        "comparison_to_v2",
        "delta_precision",
        "delta_recall",
        "delta_specificity",
        "delta_fp",
        "delta_fn",
        "delta_cost",
        "delta_mr_rate",
        "delta_capture_at_20",
        "comment",
        "source_evidence_file",
        "policy_changed_after_test",
        "test_set_used_for_selection",
    ]
    df = df[ordered_cols]
    df[df["dataset"].eq("taiwan")].to_csv(
        OUT_DIR / "final_locked_test_evaluation_taiwan.csv", index=False
    )
    df[df["dataset"].eq("heloc")].to_csv(
        OUT_DIR / "final_locked_test_evaluation_heloc.csv", index=False
    )
    write_summary(df)
    print(f"Wrote held-out evidence to {OUT_DIR}")
    print(
        df[
            [
                "dataset",
                "system_name",
                "model",
                "test_precision",
                "test_recall",
                "test_specificity",
                "test_fp",
                "test_fn",
                "test_review_adjusted_cost",
                "test_manual_review_rate",
                "test_default_capture_at_20",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
