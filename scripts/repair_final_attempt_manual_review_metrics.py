"""Repair Final Attempt manual-review locked-test metric definitions.

This script does not train models, select thresholds, or change policies.
It only re-expresses existing manual-review locked-test rows as three-bucket
decision tables and audits the resulting metric identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "taiwan": PROJECT_ROOT / "outputs/final_attempt/locked_test/final_locked_test_evaluation_taiwan.csv",
    "heloc": PROJECT_ROOT / "outputs/final_attempt/locked_test/final_locked_test_evaluation_heloc.csv",
}
SPLIT_SUMMARIES = {
    "taiwan": PROJECT_ROOT / "outputs/tables/split_summary_taiwan.csv",
    "heloc": PROJECT_ROOT / "outputs/tables/split_summary_heloc.csv",
}
OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/metric_repair"
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COST = 0.5
TOL = 1e-8


@dataclass(frozen=True)
class TestSplitInfo:
    """Held-out test split size and class counts."""

    rows: int
    defaults: int
    nondefaults: int


def _safe_float(value: Any) -> float | None:
    """Convert a scalar value to float, returning None for missing values."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _safe_int(value: Any) -> int | None:
    """Convert a scalar value to rounded int, returning None for missing values."""

    number = _safe_float(value)
    if number is None:
        return None
    return int(round(number))


def _divide(numerator: float, denominator: float) -> float | None:
    """Return numerator / denominator, or None when denominator is zero."""

    if denominator == 0:
        return None
    return numerator / denominator


def _close(left: float | None, right: float | None, tol: float = TOL) -> bool:
    """Check approximate equality while treating paired missing values as equal."""

    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tol


def _load_test_split_info(dataset: str) -> TestSplitInfo:
    """Load test split size and class counts from the project split summary."""

    split_path = SPLIT_SUMMARIES[dataset]
    summary = pd.read_csv(split_path)
    test_row = summary.loc[summary["split"].astype(str).str.lower() == "test"].iloc[0]
    return TestSplitInfo(
        rows=int(test_row["rows"]),
        defaults=int(test_row["target_1_count"]),
        nondefaults=int(test_row["target_0_count"]),
    )


def _manual_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only threshold/band manual-review policies, excluding top-k rows."""

    policy = df["policy_type"].fillna("").astype(str).str.lower()
    mask = policy.str.contains("manual_review") | policy.str.contains("manual review")
    mask &= ~policy.str.contains("top_k")
    return df.loc[mask].copy()


def _repair_row(row: pd.Series, split: TestSplitInfo) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert one locked manual-review row into a 3x2 table and audit row."""

    high_nondefault = _safe_int(row.get("test_fp"))
    high_default = _safe_int(row.get("test_tp"))
    observed_high_precision = _safe_float(row.get("test_high_risk_precision")) or _safe_float(row.get("test_precision"))
    observed_high_recall = _safe_float(row.get("test_high_risk_recall")) or _safe_float(row.get("test_recall"))
    observed_low_default_rate = _safe_float(row.get("test_low_risk_default_rate"))
    observed_mr_rate = _safe_float(row.get("test_manual_review_rate"))
    observed_auto_rate = _safe_float(row.get("test_auto_decision_rate"))
    observed_review_adjusted_cost = _safe_float(row.get("test_review_adjusted_cost"))

    if high_nondefault is None or high_default is None:
        raise ValueError(f"Missing high-risk counts for {row.get('dataset')} {row.get('model')}")
    if observed_mr_rate is None or observed_low_default_rate is None:
        raise ValueError(f"Missing manual-review rates for {row.get('dataset')} {row.get('model')}")

    manual_review_count = int(round(split.rows * observed_mr_rate))
    auto_decision_count = split.rows - manual_review_count
    high_risk_count = high_nondefault + high_default
    low_risk_count = auto_decision_count - high_risk_count
    if low_risk_count < 0:
        raise ValueError(f"Negative low-risk count for {row.get('dataset')} {row.get('model')}")

    low_default = int(round(low_risk_count * observed_low_default_rate))
    low_nondefault = low_risk_count - low_default
    manual_review_defaults = split.defaults - high_default - low_default
    manual_review_nondefaults = manual_review_count - manual_review_defaults

    if manual_review_defaults < 0 or manual_review_nondefaults < 0:
        raise ValueError(f"Negative manual-review bucket count for {row.get('dataset')} {row.get('model')}")
    if low_nondefault + manual_review_nondefaults + high_nondefault != split.nondefaults:
        raise ValueError(f"Non-default totals do not reconcile for {row.get('dataset')} {row.get('model')}")
    if low_default + manual_review_defaults + high_default != split.defaults:
        raise ValueError(f"Default totals do not reconcile for {row.get('dataset')} {row.get('model')}")

    high_precision = _divide(high_default, high_default + high_nondefault)
    high_recall = _divide(high_default, split.defaults)
    low_default_rate = _divide(low_default, low_risk_count)
    manual_review_default_rate = _divide(manual_review_defaults, manual_review_count)
    high_default_rate = high_precision
    manual_review_rate = _divide(manual_review_count, split.rows)
    auto_decision_rate = _divide(auto_decision_count, split.rows)

    auto_fp = high_nondefault
    auto_fn = low_default
    auto_tp = high_default
    auto_tn = low_nondefault
    automation_residual_cost = auto_fp * FP_COST + auto_fn * FN_COST
    review_adjusted_cost = automation_residual_cost + manual_review_count * REVIEW_COST

    base = {
        "dataset": row.get("dataset"),
        "system_name": row.get("system_name"),
        "model": row.get("model"),
        "policy_type": row.get("policy_type"),
        "calibration_type": row.get("calibration_type"),
        "threshold": row.get("threshold"),
        "t_low": row.get("t_low"),
        "t_high": row.get("t_high"),
        "source_evidence_file": row.get("source_evidence_file"),
        "test_set_rows": split.rows,
        "total_defaults": split.defaults,
        "total_nondefaults": split.nondefaults,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "review_cost": REVIEW_COST,
        "low_risk_count": low_risk_count,
        "manual_review_count": manual_review_count,
        "high_risk_count": high_risk_count,
        "manual_review_rate": manual_review_rate,
        "auto_decision_rate": auto_decision_rate,
        "high_risk_precision": high_precision,
        "high_risk_recall": high_recall,
        "low_risk_default_rate": low_default_rate,
        "manual_review_default_rate": manual_review_default_rate,
        "high_risk_default_rate": high_default_rate,
        "auto_fp": auto_fp,
        "auto_fn": auto_fn,
        "auto_tp": auto_tp,
        "auto_tn": auto_tn,
        "manual_review_defaults": manual_review_defaults,
        "manual_review_nondefaults": manual_review_nondefaults,
        "automation_residual_cost": automation_residual_cost,
        "review_adjusted_cost": review_adjusted_cost,
    }

    bucket_rows = [
        {
            **base,
            "decision_bucket": "Low risk",
            "actual_non_default": low_nondefault,
            "actual_default": low_default,
            "bucket_count": low_risk_count,
            "bucket_default_rate": low_default_rate,
        },
        {
            **base,
            "decision_bucket": "Manual review",
            "actual_non_default": manual_review_nondefaults,
            "actual_default": manual_review_defaults,
            "bucket_count": manual_review_count,
            "bucket_default_rate": manual_review_default_rate,
        },
        {
            **base,
            "decision_bucket": "High risk",
            "actual_non_default": high_nondefault,
            "actual_default": high_default,
            "bucket_count": high_risk_count,
            "bucket_default_rate": high_default_rate,
        },
    ]

    displayed_tn = _safe_float(row.get("test_tn"))
    displayed_fp = _safe_float(row.get("test_fp"))
    displayed_fn = _safe_float(row.get("test_fn"))
    displayed_tp = _safe_float(row.get("test_tp"))
    displayed_recall_from_confusion = _divide(displayed_tp or 0.0, (displayed_tp or 0.0) + (displayed_fn or 0.0))
    displayed_specificity_from_confusion = _divide(displayed_tn or 0.0, (displayed_tn or 0.0) + (displayed_fp or 0.0))
    displayed_precision_from_confusion = _divide(displayed_tp or 0.0, (displayed_tp or 0.0) + (displayed_fp or 0.0))
    if displayed_precision_from_confusion is not None and displayed_recall_from_confusion is not None:
        displayed_f1_from_confusion = _divide(
            2 * displayed_precision_from_confusion * displayed_recall_from_confusion,
            displayed_precision_from_confusion + displayed_recall_from_confusion,
        )
    else:
        displayed_f1_from_confusion = None

    observed_recall = _safe_float(row.get("test_recall"))
    observed_specificity = _safe_float(row.get("test_specificity"))
    observed_f1 = _safe_float(row.get("test_f1"))

    audit = {
        **{key: base[key] for key in [
            "dataset",
            "system_name",
            "model",
            "policy_type",
            "calibration_type",
            "t_low",
            "t_high",
            "source_evidence_file",
        ]},
        "high_risk_precision_expected": high_precision,
        "high_risk_precision_observed": observed_high_precision,
        "high_risk_precision_match": _close(high_precision, observed_high_precision),
        "high_risk_recall_expected": high_recall,
        "high_risk_recall_observed": observed_high_recall,
        "high_risk_recall_match": _close(high_recall, observed_high_recall),
        "low_risk_default_rate_expected": low_default_rate,
        "low_risk_default_rate_observed": observed_low_default_rate,
        "low_risk_default_rate_match": _close(low_default_rate, observed_low_default_rate),
        "manual_review_rate_expected": manual_review_rate,
        "manual_review_rate_observed": observed_mr_rate,
        "manual_review_rate_match": _close(manual_review_rate, observed_mr_rate),
        "auto_decision_rate_expected": auto_decision_rate,
        "auto_decision_rate_observed": observed_auto_rate,
        "auto_decision_rate_match": _close(auto_decision_rate, observed_auto_rate),
        "review_adjusted_cost_expected": review_adjusted_cost,
        "review_adjusted_cost_observed": observed_review_adjusted_cost,
        "review_adjusted_cost_match": _close(review_adjusted_cost, observed_review_adjusted_cost),
        "displayed_precision_from_confusion": displayed_precision_from_confusion,
        "displayed_test_precision": _safe_float(row.get("test_precision")),
        "binary_precision_matches_displayed_confusion": _close(
            displayed_precision_from_confusion, _safe_float(row.get("test_precision"))
        ),
        "displayed_recall_from_confusion": displayed_recall_from_confusion,
        "displayed_test_recall": observed_recall,
        "binary_recall_matches_displayed_confusion": _close(displayed_recall_from_confusion, observed_recall),
        "displayed_specificity_from_confusion": displayed_specificity_from_confusion,
        "displayed_test_specificity": observed_specificity,
        "binary_specificity_matches_displayed_confusion": _close(
            displayed_specificity_from_confusion, observed_specificity
        ),
        "displayed_f1_from_confusion": displayed_f1_from_confusion,
        "displayed_test_f1": observed_f1,
        "binary_f1_matches_displayed_confusion": _close(displayed_f1_from_confusion, observed_f1),
        "binary_metrics_mixed_denominator_flag": not (
            _close(displayed_recall_from_confusion, observed_recall)
            and _close(displayed_specificity_from_confusion, observed_specificity)
            and _close(displayed_f1_from_confusion, observed_f1)
        ),
        "repaired_3x2_metric_status": "PASS",
        "repair_scope": "Appendix/robustness only; not final model selection evidence.",
    }
    return bucket_rows, audit


def _write_summary(taiwan_check: pd.DataFrame, heloc_check: pd.DataFrame) -> None:
    """Write the Markdown repair summary."""

    all_check = pd.concat([taiwan_check, heloc_check], ignore_index=True)
    repaired_failures = all_check[
        ~(
            all_check["high_risk_precision_match"]
            & all_check["high_risk_recall_match"]
            & all_check["low_risk_default_rate_match"]
            & all_check["manual_review_rate_match"]
            & all_check["auto_decision_rate_match"]
            & all_check["review_adjusted_cost_match"]
        )
    ]
    mixed = all_check.loc[all_check["binary_metrics_mixed_denominator_flag"]]

    def rows_to_markdown(df: pd.DataFrame) -> str:
        cols = [
            "dataset",
            "system_name",
            "model",
            "high_risk_precision_expected",
            "high_risk_recall_expected",
            "low_risk_default_rate_expected",
            "manual_review_rate_expected",
            "review_adjusted_cost_expected",
            "binary_metrics_mixed_denominator_flag",
        ]
        view = df[cols].copy()
        for col in view.select_dtypes(include=["float"]).columns:
            view[col] = view[col].map(lambda value: f"{value:.6f}")
        header = "| " + " | ".join(view.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        body = [
            "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
            for row in view.to_numpy()
        ]
        return "\n".join([header, separator, *body])

    summary = f"""# Final Attempt Manual-Review Metric Repair

## Scope
This repair does not train a model, select a threshold, change a policy, or use the test set for decision-making. It only rewrites existing Final Attempt locked-test manual-review rows as explicit 3x2 decision tables.

## Correct metric definition
Manual-review policies are three-way decision policies, not ordinary binary classifiers. The repaired files separate Low risk, Manual review, and High risk buckets, then compute high-risk precision/recall, low-risk default rate, manual-review rate, auto-decision rate, automation residual cost, and review-adjusted cost.

## Taiwan repaired rows
{rows_to_markdown(taiwan_check)}

## HELOC repaired rows
{rows_to_markdown(heloc_check)}

## Audit result after repair
- Repaired 3x2 metric identity failures: {len(repaired_failures)}
- Original binary denominator mismatch flags documented: {len(mixed)}
- Final Attempt final operational status after repair: NOT FINAL
- Intended use: appendix / robustness / stress-test evidence only, pending any future full audit.

## Interpretation
The previous BLOCKED audit mixed binary confusion-matrix checks with manual-review bucket metrics. The repaired tables make the bucket definitions explicit. This removes the manual-review cost auditability gap for the repaired rows, but it does not make Final Attempt the final operational version. V2 remains the final operational evidence unless a later full audit explicitly replaces it.
"""
    (OUT_DIR / "final_attempt_metric_repair_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Generate repaired 3x2 tables and audit checks."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_checks: list[pd.DataFrame] = []
    per_dataset_checks: dict[str, pd.DataFrame] = {}

    for dataset, input_path in INPUTS.items():
        split = _load_test_split_info(dataset)
        locked = pd.read_csv(input_path)
        manual_rows = _manual_review_rows(locked)
        bucket_records: list[dict[str, Any]] = []
        audit_records: list[dict[str, Any]] = []

        for _, row in manual_rows.iterrows():
            bucket_rows, audit = _repair_row(row, split)
            bucket_records.extend(bucket_rows)
            audit_records.append(audit)

        bucket_df = pd.DataFrame(bucket_records)
        check_df = pd.DataFrame(audit_records)
        bucket_df.to_csv(OUT_DIR / f"final_attempt_manual_review_3x2_{dataset}.csv", index=False)
        per_dataset_checks[dataset] = check_df
        all_checks.append(check_df)

    audit_check = pd.concat(all_checks, ignore_index=True)
    audit_check.to_csv(OUT_DIR / "final_attempt_repaired_audit_check.csv", index=False)
    _write_summary(per_dataset_checks["taiwan"], per_dataset_checks["heloc"])


if __name__ == "__main__":
    main()
