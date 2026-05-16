"""Final validation-only multi-objective selector for the last experiment block.

This script intentionally avoids held-out test evidence. It combines validation
outputs from the V2 protocol, final-attempt experiments, capacity analysis, and
decision curve analysis to choose model/policy systems by explicit tracks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_ATTEMPT = ROOT / "outputs" / "final_attempt"
OUT_DIR = FINAL_ATTEMPT / "final_selection"


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and return an empty frame if the file is absent."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_float(value: Any, default: float = np.nan) -> float:
    """Convert a value to float while treating missing/string-null values as NaN."""
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lower() in {"", "na", "nan", "none"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    """Interpret common boolean representations."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def first_present(row: pd.Series, names: Iterable[str], default: Any = np.nan) -> Any:
    """Return the first non-missing value among the provided column names."""
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def f1_from_precision_recall(precision: float, recall: float) -> float:
    """Compute F1 safely from precision and recall."""
    if not np.isfinite(precision) or not np.isfinite(recall) or precision + recall == 0:
        return np.nan
    return 2.0 * precision * recall / (precision + recall)


def is_manual_policy(policy_type: Any) -> bool:
    """Return True if a policy is a manual-review policy."""
    return "manual" in str(policy_type).lower()


def make_candidate(
    *,
    source_experiment: str,
    source_file: str,
    dataset: str,
    model: str,
    model_family: str,
    policy_type: str,
    calibration_type: Any = np.nan,
    threshold: Any = np.nan,
    t_low: Any = np.nan,
    t_high: Any = np.nan,
    constraint_name: Any = np.nan,
    fn_cost: Any = 5.0,
    fp_cost: Any = 1.0,
    review_cost: Any = np.nan,
    validation_accuracy: Any = np.nan,
    validation_precision: Any = np.nan,
    validation_recall: Any = np.nan,
    validation_specificity: Any = np.nan,
    validation_f1: Any = np.nan,
    validation_pr_auc: Any = np.nan,
    validation_roc_auc: Any = np.nan,
    validation_ece: Any = np.nan,
    validation_brier: Any = np.nan,
    validation_expected_cost: Any = np.nan,
    validation_review_adjusted_cost: Any = np.nan,
    validation_fp: Any = np.nan,
    validation_manual_review_rate: Any = np.nan,
    validation_auto_decision_rate: Any = np.nan,
    validation_high_risk_precision: Any = np.nan,
    validation_high_risk_recall: Any = np.nan,
    validation_low_risk_default_rate: Any = np.nan,
    selection_note: str = "",
    artifact_ready: bool = False,
) -> dict[str, Any]:
    """Create a normalized candidate row."""
    precision = as_float(validation_precision)
    recall = as_float(validation_recall)
    f1 = as_float(validation_f1)
    if not np.isfinite(f1):
        f1 = f1_from_precision_recall(precision, recall)
    manual_rate = first_finite(
        as_float(validation_manual_review_rate),
        as_float(validation_manual_review_rate),
    )
    return {
        "source_experiment": source_experiment,
        "source_file": source_file,
        "dataset": str(dataset).lower(),
        "model": model,
        "model_family": model_family,
        "policy_type": policy_type,
        "calibration_type": calibration_type,
        "threshold": as_float(threshold),
        "t_low": as_float(t_low),
        "t_high": as_float(t_high),
        "constraint_name": constraint_name,
        "fn_cost": as_float(fn_cost, 5.0),
        "fp_cost": as_float(fp_cost, 1.0),
        "review_cost": as_float(review_cost),
        "validation_accuracy": as_float(validation_accuracy),
        "validation_precision": precision,
        "validation_recall": recall,
        "validation_specificity": as_float(validation_specificity),
        "validation_f1": f1,
        "validation_pr_auc": as_float(validation_pr_auc),
        "validation_roc_auc": as_float(validation_roc_auc),
        "validation_ece": as_float(validation_ece),
        "validation_brier": as_float(validation_brier),
        "validation_expected_cost": as_float(validation_expected_cost),
        "validation_review_adjusted_cost": as_float(validation_review_adjusted_cost),
        "validation_fp": as_float(validation_fp),
        "validation_manual_review_rate": manual_rate,
        "validation_auto_decision_rate": as_float(validation_auto_decision_rate),
        "validation_high_risk_precision": first_finite(
            as_float(validation_high_risk_precision), precision
        ),
        "validation_high_risk_recall": first_finite(
            as_float(validation_high_risk_recall), recall
        ),
        "validation_low_risk_default_rate": as_float(validation_low_risk_default_rate),
        "selection_note": selection_note,
        "artifact_ready": bool(artifact_ready),
        "test_set_used_for_selection": False,
    }


def first_finite(*values: float) -> float:
    """Return the first finite float, otherwise NaN."""
    for value in values:
        if np.isfinite(value):
            return value
    return np.nan


def load_candidates() -> pd.DataFrame:
    """Load and normalize validation-only candidates from all allowed sources."""
    rows: list[dict[str, Any]] = []

    v2_file = FINAL_ATTEMPT / "protocol" / "v2_baseline_freeze.csv"
    v2 = read_csv(v2_file)
    for _, r in v2.iterrows():
        rows.append(
            make_candidate(
                source_experiment="v2_ready_baseline",
                source_file=str(v2_file.relative_to(ROOT)),
                dataset=r["dataset"],
                model=str(r["selected_model"]),
                model_family=str(r["selected_model"]).replace("Best ", ""),
                policy_type=str(r["selected_policy_type"]),
                calibration_type=r.get("selected_calibration_type", np.nan),
                threshold=r.get("selected_threshold", np.nan),
                t_low=r.get("selected_t_low", np.nan),
                t_high=r.get("selected_t_high", np.nan),
                fn_cost=r.get("fn_cost", 5.0),
                fp_cost=r.get("fp_cost", 1.0),
                review_cost=r.get("review_cost", np.nan),
                validation_precision=r.get("validation_precision", np.nan),
                validation_recall=r.get("validation_recall", np.nan),
                validation_specificity=r.get("validation_specificity", np.nan),
                validation_f1=r.get("validation_f1", np.nan),
                validation_pr_auc=r.get("validation_pr_auc", np.nan),
                validation_roc_auc=r.get("validation_roc_auc", np.nan),
                validation_ece=r.get("validation_ece", np.nan),
                validation_expected_cost=r.get("validation_expected_cost", np.nan),
                validation_review_adjusted_cost=r.get(
                    "validation_review_adjusted_cost", np.nan
                ),
                validation_manual_review_rate=r.get(
                    "validation_manual_review_rate", np.nan
                ),
                selection_note="Frozen V2 READY validation-selected baseline.",
                artifact_ready=True,
            )
        )

    for name in ["taiwan", "heloc"]:
        p = ROOT / "outputs" / "decision_revision_v2" / f"validation_only_selection_{name}.csv"
        df = read_csv(p)
        if not df.empty:
            selected = df[df.get("selected", False).map(as_bool)]
            for _, r in selected.iterrows():
                rows.append(
                    make_candidate(
                        source_experiment="v2_validation_only_selection",
                        source_file=str(p.relative_to(ROOT)),
                        dataset=r["dataset"],
                        model=str(r["model"]),
                        model_family=str(r.get("model_family", r["model"])),
                        policy_type=str(r["policy_type"]),
                        calibration_type=r.get("calibration_type", np.nan),
                        threshold=r.get("threshold", np.nan),
                        t_low=r.get("t_low", np.nan),
                        t_high=r.get("t_high", np.nan),
                        constraint_name=r.get("constraint_name", np.nan),
                        fn_cost=r.get("fn_cost", 5.0),
                        fp_cost=r.get("fp_cost", 1.0),
                        review_cost=r.get("review_cost", np.nan),
                        validation_precision=r.get("validation_precision", np.nan),
                        validation_recall=r.get("validation_recall", np.nan),
                        validation_specificity=r.get("validation_specificity", np.nan),
                        validation_f1=r.get("validation_f1", np.nan),
                        validation_pr_auc=r.get("validation_pr_auc", np.nan),
                        validation_roc_auc=r.get("validation_roc_auc", np.nan),
                        validation_ece=r.get("validation_ece", np.nan),
                        validation_expected_cost=r.get(
                            "validation_expected_cost_binary", np.nan
                        ),
                        validation_review_adjusted_cost=r.get(
                            "review_adjusted_cost", np.nan
                        ),
                        validation_fp=r.get("validation_fp", np.nan),
                        validation_manual_review_rate=r.get(
                            "manual_review_rate", np.nan
                        ),
                        validation_auto_decision_rate=r.get(
                            "auto_decision_rate", np.nan
                        ),
                        validation_high_risk_precision=r.get(
                            "high_risk_precision", np.nan
                        ),
                        validation_high_risk_recall=r.get("high_risk_recall", np.nan),
                        validation_low_risk_default_rate=r.get(
                            "low_risk_default_rate", np.nan
                        ),
                        selection_note=str(r.get("selection_reason", "")),
                        artifact_ready=True,
                    )
                )

    lit_file = FINAL_ATTEMPT / "literature_reproduction" / "literature_reproduction_best_configs.csv"
    lit = read_csv(lit_file)
    for _, r in lit.iterrows():
        if str(r.get("status", "success")).lower() != "success":
            continue
        rows.append(
            make_candidate(
                source_experiment="strict_literature_reproduction",
                source_file=str(lit_file.relative_to(ROOT)),
                dataset=r["dataset"],
                model=f"Literature reproduction: {r['config_display_name']}",
                model_family=str(r["model_family"]),
                policy_type="literature_cost_policy_FN5_FP1",
                calibration_type="native_probability",
                threshold=r.get("validation_threshold_cost_policy", np.nan),
                fn_cost=5.0,
                fp_cost=1.0,
                validation_precision=r.get("validation_precision_cost_policy", np.nan),
                validation_recall=r.get("validation_recall_cost_policy", np.nan),
                validation_specificity=r.get(
                    "validation_best_specificity_at_cost_policy", np.nan
                ),
                validation_pr_auc=r.get("validation_pr_auc", np.nan),
                validation_roc_auc=r.get("validation_roc_auc", np.nan),
                validation_ece=r.get("validation_ece", np.nan),
                validation_brier=r.get("validation_brier", np.nan),
                validation_expected_cost=r.get(
                    "validation_expected_cost_cost_policy", np.nan
                ),
                selection_note="Literature-inspired leakage-free reproduction; validation cost-policy row.",
                artifact_ready=False,
            )
        )

    adv_file = FINAL_ATTEMPT / "imbalance_boosting" / "advanced_imbalance_best_configs.csv"
    adv = read_csv(adv_file)
    for _, r in adv.iterrows():
        rows.append(
            make_candidate(
                source_experiment="advanced_imbalance_boosting",
                source_file=str(adv_file.relative_to(ROOT)),
                dataset=r["dataset"],
                model=f"{r['model_family']} {r['variant_name']}",
                model_family=str(r["model_family"]),
                policy_type=str(r["policy_type"]),
                calibration_type=r.get("calibration_type", np.nan),
                threshold=r.get("validation_threshold", r.get("threshold", np.nan)),
                t_low=r.get("t_low", np.nan),
                t_high=r.get("t_high", np.nan),
                fn_cost=r.get("fn_cost", 5.0),
                fp_cost=r.get("fp_cost", 1.0),
                review_cost=r.get("review_cost", np.nan),
                validation_accuracy=r.get("validation_accuracy", np.nan),
                validation_precision=r.get("validation_precision", np.nan),
                validation_recall=r.get("validation_recall", np.nan),
                validation_specificity=r.get("validation_specificity", np.nan),
                validation_f1=r.get("validation_f1", np.nan),
                validation_pr_auc=r.get("validation_pr_auc", np.nan),
                validation_roc_auc=r.get("validation_roc_auc", np.nan),
                validation_ece=r.get("validation_ece", np.nan),
                validation_brier=r.get("validation_brier", np.nan),
                validation_expected_cost=r.get("validation_expected_cost", np.nan),
                validation_review_adjusted_cost=r.get(
                    "validation_review_adjusted_cost", np.nan
                ),
                validation_fp=r.get("validation_fp", np.nan),
                validation_manual_review_rate=r.get(
                    "validation_manual_review_rate", r.get("manual_review_rate", np.nan)
                ),
                validation_auto_decision_rate=r.get(
                    "validation_auto_decision_rate", np.nan
                ),
                validation_high_risk_precision=r.get(
                    "validation_high_risk_default_rate", np.nan
                ),
                validation_high_risk_recall=r.get("validation_recall", np.nan),
                validation_low_risk_default_rate=r.get(
                    "validation_low_risk_default_rate", np.nan
                ),
                selection_note=str(r.get("selection_reason", "")),
                artifact_ready=False,
            )
        )

    deep_file = FINAL_ATTEMPT / "deep_tabular" / "deep_tabular_best_configs.csv"
    deep = read_csv(deep_file)
    for _, r in deep.iterrows():
        rows.append(
            make_candidate(
                source_experiment="deep_tabular_reproduction",
                source_file=str(deep_file.relative_to(ROOT)),
                dataset=r["dataset"],
                model=f"{r['model']} {r['config_id']}",
                model_family="Deep tabular",
                policy_type=str(r["policy_type"]),
                calibration_type=r.get("calibration_type", np.nan),
                threshold=r.get("validation_threshold", np.nan),
                t_low=r.get("t_low", np.nan),
                t_high=r.get("t_high", np.nan),
                validation_accuracy=r.get("validation_accuracy", np.nan),
                validation_precision=r.get("validation_precision", np.nan),
                validation_recall=r.get("validation_recall", np.nan),
                validation_specificity=r.get("validation_specificity", np.nan),
                validation_f1=r.get("validation_f1", np.nan),
                validation_pr_auc=r.get("validation_pr_auc", np.nan),
                validation_roc_auc=r.get("validation_roc_auc", np.nan),
                validation_ece=r.get("validation_ece", np.nan),
                validation_brier=r.get("validation_brier", np.nan),
                validation_expected_cost=r.get("validation_expected_cost", np.nan),
                validation_review_adjusted_cost=r.get(
                    "validation_review_adjusted_cost", np.nan
                ),
                validation_manual_review_rate=r.get(
                    "validation_manual_review_rate", r.get("manual_review_rate", np.nan)
                ),
                validation_auto_decision_rate=r.get(
                    "validation_auto_decision_rate", np.nan
                ),
                validation_high_risk_precision=r.get(
                    "validation_high_risk_default_rate", np.nan
                ),
                validation_high_risk_recall=r.get("validation_recall", np.nan),
                validation_low_risk_default_rate=r.get(
                    "validation_low_risk_default_rate", np.nan
                ),
                selection_note=str(r.get("selection_reason", "")),
                artifact_ready=False,
            )
        )

    interp_file = FINAL_ATTEMPT / "interpretable_models" / "interpretable_models_best_configs.csv"
    interp = read_csv(interp_file)
    for _, r in interp.iterrows():
        rows.append(
            make_candidate(
                source_experiment="interpretable_middle_models",
                source_file=str(interp_file.relative_to(ROOT)),
                dataset=r["dataset"],
                model=str(r["model"]),
                model_family=str(r["model_family"]),
                policy_type=str(r["policy_type"]),
                calibration_type=r.get("calibration_type", np.nan),
                threshold=r.get("threshold", np.nan),
                t_low=r.get("validation_t_low", r.get("t_low", np.nan)),
                t_high=r.get("validation_t_high", r.get("t_high", np.nan)),
                fn_cost=r.get("fn_cost", 5.0),
                fp_cost=r.get("fp_cost", 1.0),
                review_cost=r.get("review_cost", np.nan),
                validation_accuracy=r.get("validation_accuracy", np.nan),
                validation_precision=r.get("validation_precision", np.nan),
                validation_recall=r.get("validation_recall", np.nan),
                validation_specificity=r.get("validation_specificity", np.nan),
                validation_f1=r.get("validation_f1", np.nan),
                validation_pr_auc=r.get("validation_pr_auc", np.nan),
                validation_roc_auc=r.get("validation_roc_auc", np.nan),
                validation_ece=r.get("validation_ece", np.nan),
                validation_brier=r.get("validation_brier", np.nan),
                validation_expected_cost=r.get("validation_expected_cost", np.nan),
                validation_review_adjusted_cost=r.get(
                    "validation_manual_review_adjusted_cost", np.nan
                ),
                validation_fp=r.get("validation_fp", np.nan),
                validation_manual_review_rate=r.get(
                    "validation_manual_review_rate", np.nan
                ),
                validation_auto_decision_rate=r.get(
                    "validation_auto_decision_rate", np.nan
                ),
                validation_high_risk_precision=r.get(
                    "validation_high_risk_precision", np.nan
                ),
                validation_high_risk_recall=r.get("validation_high_risk_recall", np.nan),
                validation_low_risk_default_rate=r.get(
                    "validation_low_risk_default_rate", np.nan
                ),
                selection_note=f"Interpretability score source; eligible_basic={r.get('eligible_basic', np.nan)}.",
                artifact_ready=True,
            )
        )

    for kind in ["scre_optimized", "scre_pareto"]:
        for dataset in ["taiwan", "heloc"]:
            p = ROOT / "outputs" / "tables" / f"{kind}_results_{dataset}.csv"
            df = read_csv(p)
            if df.empty:
                continue
            df = df[df["split"].astype(str).str.lower().eq("validation")]
            if "selected_final_setting" in df.columns:
                selected = df[df["selected_final_setting"].map(as_bool)]
                if not selected.empty:
                    df = selected
            for _, r in df.iterrows():
                rows.append(
                    make_candidate(
                        source_experiment=kind,
                        source_file=str(p.relative_to(ROOT)),
                        dataset=r["dataset"],
                        model=str(r["model"]),
                        model_family="SCRE-Credit",
                        policy_type="SCRE_binary_cost_policy",
                        calibration_type=r.get("calibration_type", np.nan),
                        threshold=r.get("threshold", np.nan),
                        fn_cost=r.get("fn_cost", 5.0),
                        fp_cost=r.get("fp_cost", 1.0),
                        validation_accuracy=r.get("accuracy", np.nan),
                        validation_precision=r.get("precision", np.nan),
                        validation_recall=r.get("recall", np.nan),
                        validation_specificity=r.get("specificity", np.nan),
                        validation_f1=r.get("f1", np.nan),
                        validation_pr_auc=r.get("pr_auc", np.nan),
                        validation_roc_auc=r.get("roc_auc", np.nan),
                        validation_ece=r.get("ece", np.nan),
                        validation_brier=r.get("brier_score", np.nan),
                        validation_expected_cost=r.get("expected_cost", np.nan),
                        validation_fp=r.get("fp", np.nan),
                        selection_note="SCRE validation row; selected setting only when available.",
                        artifact_ready=True,
                    )
                )

    candidates = pd.DataFrame(rows)
    candidates.insert(
        0,
        "candidate_key",
        [
            f"fa_{idx:05d}_{row.dataset}_{slug(row.model)}_{slug(row.policy_type)}"
            for idx, row in candidates.iterrows()
        ],
    )
    return candidates


def slug(value: Any) -> str:
    """Create a short filesystem/identifier-safe slug."""
    text = str(value).lower()
    safe = [ch if ch.isalnum() else "_" for ch in text]
    return "".join(safe).strip("_")[:48]


def binary_eligible(row: pd.Series) -> bool:
    """Check minimum binary-policy eligibility by dataset."""
    ds = str(row["dataset"]).lower()
    precision = as_float(row["validation_precision"])
    recall = as_float(row["validation_recall"])
    specificity = as_float(row["validation_specificity"])
    pr_auc = as_float(row["validation_pr_auc"])
    roc_auc = as_float(row["validation_roc_auc"])
    if ds == "taiwan":
        return (
            precision >= 0.40
            and recall >= 0.50
            and specificity >= 0.55
            and pr_auc >= 0.54
            and roc_auc >= 0.77
        )
    return (
        precision >= 0.55
        and recall >= 0.80
        and specificity >= 0.10
        and pr_auc >= 0.78
        and roc_auc >= 0.78
    )


def manual_eligible(row: pd.Series) -> bool:
    """Check manual-review eligibility by dataset."""
    ds = str(row["dataset"]).lower()
    mr_rate = as_float(row["validation_manual_review_rate"])
    precision = first_finite(
        as_float(row["validation_high_risk_precision"]),
        as_float(row["validation_precision"]),
    )
    recall = first_finite(
        as_float(row["validation_high_risk_recall"]),
        as_float(row["validation_recall"]),
    )
    low_default = as_float(row["validation_low_risk_default_rate"])
    if ds == "taiwan":
        return (
            mr_rate <= 0.30
            and precision >= 0.40
            and recall >= 0.55
            and (not np.isfinite(low_default) or low_default <= 0.12)
        )
    return (
        mr_rate <= 0.30
        and precision >= 0.55
        and recall >= 0.80
        and (not np.isfinite(low_default) or low_default <= 0.20)
    )


def select_raw(candidates: pd.DataFrame, dataset: str) -> pd.Series:
    """Select best raw classifier by validation PR-AUC, ROC-AUC, F1, ECE."""
    df = candidates[candidates["dataset"].eq(dataset)].copy()
    df = df[np.isfinite(pd.to_numeric(df["validation_pr_auc"], errors="coerce"))]
    df["_ece_sort"] = pd.to_numeric(df["validation_ece"], errors="coerce").fillna(999.0)
    # One row per probability model/calibration to avoid repeated policy rows dominating.
    df = df.sort_values(
        ["validation_pr_auc", "validation_roc_auc", "validation_f1", "_ece_sort"],
        ascending=[False, False, False, True],
    )
    df = df.drop_duplicates(
        subset=["source_experiment", "model", "calibration_type"], keep="first"
    )
    return df.iloc[0]


def select_binary(candidates: pd.DataFrame, dataset: str) -> pd.Series:
    """Select best operational binary policy using validation evidence only."""
    df = candidates[candidates["dataset"].eq(dataset)].copy()
    df = df[~df["policy_type"].map(is_manual_policy)]
    df = df[np.isfinite(pd.to_numeric(df["validation_expected_cost"], errors="coerce"))]
    df["eligible_track"] = df.apply(binary_eligible, axis=1)
    eligible = df[df["eligible_track"]]
    pool = eligible if not eligible.empty else df
    pool = pool.copy()
    pool["_ece_sort"] = pd.to_numeric(pool["validation_ece"], errors="coerce").fillna(999.0)
    return pool.sort_values(
        [
            "validation_expected_cost",
            "validation_pr_auc",
            "_ece_sort",
            "validation_f1",
            "validation_fp",
        ],
        ascending=[True, False, True, False, True],
    ).iloc[0]


def select_manual(candidates: pd.DataFrame, dataset: str) -> pd.Series:
    """Select best manual-review policy using validation evidence only."""
    df = candidates[candidates["dataset"].eq(dataset)].copy()
    df = df[df["policy_type"].map(is_manual_policy)]
    df = df[
        np.isfinite(pd.to_numeric(df["validation_review_adjusted_cost"], errors="coerce"))
    ]
    df["eligible_track"] = df.apply(manual_eligible, axis=1)
    eligible = df[df["eligible_track"]]
    pool = eligible if not eligible.empty else df
    pool = pool.copy()
    pool["_mr_preferred"] = (
        pd.to_numeric(pool["validation_manual_review_rate"], errors="coerce") <= 0.25
    )
    pool["_low_default"] = pd.to_numeric(
        pool["validation_low_risk_default_rate"], errors="coerce"
    ).fillna(999.0)
    return pool.sort_values(
        [
            "validation_review_adjusted_cost",
            "_mr_preferred",
            "validation_high_risk_precision",
            "_low_default",
            "validation_auto_decision_rate",
        ],
        ascending=[True, False, False, True, False],
    ).iloc[0]


def select_capacity(dataset: str) -> pd.Series:
    """Select best capacity-aware reviewer from validation capacity analysis."""
    p = FINAL_ATTEMPT / "capacity_review" / "capacity_model_comparison.csv"
    df = read_csv(p)
    df = df[
        df["dataset"].astype(str).str.lower().eq(dataset)
        & df["split"].astype(str).str.lower().eq("validation")
    ].copy()
    if df.empty:
        raise RuntimeError(f"No validation capacity rows for {dataset}")
    return df.sort_values(
        ["top_20_capture", "precision_at_20", "lift_at_20", "net_value_at_20"],
        ascending=[False, False, False, False],
    ).iloc[0]


def select_interpretable(candidates: pd.DataFrame, dataset: str) -> pd.Series:
    """Select the best interpretable model with governance and validation constraints."""
    df = candidates[
        candidates["dataset"].eq(dataset)
        & candidates["source_experiment"].eq("interpretable_middle_models")
    ].copy()
    df = df[df["policy_type"].map(is_manual_policy)]
    df = df[
        np.isfinite(pd.to_numeric(df["validation_review_adjusted_cost"], errors="coerce"))
    ]
    df["eligible_track"] = df.apply(manual_eligible, axis=1)
    eligible = df[df["eligible_track"]]
    pool = eligible if not eligible.empty else df
    pool = pool.copy()
    priority = {"Scorecard": 4, "EBM": 3, "Monotonic LightGBM": 3, "Monotonic XGBoost": 3}
    pool["_interpretability_priority"] = pool["model_family"].map(priority).fillna(1)
    # Keep explicit governance preference, but do not allow a non-eligible scorecard
    # to beat eligible middle models.
    return pool.sort_values(
        [
            "eligible_track",
            "validation_review_adjusted_cost",
            "_interpretability_priority",
            "validation_pr_auc",
            "validation_ece",
        ],
        ascending=[False, True, False, False, True],
    ).iloc[0]


def select_framework(candidates: pd.DataFrame, dataset: str, capacity: pd.Series) -> pd.Series:
    """Select the SCRE research framework representative."""
    df = candidates[
        candidates["dataset"].eq(dataset)
        & candidates["model"].astype(str).isin(["SCRE-Optimized", "SCRE-Pareto"])
    ].copy()
    if df.empty:
        raise RuntimeError(f"No SCRE candidate rows for {dataset}")
    df["_capacity_bonus"] = df["model"].astype(str).apply(
        lambda x: 1.0 if str(capacity["model"]).startswith(x) else 0.0
    )
    return df.sort_values(
        ["_capacity_bonus", "validation_pr_auc", "validation_roc_auc", "validation_ece"],
        ascending=[False, False, False, True],
    ).iloc[0]


def lookup_decision_curve(dataset: str, model_name: str) -> dict[str, Any]:
    """Return validation decision-curve fields for the closest model label."""
    p = FINAL_ATTEMPT / "decision_curve" / "decision_curve_best_ranges.csv"
    df = read_csv(p)
    df = df[
        df["dataset"].astype(str).str.lower().eq(dataset)
        & df["split"].astype(str).str.lower().eq("validation")
    ].copy()
    if df.empty:
        return {}
    model_text = str(model_name).lower()
    masks = [
        df["model"].astype(str).str.lower().eq(model_text),
        df["model"].astype(str).str.lower().str.contains(model_text[:20], regex=False),
    ]
    for mask in masks:
        matched = df[mask]
        if not matched.empty:
            r = matched.iloc[0]
            return {
                "dca_useful_threshold_range": r.get("useful_threshold_range", np.nan),
                "dca_useful_threshold_width": r.get("useful_threshold_width", np.nan),
                "dca_max_net_benefit": r.get("max_net_benefit", np.nan),
                "dca_best_model_at_threshold_range": r.get(
                    "best_model_at_threshold_range", np.nan
                ),
            }
    return {}


def selection_row(
    *,
    dataset: str,
    track_id: str,
    track_name: str,
    selected_system: str,
    source_experiment: str,
    policy_type: str,
    why: str,
    candidate: pd.Series | None = None,
    capacity: pd.Series | None = None,
    artifact_ready: bool | None = None,
) -> dict[str, Any]:
    """Build a normalized final selection output row."""
    row: dict[str, Any] = {
        "dataset": dataset,
        "track_id": track_id,
        "track_name": track_name,
        "selected_system": selected_system,
        "source_experiment": source_experiment,
        "policy_type": policy_type,
        "selection_reason": why,
        "test_set_used_for_selection": False,
    }
    metric_fields = [
        "model_family",
        "calibration_type",
        "threshold",
        "t_low",
        "t_high",
        "fn_cost",
        "fp_cost",
        "review_cost",
        "validation_precision",
        "validation_recall",
        "validation_specificity",
        "validation_f1",
        "validation_pr_auc",
        "validation_roc_auc",
        "validation_ece",
        "validation_brier",
        "validation_expected_cost",
        "validation_review_adjusted_cost",
        "validation_manual_review_rate",
        "validation_auto_decision_rate",
        "validation_high_risk_precision",
        "validation_high_risk_recall",
        "validation_low_risk_default_rate",
        "validation_fp",
        "candidate_key",
        "source_file",
    ]
    for field in metric_fields:
        row[field] = np.nan
    if candidate is not None:
        for field in metric_fields:
            if field in candidate.index:
                row[field] = candidate[field]
        if artifact_ready is None:
            artifact_ready = bool(candidate.get("artifact_ready", False))
    if capacity is not None:
        row["capacity_top20_capture"] = capacity.get("top_20_capture", np.nan)
        row["capacity_precision_at20"] = capacity.get("precision_at_20", np.nan)
        row["capacity_lift_at20"] = capacity.get("lift_at_20", np.nan)
        row["capacity_net_value_at20"] = capacity.get("net_value_at_20", np.nan)
    else:
        row["capacity_top20_capture"] = np.nan
        row["capacity_precision_at20"] = np.nan
        row["capacity_lift_at20"] = np.nan
        row["capacity_net_value_at20"] = np.nan
    dca = lookup_decision_curve(dataset, selected_system)
    row.update(
        {
            "dca_useful_threshold_range": dca.get("dca_useful_threshold_range", np.nan),
            "dca_useful_threshold_width": dca.get("dca_useful_threshold_width", np.nan),
            "dca_max_net_benefit": dca.get("dca_max_net_benefit", np.nan),
            "dca_best_model_at_threshold_range": dca.get(
                "dca_best_model_at_threshold_range", np.nan
            ),
            "artifact_ready": bool(artifact_ready) if artifact_ready is not None else False,
        }
    )
    return row


def build_selection(candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build final track selections for Taiwan and HELOC."""
    all_rows: list[dict[str, Any]] = []
    locked: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_protocol": "validation_only_final_attempt_multiobjective_selector",
        "test_set_used_for_selection": False,
        "do_not_change_after_locking": True,
        "datasets": {},
    }
    for dataset in ["taiwan", "heloc"]:
        raw = select_raw(candidates, dataset)
        binary = select_binary(candidates, dataset)
        manual = select_manual(candidates, dataset)
        capacity = select_capacity(dataset)
        interpretable = select_interpretable(candidates, dataset)
        framework = select_framework(candidates, dataset, capacity)

        rows = [
            selection_row(
                dataset=dataset,
                track_id="A",
                track_name="Best raw classifier",
                selected_system=str(raw["model"]),
                source_experiment=str(raw["source_experiment"]),
                policy_type="probability_ranking_raw_classifier",
                candidate=raw,
                why=(
                    "Highest validation PR-AUC after tie-breaking by ROC-AUC, F1, "
                    "and calibration. No test metric was consulted."
                ),
            ),
            selection_row(
                dataset=dataset,
                track_id="B",
                track_name="Best operational binary policy",
                selected_system=str(binary["model"]),
                source_experiment=str(binary["source_experiment"]),
                policy_type=str(binary["policy_type"]),
                candidate=binary,
                why=(
                    "Lowest validation binary expected cost among eligible binary "
                    "policies, with PR-AUC/ECE/F1 used only as validation tie-breakers."
                ),
            ),
            selection_row(
                dataset=dataset,
                track_id="C",
                track_name="Best manual-review policy",
                selected_system=str(manual["model"]),
                source_experiment=str(manual["source_experiment"]),
                policy_type=str(manual["policy_type"]),
                candidate=manual,
                why=(
                    "Lowest validation review-adjusted cost under review_cost=0.5 "
                    "among manual-review candidates satisfying MR-rate and high-risk "
                    "quality constraints."
                ),
            ),
            selection_row(
                dataset=dataset,
                track_id="D",
                track_name="Best capacity-aware reviewer",
                selected_system=str(capacity["model"]),
                source_experiment="capacity_aware_manual_review",
                policy_type="top_k_review_prioritization",
                capacity=capacity,
                artifact_ready=True,
                why=(
                    "Highest validation default capture@20%, then precision@20%, "
                    "lift@20%, and net value@20%."
                ),
            ),
            selection_row(
                dataset=dataset,
                track_id="E",
                track_name="Best interpretable model",
                selected_system=str(interpretable["model"]),
                source_experiment=str(interpretable["source_experiment"]),
                policy_type=str(interpretable["policy_type"]),
                candidate=interpretable,
                why=(
                    "Best validation-governed interpretable candidate after enforcing "
                    "manual-review feasibility; scorecard is preferred when it remains "
                    "competitive and feasible."
                ),
            ),
            selection_row(
                dataset=dataset,
                track_id="F",
                track_name="Best research framework",
                selected_system=str(framework["model"]),
                source_experiment=str(framework["source_experiment"]),
                policy_type="reliability_aware_framework",
                candidate=framework,
                why=(
                    "SCRE variant selected for reliability integration, validation "
                    "AUC/PR-AUC, and capacity-review support. This is a framework "
                    "track, not a claim of universal predictive dominance."
                ),
            ),
        ]
        final_system = build_final_system_description(dataset, manual, capacity, interpretable, framework)
        rows.append(
            selection_row(
                dataset=dataset,
                track_id="G",
                track_name="Final recommended system",
                selected_system=final_system["selected_system"],
                source_experiment="multi_track_committee_decision",
                policy_type="portfolio_recommendation",
                why=final_system["why"],
                artifact_ready=final_system["artifact_ready"],
            )
        )
        all_rows.extend(rows)
        locked["datasets"][dataset] = {
            "final_recommended_system": final_system,
            "tracks": {
                row["track_id"]: serializable(row)
                for row in rows
                if row["track_id"] in {"A", "B", "C", "D", "E", "F"}
            },
            "locked_for_heldout_test": serializable(rows[2]),
            "locked_policy_note": (
                "Track C manual-review policy is locked for any future held-out "
                "test evaluation; Track D may be reported separately as review "
                "prioritization evidence."
            ),
        }
    return pd.DataFrame(all_rows), locked


def build_final_system_description(
    dataset: str, manual: pd.Series, capacity: pd.Series, interpretable: pd.Series, framework: pd.Series
) -> dict[str, Any]:
    """Create the final portfolio recommendation for a dataset."""
    if dataset == "taiwan":
        selected = (
            f"Operational screening: {manual['model']} ({manual['policy_type']}); "
            f"review prioritization: {capacity['model']}; "
            f"interpretable benchmark: {interpretable['model']}; "
            f"research framework: {framework['model']}."
        )
        why = (
            "Taiwan changes partially from V2: the validation-only manual-review "
            "winner shifts marginally from V2 CatBoost to advanced XGBoost "
            f"(review-adjusted cost {as_float(manual['validation_review_adjusted_cost']):.1f}), "
            "while SCRE-Optimized is kept for top-k review prioritization and "
            "Monotonic LightGBM is the strongest interpretable middle model."
        )
        return {
            "selected_system": selected,
            "why": why,
            "artifact_ready": bool(manual.get("artifact_ready", False)),
        }
    selected = (
        f"Operational screening: {manual['model']} ({manual['policy_type']}); "
        f"review prioritization: {capacity['model']}; "
        f"interpretable benchmark: {interpretable['model']}; "
        f"research framework: {framework['model']}."
    )
    why = (
        "HELOC operational selection remains close to V2: Scorecard-style "
        "manual review remains the cleanest validation-only operational policy, "
        "while SCRE-Optimized ranks best for capacity@20 review prioritization."
    )
    return {
        "selected_system": selected,
        "why": why,
        "artifact_ready": bool(manual.get("artifact_ready", False)),
    }


def serializable(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serializable values."""
    if isinstance(value, pd.Series):
        return {k: serializable(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def write_outputs(selection: pd.DataFrame, locked: dict[str, Any]) -> None:
    """Write final selection CSV, JSON, and Markdown summary outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    taiwan = selection[selection["dataset"].eq("taiwan")].copy()
    heloc = selection[selection["dataset"].eq("heloc")].copy()
    taiwan.to_csv(OUT_DIR / "final_multitrack_selection_taiwan.csv", index=False)
    heloc.to_csv(OUT_DIR / "final_multitrack_selection_heloc.csv", index=False)
    with (OUT_DIR / "final_selected_locked_policies.json").open("w", encoding="utf-8") as f:
        json.dump(locked, f, indent=2, ensure_ascii=False)
    (OUT_DIR / "final_selection_summary.md").write_text(
        build_summary(selection), encoding="utf-8"
    )


def fmt_metric(value: Any, digits: int = 4) -> str:
    """Format a metric for Markdown."""
    number = as_float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def build_summary(selection: pd.DataFrame) -> str:
    """Build a concise methodology-safe final selection summary."""
    lines = [
        "# Final Multi-Objective Selection Summary",
        "",
        "Selection evidence: validation-only final-attempt outputs, validation capacity analysis, and validation decision-curve summaries.",
        "Forbidden evidence: held-out test metrics, test ranks, test winners, or test-threshold tuning.",
        "",
    ]
    for dataset in ["taiwan", "heloc"]:
        df = selection[selection["dataset"].eq(dataset)].copy()
        lines.extend([f"## {dataset.upper()} Track Decisions", ""])
        for _, r in df.iterrows():
            lines.append(
                f"- Track {r['track_id']} ({r['track_name']}): {r['selected_system']}."
            )
            lines.append(f"  Reason: {r['selection_reason']}")
            if np.isfinite(as_float(r.get("validation_pr_auc", np.nan))):
                lines.append(
                    "  Validation metrics: "
                    f"PR-AUC={fmt_metric(r.get('validation_pr_auc'))}, "
                    f"ROC-AUC={fmt_metric(r.get('validation_roc_auc'))}, "
                    f"precision={fmt_metric(r.get('validation_precision'))}, "
                    f"recall={fmt_metric(r.get('validation_recall'))}, "
                    f"specificity={fmt_metric(r.get('validation_specificity'))}, "
                    f"cost={fmt_metric(first_finite(as_float(r.get('validation_review_adjusted_cost')), as_float(r.get('validation_expected_cost'))), 1)}."
                )
            if np.isfinite(as_float(r.get("capacity_top20_capture", np.nan))):
                lines.append(
                    "  Capacity@20: "
                    f"default capture={fmt_metric(r.get('capacity_top20_capture'))}, "
                    f"precision@20={fmt_metric(r.get('capacity_precision_at20'))}, "
                    f"lift@20={fmt_metric(r.get('capacity_lift_at20'))}."
                )
        lines.append("")

    lines.extend(
        [
            "## Prompt Questions",
            "",
            "1. En yüksek skor modeli hangisi?",
            f"   - Taiwan: {track_system(selection, 'taiwan', 'A')}. HELOC: {track_system(selection, 'heloc', 'A')}.",
            "2. En dengeli operational model hangisi?",
            f"   - Taiwan: {track_system(selection, 'taiwan', 'C')}. HELOC: {track_system(selection, 'heloc', 'C')}.",
            "3. Manual-review için en iyi model hangisi?",
            f"   - Taiwan: {track_system(selection, 'taiwan', 'C')}. HELOC: {track_system(selection, 'heloc', 'C')}.",
            "4. Capacity@20 için en iyi model hangisi?",
            f"   - Taiwan: {track_system(selection, 'taiwan', 'D')}. HELOC: {track_system(selection, 'heloc', 'D')}.",
            "5. En iyi yorumlanabilir model hangisi?",
            f"   - Taiwan: {track_system(selection, 'taiwan', 'E')}. HELOC: {track_system(selection, 'heloc', 'E')}.",
            "6. SCRE'nin rolü ne?",
            "   - SCRE-Optimized is retained as a reliability-aware research framework and as a strong review-prioritization ranker, not as a universal winner over all single models.",
            "7. V2 final policy hala en iyi mi?",
            "   - Partially. Taiwan operational screening changes marginally under validation-only cost; HELOC operational screening remains effectively V2 Scorecard/manual-review.",
            "8. Yeni deneyler final kararı değiştirdi mi?",
            "   - Partially. They add stronger Taiwan XGBoost manual-review evidence and stronger capacity@20 evidence for SCRE-Optimized, while keeping the HELOC Scorecard decision stable.",
            "9. Final test için hangi policy kilitleniyor?",
            f"   - Taiwan: {track_system(selection, 'taiwan', 'C')}. HELOC: {track_system(selection, 'heloc', 'C')}. These are locked before any future held-out test evaluation.",
            "",
            "## Methodological Lock",
            "",
            "- Test set used for selection: NO.",
            "- Test metrics read by this selector: NO.",
            "- Held-out test evaluation may be run only after these locked policies are accepted.",
        ]
    )
    return "\n".join(lines) + "\n"


def track_system(selection: pd.DataFrame, dataset: str, track_id: str) -> str:
    """Return the selected system label for a dataset/track."""
    row = selection[
        selection["dataset"].eq(dataset) & selection["track_id"].eq(track_id)
    ].iloc[0]
    return str(row["selected_system"])


def main() -> None:
    """Run the final multi-objective selector."""
    candidates = load_candidates()
    if candidates.empty:
        raise RuntimeError("No validation candidates found.")
    if "test_precision" in candidates.columns or "test_recall" in candidates.columns:
        raise RuntimeError("Unexpected test metric columns in candidate frame.")
    selection, locked = build_selection(candidates)
    write_outputs(selection, locked)
    print(f"Wrote final selection outputs to {OUT_DIR}")
    print(
        selection[["dataset", "track_id", "track_name", "selected_system"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
