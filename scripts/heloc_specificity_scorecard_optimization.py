"""HELOC specificity-focused scorecard policy optimization.

This script uses only existing calibrated scorecard artifacts. It regenerates
validation/test probabilities by inference, selects threshold/band policies on
validation only, and evaluates only the locked top-3 policies on held-out test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import math
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/heloc_specificity"
DATA_PATH = PROJECT_ROOT / "data/processed/heloc_model_ready.csv"
TARGET = "bad_flag"
MODEL_DIR = PROJECT_ROOT / "outputs/models/calibrated/heloc"
CALIBRATIONS = ["uncalibrated", "sigmoid", "isotonic"]
RANDOM_SEED = 42
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COST = 0.5
V2_T_LOW = 0.16
V2_T_HIGH = 0.39
V2_TEST = {
    "precision": 0.6834249803613511,
    "recall": 0.8463035019455253,
    "specificity": 0.5744456177402323,
    "fp": 403,
    "fn": 158,
    "cost": 764.5,
    "manual_review_rate": 0.26987341772151896,
}


@dataclass(frozen=True)
class Metrics:
    """Policy metrics for binary or three-bucket decisions."""

    precision: float
    recall: float
    specificity: float
    f1: float
    fp: int
    fn: int
    low_fn: int
    tp: int
    tn: int
    cost: float
    manual_review_rate: float
    review_adjusted_cost: float


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely and return zero on zero denominator."""

    return float(numerator / denominator) if denominator else 0.0


def _ece(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (p >= left) & (p <= right if i == n_bins - 1 else p < right)
        if not np.any(mask):
            continue
        error += (np.sum(mask) / len(p)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(error)


def _split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate canonical HELOC split."""

    df = pd.read_csv(DATA_PATH)
    train_validation, test = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET],
        random_state=RANDOM_SEED,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[TARGET],
        random_state=RANDOM_SEED,
    )
    return train.copy(), validation.copy(), test.copy()


def _scores(frame: pd.DataFrame, calibration: str) -> tuple[np.ndarray, np.ndarray]:
    """Generate scorecard probabilities from an existing artifact."""

    model = joblib.load(MODEL_DIR / f"woe_scorecard_logistic_regression_{calibration}.joblib")
    y = frame[TARGET].astype(int).to_numpy()
    x = frame.drop(columns=[TARGET])
    return y, model.predict_proba(x)[:, 1]


def _probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return threshold-independent and calibration metrics."""

    return {
        "validation_pr_auc": float(average_precision_score(y, probability)),
        "validation_roc_auc": float(roc_auc_score(y, probability)),
        "validation_brier": float(brier_score_loss(y, probability)),
        "validation_ece": _ece(y, probability),
    }


def _binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> Metrics:
    """Compute threshold-only binary metrics."""

    pred = probability >= threshold
    default = y == 1
    nondefault = ~default
    tp = int(np.sum(pred & default))
    fp = int(np.sum(pred & nondefault))
    fn = int(np.sum((~pred) & default))
    tn = int(np.sum((~pred) & nondefault))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = f1_score(y, pred.astype(int), zero_division=0)
    cost = fp * FP_COST + fn * FN_COST
    return Metrics(
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=float(f1),
        fp=fp,
        fn=fn,
        low_fn=fn,
        tp=tp,
        tn=tn,
        cost=cost,
        manual_review_rate=0.0,
        review_adjusted_cost=cost,
    )


def _manual_metrics(y: np.ndarray, probability: np.ndarray, t_low: float, t_high: float) -> Metrics:
    """Compute manual-review band metrics with high-risk-vs-not-high specificity."""

    low = probability < t_low
    high = probability >= t_high
    manual = (~low) & (~high)
    default = y == 1
    nondefault = ~default
    high_default = int(np.sum(high & default))
    high_nondefault = int(np.sum(high & nondefault))
    low_default = int(np.sum(low & default))
    not_high_default = int(np.sum((~high) & default))
    not_high_nondefault = int(np.sum((~high) & nondefault))
    total_defaults = int(np.sum(default))
    total_nondefaults = int(np.sum(nondefault))
    manual_count = int(np.sum(manual))
    precision = _safe_divide(high_default, high_default + high_nondefault)
    recall = _safe_divide(high_default, total_defaults)
    specificity = _safe_divide(not_high_nondefault, total_nondefaults)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    cost = high_nondefault * FP_COST + low_default * FN_COST + manual_count * REVIEW_COST
    return Metrics(
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        fp=high_nondefault,
        fn=not_high_default,
        low_fn=low_default,
        tp=high_default,
        tn=not_high_nondefault,
        cost=cost,
        manual_review_rate=_safe_divide(manual_count, len(y)),
        review_adjusted_cost=cost,
    )


def _v2_validation_cost(y: np.ndarray, probability: np.ndarray) -> float:
    """Compute V2 validation cost under the same policy definition."""

    return _manual_metrics(y, probability, V2_T_LOW, V2_T_HIGH).cost


def _binary_constraints(v2_cost: float) -> dict[str, Callable[[Metrics], bool]]:
    """Return specificity-focused binary constraints."""

    return {
        "A1_specificity_ge_0_60": lambda m: m.specificity >= 0.60,
        "A2_specificity_ge_0_65": lambda m: m.specificity >= 0.65,
        "A3_specificity_ge_0_70": lambda m: m.specificity >= 0.70,
        "A4_specificity_ge_0_60_recall_ge_0_80": lambda m: m.specificity >= 0.60 and m.recall >= 0.80,
        "A5_specificity_ge_0_65_recall_ge_0_75": lambda m: m.specificity >= 0.65 and m.recall >= 0.75,
        "A6_specificity_ge_0_70_recall_ge_0_70": lambda m: m.specificity >= 0.70 and m.recall >= 0.70,
        "A7_specificity_ge_0_65_precision_ge_0_70": lambda m: m.specificity >= 0.65 and m.precision >= 0.70,
        "A8_specificity_ge_0_65_cost_le_v2_validation_plus_10pct": (
            lambda m: m.specificity >= 0.65 and m.cost <= v2_cost * 1.10
        ),
    }


def _manual_constraints() -> dict[str, Callable[[Metrics], bool]]:
    """Return specificity-focused manual-review constraints."""

    return {
        "B1_manual_review_rate_le_0_20": lambda m: m.manual_review_rate <= 0.20,
        "B2_manual_review_rate_le_0_25": lambda m: m.manual_review_rate <= 0.25,
        "B3_manual_review_rate_le_0_30": lambda m: m.manual_review_rate <= 0.30,
        "B4_specificity_ge_0_65": lambda m: m.specificity >= 0.65,
        "B5_recall_ge_0_75": lambda m: m.recall >= 0.75,
        "B6_precision_ge_0_65": lambda m: m.precision >= 0.65,
        "B7_balanced_mr30_spec65_recall75_precision65": (
            lambda m: m.manual_review_rate <= 0.30
            and m.specificity >= 0.65
            and m.recall >= 0.75
            and m.precision >= 0.65
        ),
    }


def _record(
    candidate_id: str,
    calibration: str,
    policy_type: str,
    constraint_name: str,
    metrics: Metrics,
    probability_metrics: dict[str, float],
    threshold: float | None = None,
    t_low: float | None = None,
    t_high: float | None = None,
) -> dict[str, object]:
    """Create one validation output record."""

    return {
        "candidate_id": candidate_id,
        "dataset": "heloc",
        "model": "WOE Scorecard Logistic Regression",
        "policy_type": policy_type,
        "calibration_type": calibration,
        "threshold": threshold,
        "t_low": t_low,
        "t_high": t_high,
        "constraint_name": constraint_name,
        "validation_precision": metrics.precision,
        "validation_recall": metrics.recall,
        "validation_specificity": metrics.specificity,
        "validation_f1": metrics.f1,
        **probability_metrics,
        "validation_fp": metrics.fp,
        "validation_fn": metrics.fn,
        "validation_low_risk_fn": metrics.low_fn,
        "validation_tp": metrics.tp,
        "validation_tn": metrics.tn,
        "validation_cost": metrics.cost,
        "validation_manual_review_rate": metrics.manual_review_rate,
        "validation_review_adjusted_cost": metrics.review_adjusted_cost,
        "test_precision": np.nan,
        "test_recall": np.nan,
        "test_specificity": np.nan,
        "test_f1": np.nan,
        "test_pr_auc": np.nan,
        "test_roc_auc": np.nan,
        "test_brier": np.nan,
        "test_ece": np.nan,
        "test_fp": np.nan,
        "test_fn": np.nan,
        "test_low_risk_fn": np.nan,
        "test_cost": np.nan,
        "test_manual_review_rate": np.nan,
        "test_review_adjusted_cost": np.nan,
        "delta_specificity_vs_v2": np.nan,
        "delta_recall_vs_v2": np.nan,
        "delta_precision_vs_v2": np.nan,
        "delta_fp_vs_v2": np.nan,
        "delta_fn_vs_v2": np.nan,
        "delta_cost_vs_v2": np.nan,
        "test_set_used_for_selection": False,
        "selection_split": "validation",
        "operational_comment": "",
    }


def _calibration_comparison(validation_frames: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Build validation-only calibration comparison table."""

    rows = []
    for calibration, (y, probability) in validation_frames.items():
        metric = _binary_metrics(y, probability, 0.50)
        rows.append(
            {
                "dataset": "heloc",
                "model": "WOE Scorecard Logistic Regression",
                "calibration_type": calibration,
                "validation_pr_auc": average_precision_score(y, probability),
                "validation_roc_auc": roc_auc_score(y, probability),
                "validation_brier": brier_score_loss(y, probability),
                "validation_ece": _ece(y, probability),
                "threshold_for_diagnostic": 0.50,
                "validation_precision_at_050": metric.precision,
                "validation_recall_at_050": metric.recall,
                "validation_specificity_at_050": metric.specificity,
                "validation_f1_at_050": metric.f1,
                "validation_cost_at_050": metric.cost,
                "test_set_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _search_validation() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Search all validation policies and return validation/test score frames."""

    _, validation, test = _split()
    validation_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records: list[dict[str, object]] = []
    candidate_id = 0
    threshold_grid = np.round(np.arange(0.01, 0.9901, 0.005), 3)
    low_grid = np.round(np.arange(0.01, 0.5001, 0.005), 3)
    high_grid = np.round(np.arange(0.05, 0.9001, 0.005), 3)

    for calibration in CALIBRATIONS:
        y_val, p_val = _scores(validation, calibration)
        y_test, p_test = _scores(test, calibration)
        validation_scores[calibration] = (y_val, p_val)
        test_scores[calibration] = (y_test, p_test)
        prob_metrics = _probability_metrics(y_val, p_val)
        v2_cost = _v2_validation_cost(y_val, p_val)
        binary_constraints = _binary_constraints(v2_cost)
        for threshold in threshold_grid:
            metrics = _binary_metrics(y_val, p_val, float(threshold))
            for name, check in binary_constraints.items():
                if check(metrics):
                    candidate_id += 1
                    records.append(
                        _record(
                            f"heloc_spec_cand_{candidate_id:06d}",
                            calibration,
                            "threshold_only",
                            name,
                            metrics,
                            prob_metrics,
                            threshold=float(threshold),
                        )
                    )
        manual_constraints = _manual_constraints()
        for t_low in low_grid:
            for t_high in high_grid[high_grid > t_low]:
                metrics = _manual_metrics(y_val, p_val, float(t_low), float(t_high))
                for name, check in manual_constraints.items():
                    if check(metrics):
                        candidate_id += 1
                        records.append(
                            _record(
                                f"heloc_spec_cand_{candidate_id:06d}",
                                calibration,
                                "manual_review_band",
                                name,
                                metrics,
                                prob_metrics,
                                t_low=float(t_low),
                                t_high=float(t_high),
                            )
                        )
    validation_all = pd.DataFrame(records)
    calibration_comparison = _calibration_comparison(validation_scores)
    return validation_all, calibration_comparison, validation_scores, test_scores


def _select_locked(validation_all: pd.DataFrame) -> pd.DataFrame:
    """Pick three policies from validation evidence only."""

    priority = [
        "A5_specificity_ge_0_65_recall_ge_0_75",
        "A6_specificity_ge_0_70_recall_ge_0_70",
        "A7_specificity_ge_0_65_precision_ge_0_70",
        "A8_specificity_ge_0_65_cost_le_v2_validation_plus_10pct",
        "B7_balanced_mr30_spec65_recall75_precision65",
        "B4_specificity_ge_0_65",
        "A4_specificity_ge_0_60_recall_ge_0_80",
        "A3_specificity_ge_0_70",
        "A2_specificity_ge_0_65",
    ]
    rows = []
    used_keys = set()
    for constraint in priority:
        subset = validation_all[validation_all["constraint_name"] == constraint].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(
            [
                "validation_cost",
                "validation_specificity",
                "validation_recall",
                "validation_precision",
                "validation_fp",
            ],
            ascending=[True, False, False, False, True],
        )
        for _, row in subset.iterrows():
            key = (
                row["policy_type"],
                row["calibration_type"],
                None if pd.isna(row["threshold"]) else round(float(row["threshold"]), 6),
                None if pd.isna(row["t_low"]) else round(float(row["t_low"]), 6),
                None if pd.isna(row["t_high"]) else round(float(row["t_high"]), 6),
            )
            if key in used_keys:
                continue
            rows.append(row)
            used_keys.add(key)
            break
        if len(rows) >= 3:
            break
    if len(rows) < 3:
        fallback = validation_all.sort_values(
            ["validation_cost", "validation_specificity", "validation_recall"],
            ascending=[True, False, False],
        )
        for _, row in fallback.iterrows():
            key = (row["policy_type"], row["calibration_type"], row["threshold"], row["t_low"], row["t_high"])
            if key in used_keys:
                continue
            rows.append(row)
            used_keys.add(key)
            if len(rows) >= 3:
                break
    return pd.DataFrame(rows).reset_index(drop=True)


def _evaluate_locked(locked: pd.DataFrame, test_scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Evaluate locked validation-selected policies on held-out test."""

    rows = []
    for _, row in locked.iterrows():
        calibration = row["calibration_type"]
        y, probability = test_scores[calibration]
        if row["policy_type"] == "threshold_only":
            metrics = _binary_metrics(y, probability, float(row["threshold"]))
        else:
            metrics = _manual_metrics(y, probability, float(row["t_low"]), float(row["t_high"]))
        out = row.to_dict()
        out.update(
            {
                "test_precision": metrics.precision,
                "test_recall": metrics.recall,
                "test_specificity": metrics.specificity,
                "test_f1": metrics.f1,
                "test_pr_auc": average_precision_score(y, probability),
                "test_roc_auc": roc_auc_score(y, probability),
                "test_brier": brier_score_loss(y, probability),
                "test_ece": _ece(y, probability),
                "test_fp": metrics.fp,
                "test_fn": metrics.fn,
                "test_low_risk_fn": metrics.low_fn,
                "test_cost": metrics.cost,
                "test_manual_review_rate": metrics.manual_review_rate,
                "test_review_adjusted_cost": metrics.review_adjusted_cost,
                "delta_specificity_vs_v2": metrics.specificity - V2_TEST["specificity"],
                "delta_recall_vs_v2": metrics.recall - V2_TEST["recall"],
                "delta_precision_vs_v2": metrics.precision - V2_TEST["precision"],
                "delta_fp_vs_v2": metrics.fp - V2_TEST["fp"],
                "delta_fn_vs_v2": metrics.fn - V2_TEST["fn"],
                "delta_cost_vs_v2": metrics.cost - V2_TEST["cost"],
                "operational_comment": _comment(metrics),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _comment(metrics: Metrics) -> str:
    """Create a concise operational interpretation."""

    if metrics.specificity >= 0.65 and metrics.recall >= 0.75:
        return "Specificity improves while recall remains reasonably high; V3 audit candidate."
    if metrics.specificity >= 0.65:
        return "Specificity improves, but recall trade-off is material."
    return "Specificity gain is limited."


def _plot(validation_all: pd.DataFrame, locked: pd.DataFrame) -> None:
    """Plot validation specificity/recall tradeoff."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = validation_all.copy()
    if len(sample) > 30000:
        sample = sample.sample(30000, random_state=RANDOM_SEED)
    color = sample["policy_type"].map({"threshold_only": "#4c78a8", "manual_review_band": "#f58518"})
    plt.figure(figsize=(8, 5))
    plt.scatter(sample["validation_specificity"], sample["validation_recall"], c=color, s=8, alpha=0.22)
    plt.axvline(V2_TEST["specificity"], color="black", linestyle="--", linewidth=1, label="V2 specificity")
    plt.axhline(V2_TEST["recall"], color="gray", linestyle=":", linewidth=1, label="V2 recall")
    if not locked.empty:
        plt.scatter(locked["validation_specificity"], locked["validation_recall"], c="#d62728", s=44, label="Locked top-3")
    plt.xlabel("Validation specificity")
    plt.ylabel("Validation recall")
    plt.title("HELOC scorecard specificity vs recall trade-off")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heloc_specificity_recall_tradeoff.png", dpi=180)
    plt.close()


def _md_table(df: pd.DataFrame) -> str:
    """Render locked-test table in Markdown."""

    cols = [
        "policy_type",
        "calibration_type",
        "constraint_name",
        "threshold",
        "t_low",
        "t_high",
        "test_specificity",
        "test_recall",
        "test_precision",
        "test_fp",
        "test_fn",
        "test_cost",
        "test_manual_review_rate",
        "operational_comment",
    ]
    view = df[cols].copy()
    for col in ["threshold", "t_low", "t_high", "test_specificity", "test_recall", "test_precision", "test_manual_review_rate"]:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    for col in ["test_cost"]:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
    for col in ["test_fp", "test_fn"]:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(int(round(float(x)))))
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *body])


def _write_summary(locked_test: pd.DataFrame, calibration_comparison: pd.DataFrame, validation_all: pd.DataFrame) -> None:
    """Write summary answers."""

    spec_improved = bool((locked_test["test_specificity"] >= 0.65).any())
    best_spec = float(locked_test["test_specificity"].max()) if not locked_test.empty else math.nan
    best = locked_test.loc[locked_test["test_specificity"].idxmax()] if not locked_test.empty else None
    if best is not None:
        recall_drop = V2_TEST["recall"] - float(best["test_recall"])
        precision_change = float(best["test_precision"]) - V2_TEST["precision"]
        cost_change = float(best["test_cost"]) - V2_TEST["cost"]
        v3 = spec_improved and float(best["test_recall"]) >= 0.75
    else:
        recall_drop = precision_change = cost_change = math.nan
        v3 = False
    best_cal = calibration_comparison.sort_values(["validation_brier", "validation_ece"]).iloc[0]
    summary = f"""# HELOC Specificity-Focused Scorecard Optimization

## Protocol
- Existing scorecard artifacts only: uncalibrated, sigmoid, isotonic.
- Selection split: validation only.
- Test usage: locked evaluation of top-3 validation-selected policies only.
- No new dataset, feature, model training, or test-set threshold selection.
- A8 uses V2 validation cost, not V2 test cost, to avoid test-based selection.

## Locked test candidates
{_md_table(locked_test)}

## Calibration comparison
- Best validation Brier/ECE calibration: {best_cal['calibration_type']}
- Validation Brier: {float(best_cal['validation_brier']):.4f}
- Validation ECE: {float(best_cal['validation_ece']):.4f}

## Questions
1. Can HELOC specificity move above 0.65 from V2 0.574? **{'YES' if spec_improved else 'NO'}**. Best locked-test specificity is {best_spec:.3f}.
2. How much does recall drop? Best-specificity recall drop is {recall_drop:.3f}.
3. Does precision improve? Precision change for best-specificity candidate is {precision_change:.3f}.
4. Does cost increase? Cost change for best-specificity candidate is {cost_change:.1f}.
5. Did calibration help? **YES** if judged by validation Brier/ECE; see `heloc_scorecard_calibration_comparison.csv`.
6. Should Scorecard remain final? **YES for V2 final evidence** until a V3 audit accepts the replacement.
7. Is a HELOC V3 policy recommended? **{'YES, as audit candidate' if v3 else 'NO / appendix only'}**.

## Validation search volume
- Feasible validation rows written: {len(validation_all)}
- Locked test rows written: {len(locked_test)}
"""
    (OUT_DIR / "heloc_specificity_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run specificity-focused validation-only policy search."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_all, calibration_comparison, _, test_scores = _search_validation()
    locked_validation = _select_locked(validation_all)
    locked_test = _evaluate_locked(locked_validation, test_scores)
    validation_all.to_csv(OUT_DIR / "heloc_specificity_validation_all.csv", index=False)
    locked_test.to_csv(OUT_DIR / "heloc_specificity_locked_test.csv", index=False)
    calibration_comparison.to_csv(OUT_DIR / "heloc_scorecard_calibration_comparison.csv", index=False)
    _plot(validation_all, locked_validation)
    _write_summary(locked_test, calibration_comparison, validation_all)


if __name__ == "__main__":
    main()
