"""Validation-only Taiwan recall-constrained policy search.

No model is trained here. The script loads the saved calibrated CatBoost
pipeline, regenerates validation/test probabilities by inference only, selects
candidate policies on validation, and evaluates only the locked top-3 policies
on the held-out test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import math

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/taiwan_recall"
MODEL_PATH = PROJECT_ROOT / "outputs/models/calibrated/taiwan/catboost_isotonic.joblib"
DATA_PATH = PROJECT_ROOT / "data/processed/taiwan_model_ready.csv"
TARGET = "default_next_month"
RANDOM_SEED = 42
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COSTS = [0.1, 0.25, 0.5, 1.0]
V2_T_LOW = 0.14
V2_T_HIGH = 0.28
V2_TEST = {
    "precision": 0.5287595287595288,
    "recall": 0.5749811605124341,
    "specificity": 0.8544832013695699,
    "fp": 680,
    "fn": 564,
    "cost": 2705.5,
    "manual_review_rate": 0.2951666666666667,
}


@dataclass(frozen=True)
class PolicyMetrics:
    """Metrics for one binary or manual-review policy."""

    precision: float
    recall: float
    specificity: float
    fp: int
    fn: int
    low_fn: int
    cost: float
    manual_review_rate: float
    review_adjusted_cost: float
    low_risk_count: int
    manual_review_count: int
    high_risk_count: int
    tp: int
    tn_high_vs_rest: int


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a stable division result."""

    return float(numerator / denominator) if denominator else 0.0


def _split_taiwan() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate the canonical 60/20/20 stratified Taiwan split."""

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


def _load_scores(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate probabilities from the saved calibrated CatBoost model."""

    model = joblib.load(MODEL_PATH)
    y = frame[TARGET].astype(int).to_numpy()
    x = frame.drop(columns=[TARGET])
    probability = model.predict_proba(x)[:, 1]
    return y, probability


def _binary_metrics(y: np.ndarray, probability: np.ndarray, t_high: float) -> PolicyMetrics:
    """Compute binary threshold metrics."""

    high = probability >= t_high
    default = y == 1
    nondefault = ~default
    tp = int(np.sum(high & default))
    fp = int(np.sum(high & nondefault))
    fn = int(np.sum((~high) & default))
    tn = int(np.sum((~high) & nondefault))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, int(np.sum(default)))
    specificity = _safe_divide(tn, int(np.sum(nondefault)))
    cost = fp * FP_COST + fn * FN_COST
    return PolicyMetrics(
        precision=precision,
        recall=recall,
        specificity=specificity,
        fp=fp,
        fn=fn,
        low_fn=fn,
        cost=cost,
        manual_review_rate=0.0,
        review_adjusted_cost=cost,
        low_risk_count=int(np.sum(~high)),
        manual_review_count=0,
        high_risk_count=int(np.sum(high)),
        tp=tp,
        tn_high_vs_rest=tn,
    )


def _manual_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    t_low: float,
    t_high: float,
    review_cost: float,
) -> PolicyMetrics:
    """Compute three-way manual-review policy metrics."""

    low = probability < t_low
    high = probability >= t_high
    manual = (~low) & (~high)
    default = y == 1
    nondefault = ~default
    total_defaults = int(np.sum(default))
    total_nondefaults = int(np.sum(nondefault))

    high_default = int(np.sum(high & default))
    high_nondefault = int(np.sum(high & nondefault))
    low_default = int(np.sum(low & default))
    low_nondefault = int(np.sum(low & nondefault))
    not_high_default = total_defaults - high_default
    not_high_nondefault = total_nondefaults - high_nondefault
    manual_count = int(np.sum(manual))
    high_count = int(np.sum(high))
    low_count = int(np.sum(low))

    precision = _safe_divide(high_default, high_default + high_nondefault)
    recall = _safe_divide(high_default, total_defaults)
    specificity = _safe_divide(not_high_nondefault, total_nondefaults)
    review_adjusted_cost = high_nondefault * FP_COST + low_default * FN_COST + manual_count * review_cost
    return PolicyMetrics(
        precision=precision,
        recall=recall,
        specificity=specificity,
        fp=high_nondefault,
        fn=not_high_default,
        low_fn=low_default,
        cost=review_adjusted_cost,
        manual_review_rate=_safe_divide(manual_count, len(y)),
        review_adjusted_cost=review_adjusted_cost,
        low_risk_count=low_count,
        manual_review_count=manual_count,
        high_risk_count=high_count,
        tp=high_default,
        tn_high_vs_rest=not_high_nondefault,
    )


def _v2_validation_reference(y: np.ndarray, probability: np.ndarray, review_cost: float) -> PolicyMetrics:
    """Return V2 validation metrics under the current review-cost assumption."""

    return _manual_metrics(y, probability, V2_T_LOW, V2_T_HIGH, review_cost)


def _constraint_functions(v2_validation_cost_by_review_cost: dict[float, float]) -> dict[str, Callable[[PolicyMetrics, float], bool]]:
    """Build recall-constrained feasibility checks."""

    return {
        "A_recall_ge_0_60": lambda m, rc: m.recall >= 0.60,
        "B_recall_ge_0_65": lambda m, rc: m.recall >= 0.65,
        "C_recall_ge_0_70": lambda m, rc: m.recall >= 0.70,
        "D_recall_ge_0_60_precision_ge_0_45": lambda m, rc: m.recall >= 0.60 and m.precision >= 0.45,
        "E_recall_ge_0_60_specificity_ge_0_80": lambda m, rc: m.recall >= 0.60 and m.specificity >= 0.80,
        "F_recall_ge_0_65_mr_rate_le_0_35": lambda m, rc: m.recall >= 0.65 and m.manual_review_rate <= 0.35,
        "G_recall_ge_0_60_mr_rate_le_0_30": lambda m, rc: m.recall >= 0.60 and m.manual_review_rate <= 0.30,
        "H_recall_ge_0_60_cost_le_v2_validation_plus_10pct": (
            lambda m, rc: m.recall >= 0.60
            and m.cost <= v2_validation_cost_by_review_cost.get(rc, v2_validation_cost_by_review_cost[0.5]) * 1.10
        ),
        "I_recall_ge_0_65_fp_le_1000": lambda m, rc: m.recall >= 0.65 and m.fp <= 1000,
    }


def _record(
    candidate_id: str,
    policy_type: str,
    constraint_name: str,
    review_cost: float,
    metrics: PolicyMetrics,
    t_high: float,
    t_low: float | None = None,
) -> dict[str, object]:
    """Build one output record with validation fields populated."""

    return {
        "candidate_id": candidate_id,
        "dataset": "taiwan",
        "model": "V2 CatBoost isotonic",
        "policy_type": policy_type,
        "t_low": t_low,
        "t_high": t_high,
        "constraint_name": constraint_name,
        "review_cost": review_cost,
        "validation_precision": metrics.precision,
        "validation_recall": metrics.recall,
        "validation_specificity": metrics.specificity,
        "validation_fp": metrics.fp,
        "validation_fn": metrics.fn,
        "validation_low_risk_fn": metrics.low_fn,
        "validation_cost": metrics.cost,
        "validation_manual_review_rate": metrics.manual_review_rate,
        "validation_review_adjusted_cost": metrics.review_adjusted_cost,
        "test_precision": np.nan,
        "test_recall": np.nan,
        "test_specificity": np.nan,
        "test_fp": np.nan,
        "test_fn": np.nan,
        "test_low_risk_fn": np.nan,
        "test_cost": np.nan,
        "test_manual_review_rate": np.nan,
        "test_review_adjusted_cost": np.nan,
        "delta_recall_vs_v2": np.nan,
        "delta_precision_vs_v2": np.nan,
        "delta_specificity_vs_v2": np.nan,
        "delta_fp_vs_v2": np.nan,
        "delta_fn_vs_v2": np.nan,
        "delta_mr_rate_vs_v2": np.nan,
        "delta_cost_vs_v2": np.nan,
        "operational_comment": "",
        "test_set_used_for_selection": False,
        "selection_split": "validation",
        "source_model_path": str(MODEL_PATH),
    }


def _select_locked_candidates(validation_all: pd.DataFrame) -> pd.DataFrame:
    """Choose the top-3 unique operational policies using validation-only ordering.

    Pure cost sorting can over-reward very cheap manual-review workload settings.
    Because this weakness-closing prompt is explicitly about recall improvement
    without exploding FP/MR workload, locked candidates are drawn first from
    constraints that control specificity, manual-review rate, cost, or FP.
    """

    priority_constraints = [
        "G_recall_ge_0_60_mr_rate_le_0_30",
        "E_recall_ge_0_60_specificity_ge_0_80",
        "H_recall_ge_0_60_cost_le_v2_validation_plus_10pct",
        "F_recall_ge_0_65_mr_rate_le_0_35",
        "I_recall_ge_0_65_fp_le_1000",
        "D_recall_ge_0_60_precision_ge_0_45",
        "B_recall_ge_0_65",
        "C_recall_ge_0_70",
        "A_recall_ge_0_60",
    ]
    candidates = validation_all.copy()
    candidates["constraint_priority"] = candidates["constraint_name"].map(
        {name: index for index, name in enumerate(priority_constraints)}
    ).fillna(len(priority_constraints))
    sorted_candidates = candidates.sort_values(
        [
            "constraint_priority",
            "validation_review_adjusted_cost",
            "validation_fn",
            "validation_fp",
            "validation_manual_review_rate",
            "validation_specificity",
        ],
        ascending=[True, True, True, True, True, False],
    ).copy()
    selected_rows = []
    seen = set()
    for _, row in sorted_candidates.iterrows():
        key = (
            row["policy_type"],
            None if pd.isna(row["t_low"]) else round(float(row["t_low"]), 6),
            round(float(row["t_high"]), 6),
            round(float(row["review_cost"]), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        selected_rows.append(row)
        if len(selected_rows) == 3:
            break
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _evaluate_locked(
    locked_validation: pd.DataFrame,
    y_test: np.ndarray,
    p_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate locked validation-selected policies on the held-out test split."""

    rows = []
    for _, row in locked_validation.iterrows():
        if row["policy_type"] == "binary_threshold":
            metrics = _binary_metrics(y_test, p_test, float(row["t_high"]))
        else:
            metrics = _manual_metrics(
                y_test,
                p_test,
                float(row["t_low"]),
                float(row["t_high"]),
                float(row["review_cost"]),
            )
        out = row.to_dict()
        v2_same_review_cost = _manual_metrics(
            y_test,
            p_test,
            V2_T_LOW,
            V2_T_HIGH,
            float(row["review_cost"]) if row["policy_type"] == "manual_review_band" else 0.5,
        )
        out.update(
            {
                "test_precision": metrics.precision,
                "test_recall": metrics.recall,
                "test_specificity": metrics.specificity,
                "test_fp": metrics.fp,
                "test_fn": metrics.fn,
                "test_low_risk_fn": metrics.low_fn,
                "test_cost": metrics.cost,
                "test_manual_review_rate": metrics.manual_review_rate,
                "test_review_adjusted_cost": metrics.review_adjusted_cost,
                "delta_recall_vs_v2": metrics.recall - V2_TEST["recall"],
                "delta_precision_vs_v2": metrics.precision - V2_TEST["precision"],
                "delta_specificity_vs_v2": metrics.specificity - V2_TEST["specificity"],
                "delta_fp_vs_v2": metrics.fp - V2_TEST["fp"],
                "delta_fn_vs_v2": metrics.fn - V2_TEST["fn"],
                "delta_mr_rate_vs_v2": metrics.manual_review_rate - V2_TEST["manual_review_rate"],
                "delta_cost_vs_v2": metrics.cost - V2_TEST["cost"],
                "v2_cost_same_review_cost": v2_same_review_cost.cost,
                "delta_cost_vs_v2_same_review_cost": metrics.cost - v2_same_review_cost.cost,
            }
        )
        out["operational_comment"] = _comment(metrics)
        rows.append(out)
    return pd.DataFrame(rows)


def _comment(metrics: PolicyMetrics) -> str:
    """Generate a concise operational comment."""

    if metrics.recall >= 0.60 and metrics.fp <= 1000 and metrics.manual_review_rate <= 0.35:
        return "Recall improves with bounded FP and manual-review workload; candidate worth audit review."
    if metrics.recall >= 0.60:
        return "Recall improves, but FP/workload/cost trade-off needs caution."
    return "Does not materially clear the recall target."


def _plot_outputs(validation_all: pd.DataFrame, locked_test: pd.DataFrame) -> None:
    """Create trade-off plots."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = validation_all.copy()
    if len(sample) > 20000:
        sample = sample.sample(20000, random_state=RANDOM_SEED)

    plt.figure(figsize=(8, 5))
    colors = sample["policy_type"].map({"binary_threshold": "#4c78a8", "manual_review_band": "#f58518"})
    plt.scatter(sample["validation_recall"], sample["validation_precision"], c=colors, s=8, alpha=0.25)
    plt.axvline(V2_TEST["recall"], color="black", linestyle="--", linewidth=1, label="V2 test recall")
    plt.axhline(V2_TEST["precision"], color="gray", linestyle=":", linewidth=1, label="V2 test precision")
    if not locked_test.empty:
        plt.scatter(locked_test["validation_recall"], locked_test["validation_precision"], c="#d62728", s=42, label="Locked top-3")
    plt.xlabel("Validation recall")
    plt.ylabel("Validation precision")
    plt.title("Taiwan CatBoost recall vs precision trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "taiwan_recall_precision_tradeoff.png", dpi=180)
    plt.close()

    manual = sample.loc[sample["policy_type"] == "manual_review_band"].copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(
        manual["validation_manual_review_rate"],
        manual["validation_review_adjusted_cost"],
        c=manual["review_cost"],
        cmap="viridis",
        s=8,
        alpha=0.25,
    )
    plt.axvline(V2_TEST["manual_review_rate"], color="black", linestyle="--", linewidth=1, label="V2 test MR rate")
    plt.axhline(V2_TEST["cost"], color="gray", linestyle=":", linewidth=1, label="V2 test cost")
    if not locked_test.empty:
        plt.scatter(
            locked_test["validation_manual_review_rate"],
            locked_test["validation_review_adjusted_cost"],
            c="#d62728",
            s=42,
            label="Locked top-3",
        )
    plt.xlabel("Validation manual-review rate")
    plt.ylabel("Validation review-adjusted cost")
    plt.title("Taiwan CatBoost manual-review workload vs cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "taiwan_recall_mr_cost_tradeoff.png", dpi=180)
    plt.close()


def _write_summary(locked_test: pd.DataFrame, validation_all: pd.DataFrame) -> None:
    """Write Markdown summary answering the requested questions."""

    best = locked_test.sort_values(["test_cost", "test_fn", "test_fp"]).copy()
    recall_improved = bool((locked_test["test_recall"] >= 0.60).any())
    best_recall = float(locked_test["test_recall"].max()) if not locked_test.empty else math.nan
    best_recall_row = locked_test.loc[locked_test["test_recall"].idxmax()] if not locked_test.empty else None
    acceptable = bool(
        (
            (locked_test["test_recall"] >= 0.60)
            & (locked_test["test_fp"] <= 1000)
            & (locked_test["test_manual_review_rate"] <= 0.35)
            & (locked_test["test_cost"] <= V2_TEST["cost"] * 1.10)
        ).any()
    )

    def table(df: pd.DataFrame) -> str:
        cols = [
            "policy_type",
            "constraint_name",
            "review_cost",
            "t_low",
            "t_high",
            "test_recall",
            "test_precision",
            "test_specificity",
            "test_fp",
            "test_fn",
            "test_cost",
            "delta_cost_vs_v2_same_review_cost",
            "test_manual_review_rate",
            "operational_comment",
        ]
        view = df[cols].copy()
        for col in ["review_cost", "t_low", "t_high", "test_recall", "test_precision", "test_specificity", "test_manual_review_rate"]:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        for col in ["test_cost", "delta_cost_vs_v2_same_review_cost"]:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.1f}")
        for col in ["test_fp", "test_fn"]:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(int(round(float(x)))))
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
        return "\n".join([header, sep, *body])

    if best_recall_row is not None:
        fp_cost_text = f"{int(best_recall_row['test_fp'])} FP versus V2 {V2_TEST['fp']} FP"
        mr_text = f"{best_recall_row['test_manual_review_rate']:.3f} versus V2 {V2_TEST['manual_review_rate']:.3f}"
        cost_text = (
            f"{best_recall_row['test_cost']:.1f}; same-review-cost delta versus V2 is "
            f"{best_recall_row['delta_cost_vs_v2_same_review_cost']:.1f}"
        )
        value_text = (
            "conditionally useful" if acceptable else "not strong enough to replace V2 without further audit"
        )
    else:
        fp_cost_text = mr_text = cost_text = value_text = "not available"

    summary = f"""# Taiwan Recall-Constrained Policy Search

## Protocol
- Saved model used: `{MODEL_PATH.relative_to(PROJECT_ROOT)}`
- Selection split: validation only
- Test usage: locked evaluation of the top-3 validation-selected policies only
- New model training: no
- New feature engineering: no
- Test-set threshold selection: no
- H constraint uses V2 validation cost, not V2 test cost, to avoid test-based selection.

## Locked test candidates
{table(best)}

## Questions
1. Can Taiwan recall move above 0.60 from the V2 level of 0.575? **{'YES' if recall_improved else 'NO'}**. Best locked-test recall is {best_recall:.3f}.
2. What is the FP cost of that recall gain? {fp_cost_text}.
3. How much does the manual-review rate change? {mr_text}.
4. How much does the review-adjusted cost change? {cost_text}.
5. Is the recall gain operationally valuable? **{'CONDITIONAL' if acceptable else 'NO'}**. {value_text}.
6. Should V2 change now? **V2 should remain final for now** unless the candidate passes a later final audit and improves the replacement-rule criteria.

## Validation search volume
- Feasible validation rows written: {len(validation_all)}
- Locked test rows written: {len(locked_test)}
"""
    (OUT_DIR / "taiwan_recall_tradeoff_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    """Run the recall-constrained validation-only policy search."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, validation, test = _split_taiwan()
    y_val, p_val = _load_scores(validation)
    y_test, p_test = _load_scores(test)

    v2_validation_cost = {
        review_cost: _v2_validation_reference(y_val, p_val, review_cost).cost
        for review_cost in REVIEW_COSTS
    }
    constraints = _constraint_functions(v2_validation_cost)
    records: list[dict[str, object]] = []
    candidate_number = 0

    t_high_grid = np.round(np.arange(0.05, 0.9001, 0.005), 3)
    t_low_grid = np.round(np.arange(0.01, 0.5001, 0.005), 3)

    for t_high in t_high_grid:
        metrics = _binary_metrics(y_val, p_val, float(t_high))
        for constraint_name, feasible in constraints.items():
            if feasible(metrics, 0.0):
                candidate_number += 1
                records.append(
                    _record(
                        f"taiwan_recall_cand_{candidate_number:06d}",
                        "binary_threshold",
                        constraint_name,
                        0.0,
                        metrics,
                        float(t_high),
                    )
                )

    for review_cost in REVIEW_COSTS:
        for t_low in t_low_grid:
            valid_highs = t_high_grid[t_high_grid > t_low]
            for t_high in valid_highs:
                metrics = _manual_metrics(y_val, p_val, float(t_low), float(t_high), review_cost)
                for constraint_name, feasible in constraints.items():
                    if feasible(metrics, review_cost):
                        candidate_number += 1
                        records.append(
                            _record(
                                f"taiwan_recall_cand_{candidate_number:06d}",
                                "manual_review_band",
                                constraint_name,
                                review_cost,
                                metrics,
                                float(t_high),
                                float(t_low),
                            )
                        )

    validation_all = pd.DataFrame(records)
    validation_all.to_csv(OUT_DIR / "taiwan_recall_policy_validation_all.csv", index=False)

    locked_validation = _select_locked_candidates(validation_all)
    locked_test = _evaluate_locked(locked_validation, y_test, p_test)
    locked_test.to_csv(OUT_DIR / "taiwan_recall_policy_locked_test.csv", index=False)

    _plot_outputs(validation_all, locked_test)
    _write_summary(locked_test, validation_all)


if __name__ == "__main__":
    main()
