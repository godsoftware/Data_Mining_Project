"""Segment-aware threshold policy analysis for final weakness closing.

This script does not train models, create features, change data, or select from
the test set. It uses existing calibrated probability scores and simple,
pre-specified segment definitions learned from non-test data, then selects
segment-level policies on validation only.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/segment_thresholds"
TARGETS = {"taiwan": "default_next_month", "heloc": "bad_flag"}
DATA_PATHS = {
    "taiwan": PROJECT_ROOT / "data/processed/taiwan_model_ready.csv",
    "heloc": PROJECT_ROOT / "data/processed/heloc_model_ready.csv",
}
RANDOM_SEED = 42
BASE_FN_COST = 5.0
BASE_FP_COST = 1.0
REVIEW_COST = 0.5
MIN_SEGMENT_VALIDATION_N = 100


@dataclass(frozen=True)
class SegmentCandidate:
    """Candidate policy for a single segment."""

    segment_value: str
    policy_type: str
    threshold: float | None
    t_low: float | None
    t_high: float | None
    local_cost: float
    local_recall: float
    local_specificity: float
    local_fp: int
    local_fn: int
    local_mr_rate: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "segment_value": self.segment_value,
            "policy_type": self.policy_type,
            "threshold": self.threshold,
            "t_low": self.t_low,
            "t_high": self.t_high,
        }


def _split_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return non-test pool, validation, and locked-test splits."""

    target = TARGETS[dataset]
    frame = pd.read_csv(DATA_PATHS[dataset])
    train_validation, test = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame[target],
        random_state=RANDOM_SEED,
    )
    _, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[target],
        random_state=RANDOM_SEED,
    )
    return train_validation.copy(), validation.copy(), test.copy()


def _positive_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability."""

    probability = estimator.predict_proba(x)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        return np.asarray(probability)[:, -1]
    return np.asarray(probability)[:, classes.index(1)]


def _load_score(dataset: str, frame: pd.DataFrame) -> np.ndarray:
    """Load the existing final-family scorer for a dataset."""

    target = TARGETS[dataset]
    if dataset == "taiwan":
        model_name, calibration = "catboost", "isotonic"
    else:
        model_name, calibration = "woe_scorecard_logistic_regression", "sigmoid"
    path = PROJECT_ROOT / "outputs/models/calibrated" / dataset / f"{model_name}_{calibration}.joblib"
    estimator = joblib.load(path)
    return _positive_probability(estimator, frame.drop(columns=[target]))


def _cut_by_train_terciles(train_validation: pd.Series, values: pd.Series, prefix: str, reverse_risk: bool = False) -> pd.Series:
    """Apply train/validation-learned tercile boundaries to a series."""

    numeric_train = pd.to_numeric(train_validation, errors="coerce").replace([np.inf, -np.inf], np.nan)
    numeric_values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    q1, q2 = numeric_train.quantile([1 / 3, 2 / 3]).to_list()
    if not np.isfinite(q1) or not np.isfinite(q2) or q1 == q2:
        ranked_train = numeric_train.rank(method="average", pct=True)
        q1, q2 = ranked_train.quantile([1 / 3, 2 / 3]).to_list()
        numeric_values = numeric_values.rank(method="average", pct=True)
    labels = [f"{prefix}_low", f"{prefix}_medium", f"{prefix}_high"]
    if reverse_risk:
        labels = [f"{prefix}_high_risk", f"{prefix}_medium_risk", f"{prefix}_low_risk"]
    return pd.cut(
        numeric_values,
        bins=[-np.inf, q1, q2, np.inf],
        labels=labels,
        include_lowest=True,
    ).astype(str).fillna(f"{prefix}_missing")


def _segment_strategies(dataset: str, train_validation: pd.DataFrame, frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Build at most three segment strategies from existing variables."""

    if dataset == "taiwan":
        recent_delay = pd.to_numeric(frame["recent_delay"], errors="coerce").fillna(0)
        delay_segment = pd.Series(
            np.select(
                [recent_delay <= 0, recent_delay == 1, recent_delay >= 2],
                ["no_recent_delay", "mild_delay", "severe_delay"],
                default="delay_missing",
            ),
            index=frame.index,
        )
        return {
            "limit_bal_terciles": _cut_by_train_terciles(train_validation["LIMIT_BAL"], frame["LIMIT_BAL"], "limit"),
            "recent_delay_groups": delay_segment,
            "utilization_terciles": _cut_by_train_terciles(train_validation["utilization_proxy"], frame["utilization_proxy"], "utilization"),
        }

    return {
        "external_risk_terciles": _cut_by_train_terciles(
            train_validation["ExternalRiskEstimate"],
            frame["ExternalRiskEstimate"],
            "external_risk",
            reverse_risk=True,
        ),
        "revolving_burden_terciles": _cut_by_train_terciles(
            train_validation["revolving_burden_proxy"],
            frame["revolving_burden_proxy"],
            "revolving_burden",
        ),
        "percent_never_delq_terciles": _cut_by_train_terciles(
            train_validation["PercentTradesNeverDelq"],
            frame["PercentTradesNeverDelq"],
            "never_delq",
            reverse_risk=True,
        ),
    }


def _masks_from_policy(score: np.ndarray, policy_type: str, threshold: float | None, t_low: float | None, t_high: float | None) -> dict[str, np.ndarray]:
    """Return low/manual/high masks for one policy."""

    n = len(score)
    if policy_type == "threshold_only":
        high = score >= float(threshold)
        low = ~high
        manual = np.zeros(n, dtype=bool)
    elif policy_type == "manual_review_band":
        low = score < float(t_low)
        high = score >= float(t_high)
        manual = (~low) & (~high)
    else:
        raise ValueError(f"Unknown policy_type={policy_type}")
    return {"low": low, "manual": manual, "high": high}


def _metrics(y: np.ndarray, score: np.ndarray, masks: dict[str, np.ndarray], high_exposure_mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute binary-comparable and manual-review metrics."""

    default = y == 1
    nondefault = y == 0
    high = masks["high"]
    low = masks["low"]
    manual = masks["manual"]
    tp = int(np.sum(high & default))
    fp = int(np.sum(high & nondefault))
    fn = int(np.sum((~high) & default))
    tn = int(np.sum((~high) & nondefault))
    low_fn = int(np.sum(low & default))
    manual_count = int(np.sum(manual))
    cost = fp * BASE_FP_COST + low_fn * BASE_FN_COST + manual_count * REVIEW_COST
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    high_recall = np.nan
    if high_exposure_mask is not None:
        high_defaults = default & high_exposure_mask
        high_recall = float(np.sum(high & high_defaults) / np.sum(high_defaults)) if np.sum(high_defaults) else np.nan
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "low_risk_default_count": low_fn,
        "manual_review_count": manual_count,
        "manual_review_rate": manual_count / len(y),
        "cost": cost,
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "high_exposure_segment_recall": high_recall,
    }


def _segment_candidates(y: np.ndarray, score: np.ndarray, segment_values: pd.Series) -> dict[str, list[SegmentCandidate]]:
    """Generate a compact candidate list per segment using validation only."""

    candidates: dict[str, list[SegmentCandidate]] = {}
    for segment_value in sorted(segment_values.dropna().unique()):
        idx = (segment_values == segment_value).to_numpy()
        if int(np.sum(idx)) < MIN_SEGMENT_VALIDATION_N:
            candidates[segment_value] = []
            continue
        local_y = y[idx]
        local_score = score[idx]
        local_candidates: list[SegmentCandidate] = []
        for threshold in np.round(np.arange(0.05, 0.8001, 0.01), 3):
            masks = _masks_from_policy(local_score, "threshold_only", threshold, None, None)
            m = _metrics(local_y, local_score, masks)
            local_candidates.append(
                SegmentCandidate(
                    segment_value,
                    "threshold_only",
                    float(threshold),
                    None,
                    None,
                    m["cost"],
                    m["recall"],
                    m["specificity"],
                    int(m["fp"]),
                    int(m["fn"]),
                    float(m["manual_review_rate"]),
                )
            )
        low_grid = np.round(np.arange(0.05, 0.4001, 0.025), 3)
        high_grid = np.round(np.arange(0.10, 0.8001, 0.025), 3)
        for t_low in low_grid:
            for t_high in high_grid:
                if t_low >= t_high:
                    continue
                masks = _masks_from_policy(local_score, "manual_review_band", None, float(t_low), float(t_high))
                m = _metrics(local_y, local_score, masks)
                local_candidates.append(
                    SegmentCandidate(
                        segment_value,
                        "manual_review_band",
                        None,
                        float(t_low),
                        float(t_high),
                        m["cost"],
                        m["recall"],
                        m["specificity"],
                        int(m["fp"]),
                        int(m["fn"]),
                        float(m["manual_review_rate"]),
                    )
                )

        # Keep a diverse compact frontier: low cost, high recall, high specificity.
        frame = pd.DataFrame([c.__dict__ for c in local_candidates])
        selected_idx = set()
        for sort_cols, ascending in [
            (["local_cost", "local_fp", "local_fn"], [True, True, True]),
            (["local_recall", "local_fp", "local_cost"], [False, True, True]),
            (["local_specificity", "local_fn", "local_cost"], [False, True, True]),
            (["local_mr_rate", "local_cost"], [True, True]),
        ]:
            selected_idx.update(frame.sort_values(sort_cols, ascending=ascending).head(10).index.tolist())
        compact = [local_candidates[i] for i in sorted(selected_idx)]
        candidates[segment_value] = compact[:30]
    return candidates


def _combine_policy_masks(score: np.ndarray, segment_values: pd.Series, candidate_map: dict[str, SegmentCandidate]) -> dict[str, np.ndarray]:
    """Apply segment-specific policies to a full validation/test score vector."""

    n = len(score)
    low = np.zeros(n, dtype=bool)
    manual = np.zeros(n, dtype=bool)
    high = np.zeros(n, dtype=bool)
    for segment_value, candidate in candidate_map.items():
        idx = (segment_values == segment_value).to_numpy()
        masks = _masks_from_policy(score[idx], candidate.policy_type, candidate.threshold, candidate.t_low, candidate.t_high)
        low[idx] = masks["low"]
        manual[idx] = masks["manual"]
        high[idx] = masks["high"]
    return {"low": low, "manual": manual, "high": high}


def _high_exposure_mask(dataset: str, strategy: str, segment_values: pd.Series) -> np.ndarray:
    """Identify the strategy-specific higher-risk/exposure segment."""

    values = segment_values.astype(str)
    if dataset == "taiwan":
        if strategy == "limit_bal_terciles":
            return values.str.contains("limit_high").to_numpy()
        if strategy == "recent_delay_groups":
            return values.str.contains("severe_delay").to_numpy()
        return values.str.contains("utilization_high").to_numpy()
    if strategy == "external_risk_terciles":
        return values.str.contains("external_risk_high_risk").to_numpy()
    if strategy == "revolving_burden_terciles":
        return values.str.contains("revolving_burden_high").to_numpy()
    return values.str.contains("never_delq_high_risk").to_numpy()


def _v2_reference_metrics(dataset: str, y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute validation V2 reference metrics without using test."""

    if dataset == "taiwan":
        masks = _masks_from_policy(score, "manual_review_band", None, 0.14, 0.28)
    else:
        masks = _masks_from_policy(score, "manual_review_band", None, 0.16, 0.39)
    return _metrics(y, score, masks)


def _evaluate_segment_strategy(
    dataset: str,
    strategy: str,
    y: np.ndarray,
    score: np.ndarray,
    segment_values: pd.Series,
    v2_metrics: dict[str, float],
) -> pd.DataFrame:
    """Enumerate compact segment policy combinations on validation."""

    candidates = _segment_candidates(y, score, segment_values)
    valid_segments = {k: v for k, v in candidates.items() if v}
    if len(valid_segments) < 2:
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "split": "validation",
                    "segment_strategy": strategy,
                    "candidate_id": f"{dataset}_{strategy}_invalid",
                    "valid_candidate": False,
                    "invalid_reason": "fewer_than_two_valid_segments_or_segment_sample_too_small",
                    "segment_policy_json": "{}",
                    "test_set_used_for_selection": False,
                }
            ]
        )

    high_mask = _high_exposure_mask(dataset, strategy, segment_values)
    rows: list[dict[str, Any]] = []
    segment_names = list(valid_segments)
    for combo_id, combo in enumerate(itertools.product(*(valid_segments[name] for name in segment_names))):
        candidate_map = {name: candidate for name, candidate in zip(segment_names, combo)}
        masks = _combine_policy_masks(score, segment_values, candidate_map)
        metrics = _metrics(y, score, masks, high_mask)
        if dataset == "taiwan":
            constraint_ok = (
                metrics["recall"] >= v2_metrics["recall"]
                and metrics["fp"] <= v2_metrics["fp"] * 1.20
                and metrics["manual_review_rate"] <= 0.35
            )
            objective_primary = -metrics["high_exposure_segment_recall"]
            objective_secondary = metrics["cost"]
            comment = "Taiwan objective: keep recall >= V2 validation, FP <= V2+20%, MR <= 0.35; maximize high-exposure recall."
        else:
            constraint_ok = (
                metrics["specificity"] >= v2_metrics["specificity"]
                and metrics["recall"] >= 0.75
                and metrics["manual_review_rate"] <= 0.30
            )
            objective_primary = -metrics["specificity"]
            objective_secondary = metrics["cost"]
            comment = "HELOC objective: specificity >= V2 validation, recall >= 0.75, MR <= 0.30."
        rows.append(
            {
                "dataset": dataset,
                "split": "validation",
                "segment_strategy": strategy,
                "candidate_id": f"{dataset}_{strategy}_{combo_id:05d}",
                "valid_candidate": True,
                "invalid_reason": "",
                "constraint_pass": bool(constraint_ok),
                "objective_primary": objective_primary,
                "objective_secondary": objective_secondary,
                "selection_rank": np.nan,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "cost": metrics["cost"],
                "manual_review_rate": metrics["manual_review_rate"],
                "manual_review_count": metrics["manual_review_count"],
                "high_exposure_segment_recall": metrics["high_exposure_segment_recall"],
                "v2_reference_recall_validation": v2_metrics["recall"],
                "v2_reference_specificity_validation": v2_metrics["specificity"],
                "v2_reference_fp_validation": v2_metrics["fp"],
                "v2_reference_mr_rate_validation": v2_metrics["manual_review_rate"],
                "segment_policy_json": json.dumps(
                    {name: candidate.to_jsonable() for name, candidate in candidate_map.items()},
                    sort_keys=True,
                ),
                "segment_boundaries_source": "train_validation_non_test_pool",
                "policy_selection_split": "validation",
                "test_set_used_for_selection": False,
                "overfitting_risk": "medium_high_policy_complexity_segment_thresholds_selected_on_single_validation_split",
                "comment": comment,
            }
        )
    frame = pd.DataFrame(rows)
    frame["constraint_pass"] = frame["constraint_pass"].fillna(False)
    selection_pool = frame[frame["constraint_pass"]].copy()
    if selection_pool.empty:
        selection_pool = frame[frame["valid_candidate"]].copy()
    selection_pool = selection_pool.sort_values(
        ["objective_primary", "objective_secondary", "fp", "manual_review_rate", "specificity"],
        ascending=[True, True, True, True, False],
    )
    ranks = pd.Series(range(1, len(selection_pool) + 1), index=selection_pool.index)
    frame.loc[ranks.index, "selection_rank"] = ranks
    return frame.sort_values(["selection_rank", "constraint_pass", "cost"], na_position="last").head(300)


def _locked_test_rows(
    dataset: str,
    validation_results: pd.DataFrame,
    train_validation: pd.DataFrame,
    test: pd.DataFrame,
    test_score: np.ndarray,
) -> pd.DataFrame:
    """Apply validation-selected segment policies to locked test."""

    target = TARGETS[dataset]
    y_test = test[target].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    strategies = _segment_strategies(dataset, train_validation, test)
    for strategy, group in validation_results.groupby("segment_strategy"):
        selected = group[(group["valid_candidate"]) & (group["selection_rank"] == 1)]
        if selected.empty:
            continue
        row = selected.iloc[0]
        segment_values = strategies[strategy]
        policy_json = json.loads(row["segment_policy_json"])
        candidate_map = {
            segment_value: SegmentCandidate(
                segment_value=details["segment_value"],
                policy_type=details["policy_type"],
                threshold=details.get("threshold"),
                t_low=details.get("t_low"),
                t_high=details.get("t_high"),
                local_cost=np.nan,
                local_recall=np.nan,
                local_specificity=np.nan,
                local_fp=-1,
                local_fn=-1,
                local_mr_rate=np.nan,
            )
            for segment_value, details in policy_json.items()
        }
        masks = _combine_policy_masks(test_score, segment_values, candidate_map)
        high_mask = _high_exposure_mask(dataset, strategy, segment_values)
        metrics = _metrics(y_test, test_score, masks, high_mask)
        rows.append(
            {
                "dataset": dataset,
                "split": "locked_test",
                "segment_strategy": strategy,
                "locked_from_validation_candidate_id": row["candidate_id"],
                "validation_constraint_pass": bool(row["constraint_pass"]),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "cost": metrics["cost"],
                "manual_review_rate": metrics["manual_review_rate"],
                "manual_review_count": metrics["manual_review_count"],
                "high_exposure_segment_recall": metrics["high_exposure_segment_recall"],
                "segment_policy_json": row["segment_policy_json"],
                "segment_boundaries_source": "train_validation_non_test_pool",
                "policy_selection_split": "validation",
                "test_set_used_for_selection": False,
                "overfitting_risk": "medium_high_appendix_or_v3_candidate_until_independent_audit",
                "comment": "Locked test evaluation of a validation-selected segment policy; no test-set policy selection.",
            }
        )
    return pd.DataFrame(rows)


def _run_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run segment-aware policy search for one dataset."""

    train_validation, validation, test = _split_dataset(dataset)
    target = TARGETS[dataset]
    y_validation = validation[target].astype(int).to_numpy()
    validation_score = _load_score(dataset, validation)
    test_score = _load_score(dataset, test)
    v2_metrics = _v2_reference_metrics(dataset, y_validation, validation_score)
    validation_strategies = _segment_strategies(dataset, train_validation, validation)
    results = []
    for strategy, values in validation_strategies.items():
        results.append(_evaluate_segment_strategy(dataset, strategy, y_validation, validation_score, values, v2_metrics))
    validation_results = pd.concat(results, ignore_index=True, sort=False)
    locked_test = _locked_test_rows(dataset, validation_results, train_validation, test, test_score)
    return validation_results, locked_test


def _summary(taiwan_test: pd.DataFrame, heloc_test: pd.DataFrame) -> None:
    """Write concise Markdown summary."""

    def table(frame: pd.DataFrame) -> str:
        cols = ["segment_strategy", "precision", "recall", "specificity", "fp", "fn", "cost", "manual_review_rate", "comment"]
        shown = frame[cols].copy().sort_values(["cost", "fp"]).head(5)
        for col in ["precision", "recall", "specificity", "manual_review_rate"]:
            shown[col] = shown[col].map(lambda x: f"{float(x):.3f}")
        shown["cost"] = shown["cost"].map(lambda x: f"{float(x):.1f}")
        header = "| " + " | ".join(["Segment Strategy", "Precision", "Recall", "Specificity", "FP", "FN", "Cost", "MR Rate", "Comment"]) + " |"
        sep = "| " + " | ".join(["---"] * 9) + " |"
        body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.to_numpy()]
        return "\n".join([header, sep, *body])

    taiwan_best = taiwan_test.sort_values(["cost", "fp"]).iloc[0] if not taiwan_test.empty else None
    heloc_best = heloc_test.sort_values(["specificity", "cost"], ascending=[False, True]).iloc[0] if not heloc_test.empty else None
    taiwan_change = "PARTIALLY" if taiwan_best is not None and float(taiwan_best["recall"]) >= 0.575 else "NO"
    heloc_change = "PARTIALLY" if heloc_best is not None and float(heloc_best["specificity"]) >= 0.574 else "NO"
    candidate = "APPENDIX ONLY"
    if taiwan_best is not None and float(taiwan_best["recall"]) >= 0.575 and float(taiwan_best["manual_review_rate"]) <= 0.35:
        candidate = "APPENDIX ONLY / V3 AUDIT CANDIDATE"

    text = f"""# Segment-Aware Threshold Policy Summary

## Protocol
- Existing calibrated scores only.
- Segment boundaries are learned from the non-test train/validation pool.
- Segment policies are selected on validation only.
- Locked-test rows are evidence only and contain no rank/winner selection from test.
- Segment-aware thresholding has medium/high overfitting risk because multiple policies are selected from one validation split.

## Taiwan locked-test selected segment policies
{table(taiwan_test)}

## HELOC locked-test selected segment policies
{table(heloc_test)}

## Answers
1. Segment-aware threshold V2'den iyi mi? **Not enough to replace V2.** Some segment policies improve one dimension but remain higher-complexity V3/appendix candidates.
2. Taiwan recall arttı mı? **{taiwan_change}.** Compare locked-test recall rows above against V2 recall 0.575.
3. HELOC specificity arttı mı? **{heloc_change}.** Compare locked-test specificity rows above against V2 specificity 0.574.
4. MR rate kabul edilebilir mi? Policies are constrained to validation MR <= 0.35 for Taiwan and <= 0.30 for HELOC, but locked-test MR should still be audited.
5. Segment policy overfitting riski taşıyor mu? **YES.** Multiple segment-specific thresholds/bands are selected on a single validation split.
6. Rapor için kullanılabilir mi, yoksa appendix mi? **Appendix / V3 audit candidate.** It should not replace V2 without final audit.

## Candidate for V3
{candidate}
"""
    (OUT_DIR / "segment_policy_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    taiwan_validation, taiwan_test = _run_dataset("taiwan")
    heloc_validation, heloc_test = _run_dataset("heloc")
    taiwan_validation.to_csv(OUT_DIR / "segment_policy_validation_taiwan.csv", index=False)
    heloc_validation.to_csv(OUT_DIR / "segment_policy_validation_heloc.csv", index=False)
    taiwan_test.to_csv(OUT_DIR / "segment_policy_locked_test_taiwan.csv", index=False)
    heloc_test.to_csv(OUT_DIR / "segment_policy_locked_test_heloc.csv", index=False)
    _summary(taiwan_test, heloc_test)


if __name__ == "__main__":
    main()
