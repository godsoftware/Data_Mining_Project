"""Temporal / pseudo-prospective robustness check.

The Taiwan and HELOC processed datasets do not contain real application dates.
Accordingly, this script does not perform real temporal validation or rolling
model retraining. It produces a transparent data-availability check and a
diagnostic-only pseudo/order-based robustness table on the locked-test split.
"""

from __future__ import annotations

import sys
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

OUT_DIR = PROJECT_ROOT / "outputs/final_weakness_closing/temporal_robustness"
TARGETS = {"taiwan": "default_next_month", "heloc": "bad_flag"}
DATA_PATHS = {
    "taiwan": PROJECT_ROOT / "data/processed/taiwan_model_ready.csv",
    "heloc": PROJECT_ROOT / "data/processed/heloc_model_ready.csv",
}
INTERIM_PATHS = {
    "taiwan": PROJECT_ROOT / "data/interim/taiwan_cleaned.csv",
    "heloc": PROJECT_ROOT / "data/interim/heloc_cleaned.csv",
}
DATE_PATTERNS = ["date", "time", "month", "year", "application", "origination", "opened", "as_of"]
RANDOM_SEEDS = [42, 123, 456, 789, 2026]
N_FOLDS = 5


def _split_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return canonical train/validation pool and locked test split."""

    target = TARGETS[dataset]
    frame = pd.read_csv(DATA_PATHS[dataset])
    train_validation, test = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame[target],
        random_state=42,
    )
    return train_validation.copy(), test.copy()


def _positive_probability(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability."""

    probability = estimator.predict_proba(x)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        return np.asarray(probability)[:, -1]
    return np.asarray(probability)[:, classes.index(1)]


def _load_score(dataset: str, frame: pd.DataFrame) -> np.ndarray:
    """Score a frame with the existing final V2 family model."""

    target = TARGETS[dataset]
    if dataset == "taiwan":
        model_name, calibration = "catboost", "isotonic"
    else:
        model_name, calibration = "woe_scorecard_logistic_regression", "sigmoid"
    path = PROJECT_ROOT / "outputs/models/calibrated" / dataset / f"{model_name}_{calibration}.joblib"
    estimator = joblib.load(path)
    return _positive_probability(estimator, frame.drop(columns=[target]))


def _policy_masks(dataset: str, score: np.ndarray) -> dict[str, np.ndarray]:
    """Apply fixed V2 manual-review policy."""

    if dataset == "taiwan":
        t_low, t_high = 0.14, 0.28
    else:
        t_low, t_high = 0.16, 0.39
    low = score < t_low
    high = score >= t_high
    manual = (~low) & (~high)
    return {"low": low, "manual": manual, "high": high}


def _ece(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (score >= lower) & (score < upper if upper < 1 else score <= upper)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(float(np.mean(score[mask])) - float(np.mean(y[mask])))
    return float(ece)


def _metrics(dataset: str, y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Compute fixed-policy performance metrics."""

    masks = _policy_masks(dataset, score)
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
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "cost": fp + 5 * low_fn + 0.5 * manual_count,
        "manual_review_rate": manual_count / len(y),
        "calibration_error": _ece(y, score),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "target_rate": float(np.mean(y)),
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
        "low_rate": float(np.mean(low)),
        "manual_rate": float(np.mean(manual)),
        "high_rate": float(np.mean(high)),
    }


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index with expected-bin quantiles."""

    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    cuts = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) <= 2:
        cuts = np.linspace(min(np.nanmin(expected), np.nanmin(actual)), max(np.nanmax(expected), np.nanmax(actual)), bins + 1)
    cuts[0], cuts[-1] = -np.inf, np.inf
    expected_counts = np.histogram(expected, bins=cuts)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=cuts)[0] / len(actual)
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)
    return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))


def _feature_columns(dataset: str) -> list[str]:
    if dataset == "taiwan":
        return [
            "LIMIT_BAL",
            "AGE",
            "PAY_0",
            "PAY_2",
            "recent_delay",
            "delay_count",
            "utilization_proxy",
            "payment_to_bill_ratio",
            "total_bill_amt",
        ]
    return [
        "ExternalRiskEstimate",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "PercentTradesNeverDelq",
        "NetFractionRevolvingBurden",
        "revolving_burden_proxy",
        "delinquency_intensity",
        "credit_history_length_proxy",
    ]


def _decision_stability(metrics: dict[str, float], overall_metrics: dict[str, float]) -> float:
    """Measure similarity of low/manual/high decision mix to full locked-test mix."""

    distance = (
        abs(metrics["low_rate"] - overall_metrics["low_rate"])
        + abs(metrics["manual_rate"] - overall_metrics["manual_rate"])
        + abs(metrics["high_rate"] - overall_metrics["high_rate"])
    ) / 2
    return float(max(0.0, 1.0 - distance))


def _pseudo_rows_for_dataset(dataset: str) -> pd.DataFrame:
    """Build locked-test pseudo/order and random-partition robustness rows."""

    train_validation, test = _split_dataset(dataset)
    target = TARGETS[dataset]
    test = test.sort_index().copy()
    y = test[target].astype(int).to_numpy()
    score = _load_score(dataset, test)
    overall = _metrics(dataset, y, score)
    rows: list[dict[str, Any]] = []

    fold_ids = np.array_split(np.arange(len(test)), N_FOLDS)
    baseline_idx = fold_ids[0]
    baseline_score = score[baseline_idx]
    baseline_frame = test.iloc[baseline_idx]
    feature_cols = [c for c in _feature_columns(dataset) if c in test.columns]

    for fold_number, idx in enumerate(fold_ids, start=1):
        fold_y = y[idx]
        fold_score = score[idx]
        metrics = _metrics(dataset, fold_y, fold_score)
        fold_frame = test.iloc[idx]
        feature_psi_values = [
            _psi(baseline_frame[col].to_numpy(), fold_frame[col].to_numpy())
            for col in feature_cols
        ]
        rows.append(
            {
                "dataset": dataset,
                "analysis_type": "row_order_locked_test_diagnostic",
                "fold": fold_number,
                "seed": np.nan,
                "n": len(idx),
                **metrics,
                "score_psi_vs_first_order_block": _psi(baseline_score, fold_score),
                "mean_feature_psi_vs_first_order_block": float(np.nanmean(feature_psi_values)) if feature_psi_values else np.nan,
                "max_feature_psi_vs_first_order_block": float(np.nanmax(feature_psi_values)) if feature_psi_values else np.nan,
                "decision_stability": _decision_stability(metrics, overall),
                "real_temporal_validation": False,
                "use_in_report_as": "LIMITATION_ONLY_OR_APPENDIX_DIAGNOSTIC",
                "limitation": "Row order is not verified application time; this is not temporal validation.",
            }
        )

    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(np.arange(len(test)))
        for fold_number, idx in enumerate(np.array_split(shuffled, N_FOLDS), start=1):
            fold_y = y[idx]
            fold_score = score[idx]
            metrics = _metrics(dataset, fold_y, fold_score)
            rows.append(
                {
                    "dataset": dataset,
                    "analysis_type": "random_locked_test_partition_diagnostic",
                    "fold": fold_number,
                    "seed": seed,
                    "n": len(idx),
                    **metrics,
                    "score_psi_vs_first_order_block": np.nan,
                    "mean_feature_psi_vs_first_order_block": np.nan,
                    "max_feature_psi_vs_first_order_block": np.nan,
                    "decision_stability": _decision_stability(metrics, overall),
                    "real_temporal_validation": False,
                    "use_in_report_as": "LIMITATION_ONLY_OR_APPENDIX_DIAGNOSTIC",
                    "limitation": "Random partitions estimate locked-test metric variability, not temporal deployment robustness.",
                }
            )
    return pd.DataFrame(rows)


def _date_availability() -> pd.DataFrame:
    """Inspect processed/interim/raw columns for candidate temporal fields."""

    rows: list[dict[str, Any]] = []
    for dataset in ["taiwan", "heloc"]:
        for source_name, path in [
            ("processed", DATA_PATHS[dataset]),
            ("interim", INTERIM_PATHS[dataset]),
        ]:
            frame = pd.read_csv(path, nrows=5)
            cols = list(frame.columns)
            target = TARGETS[dataset]
            candidates = [
                c
                for c in cols
                if c != target
                and "default" not in c.lower()
                and "bad_flag" not in c.lower()
                and any(pattern in c.lower() for pattern in DATE_PATTERNS)
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "source": source_name,
                    "path": str(path),
                    "columns_checked": len(cols),
                    "candidate_time_columns": "; ".join(candidates),
                    "real_time_available": bool(candidates),
                    "note": "Candidate column name only; semantic validity still requires source documentation.",
                }
            )
    return pd.DataFrame(rows)


def _write_availability_markdown(availability: pd.DataFrame) -> None:
    """Write temporal data availability check."""

    real_possible = bool(availability["real_time_available"].any())
    candidate_text = availability[availability["real_time_available"]].to_string(index=False) if real_possible else "No date/month/application-time columns were found in processed or interim files."
    text = f"""# Temporal Data Availability Check

## Result
Real temporal validation possible: **{'YES' if real_possible else 'NO'}**

## Evidence
{candidate_text}

## Interpretation
- Taiwan processed data has no true date/month/application-time variable. The interim `ID` is a record identifier and must not be treated as calendar time.
- HELOC processed/interim data has no true date/month/application-time variable.
- Therefore rolling-forward temporal validation is not methodologically valid here.

## Safe wording
Because the datasets do not contain verified application timestamps, this project cannot claim real temporal deployment validation. The robustness check is limited to pseudo/order-based and random-partition diagnostics on existing locked-test evidence.
"""
    (OUT_DIR / "temporal_data_availability_check.md").write_text(text, encoding="utf-8")


def _summary(pseudo: pd.DataFrame, availability: pd.DataFrame) -> None:
    """Write final summary."""

    def dataset_summary(dataset: str) -> str:
        subset = pseudo[pseudo["dataset"] == dataset]
        row_order = subset[subset["analysis_type"] == "row_order_locked_test_diagnostic"]
        random = subset[subset["analysis_type"] == "random_locked_test_partition_diagnostic"]
        if row_order.empty:
            return "No pseudo rows generated."
        return (
            f"Row-order recall range={row_order['recall'].min():.3f}-{row_order['recall'].max():.3f}, "
            f"specificity range={row_order['specificity'].min():.3f}-{row_order['specificity'].max():.3f}, "
            f"PR-AUC range={row_order['pr_auc'].min():.3f}-{row_order['pr_auc'].max():.3f}, "
            f"mean score PSI={row_order['score_psi_vs_first_order_block'].mean():.3f}; "
            f"random-partition recall sd={random['recall'].std():.3f}, specificity sd={random['specificity'].std():.3f}."
        )

    real_possible = bool(availability["real_time_available"].any())
    text = f"""# Temporal Robustness Summary

## 1. Gerçek zamanlı validation mümkün mü?
**{'YES' if real_possible else 'NO'}**. No verified timestamp/application-date variable is available for Taiwan or HELOC.

## 2. Mümkünse sonuçlar stabil mi?
Not applicable. Rolling-forward validation was not run because there is no real time variable.

## 3. Değilse bunu nasıl limitation olarak yazmalıyız?
Safe wording: *The datasets do not include verified application timestamps, so the project cannot claim true temporal or prospective deployment validation. We report only pseudo/order-based locked-test diagnostics and random-partition variability checks as limitations/appendix evidence.*

## 4. Pseudo/order-based diagnostic
- Taiwan: {dataset_summary('taiwan')}
- HELOC: {dataset_summary('heloc')}

## 5. Real deployment eksikliği ne kadar kapanıyor?
Only slightly. These diagnostics check whether locked-test performance varies across row-order blocks and random partitions, but they do not replace real production-time validation.

## 6. Bu analiz ana sonuç mu, appendix mi?
**LIMITATION ONLY / APPENDIX DIAGNOSTIC.** It should not be used as main evidence of temporal robustness.
"""
    (OUT_DIR / "temporal_robustness_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    availability = _date_availability()
    _write_availability_markdown(availability)
    pseudo = pd.concat([_pseudo_rows_for_dataset("taiwan"), _pseudo_rows_for_dataset("heloc")], ignore_index=True)
    pseudo.to_csv(OUT_DIR / "pseudo_temporal_robustness.csv", index=False)
    _summary(pseudo, availability)


if __name__ == "__main__":
    main()
