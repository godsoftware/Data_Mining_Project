"""Subgroup reliability analysis utilities.

These helpers audit model behavior across clinically or operationally relevant
subgroups. They are not a legal fairness proof.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


AGE_GROUP_COLUMN = "AGE_GROUP"
SUBGROUP_RELIABILITY_NOTE = "subgroup reliability analysis, not legal fairness proof"
TAIWAN_SUBGROUP_VALUE_LABELS = {
    "SEX": {"1": "male", "2": "female"},
    "EDUCATION": {
        "1": "graduate_school",
        "2": "university",
        "3": "high_school",
        "4": "other",
    },
    "MARRIAGE": {"1": "married", "2": "single", "3": "other"},
}


def add_taiwan_age_group(X: pd.DataFrame, age_column: str = "AGE") -> pd.DataFrame:
    """Return a copy of X with Taiwan AGE_GROUP labels added."""

    if age_column not in X.columns:
        raise KeyError(f"Missing age column required for AGE_GROUP: {age_column}")

    result = X.copy()
    age = pd.to_numeric(result[age_column], errors="coerce")
    result[AGE_GROUP_COLUMN] = np.select(
        [
            age < 30,
            (age >= 30) & (age < 40),
            (age >= 40) & (age < 50),
            age >= 50,
        ],
        ["<30", "30-39", "40-49", "50+"],
        default="missing",
    )
    return result


def expected_cost_from_counts(
    false_negative: int,
    false_positive: int,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> float:
    """Compute expected cost from confusion-matrix counts."""

    return float(fn_cost * false_negative + fp_cost * false_positive)


def subgroup_display_label(column: str, value: object) -> str:
    """Return a human-readable subgroup label while preserving the raw value."""

    raw_value = str(value)
    mapped_value = TAIWAN_SUBGROUP_VALUE_LABELS.get(column, {}).get(raw_value)
    if mapped_value:
        return f"{column}={raw_value} ({mapped_value})"
    return f"{column}={raw_value}"


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""

    y_array = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        bin_weight = float(np.mean(mask))
        confidence = float(np.mean(proba[mask]))
        observed = float(np.mean(y_array[mask]))
        ece += bin_weight * abs(observed - confidence)
    return float(ece)


def subgroup_metric_row(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Compute reliability metrics for one subgroup."""

    y_array = np.asarray(y_true).astype(int)
    proba = np.asarray(y_proba, dtype=float)
    if len(y_array) != len(proba):
        raise ValueError("y_true and y_proba must have the same length.")
    if len(y_array) == 0:
        raise ValueError("Cannot compute subgroup metrics for an empty group.")

    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, y_pred, labels=[0, 1]).ravel()
    positives = int(tp + fn)
    negatives = int(tn + fp)

    roc_auc = float(roc_auc_score(y_array, proba)) if len(np.unique(y_array)) > 1 else np.nan
    pr_auc = float(average_precision_score(y_array, proba)) if positives else np.nan
    recall = float(recall_score(y_array, y_pred, zero_division=0))
    precision = float(precision_score(y_array, y_pred, zero_division=0))
    f1 = float(f1_score(y_array, y_pred, zero_division=0))
    false_positive_rate = float(fp / negatives) if negatives else np.nan
    false_negative_rate = float(fn / positives) if positives else np.nan
    brier = float(brier_score_loss(y_array, proba))
    ece = expected_calibration_error(y_array, proba, n_bins=n_bins)
    expected_cost = expected_cost_from_counts(fn, fp, fn_cost=fn_cost, fp_cost=fp_cost)

    return {
        "sample_size": int(len(y_array)),
        "n_default": positives,
        "n_non_default": negatives,
        "default_rate": float(np.mean(y_array)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "brier_score": brier,
        "ece": ece,
        "expected_cost": expected_cost,
        "expected_cost_per_1000": float(expected_cost / len(y_array) * 1000.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def subgroup_reliability_table(
    X: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    subgroup_columns: list[str],
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    min_group_size: int = 1,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability metrics for each value of each subgroup column."""

    if len(X) != len(y_true) or len(X) != len(y_proba):
        raise ValueError("X, y_true, and y_proba must have the same number of rows.")

    X_indexed = X.reset_index(drop=True)
    y_series = pd.Series(np.asarray(y_true).astype(int), index=X_indexed.index)
    proba_series = pd.Series(np.asarray(y_proba, dtype=float), index=X_indexed.index)
    rows: list[dict[str, object]] = []

    for column in subgroup_columns:
        if column not in X_indexed.columns:
            raise KeyError(f"Missing subgroup column: {column}")
        group_values = X_indexed[column].astype("object").where(X_indexed[column].notna(), "missing")
        for value in sorted(group_values.unique(), key=lambda item: str(item)):
            mask = group_values == value
            if int(mask.sum()) < min_group_size:
                continue
            metrics = subgroup_metric_row(
                y_series.loc[mask],
                proba_series.loc[mask],
                threshold=threshold,
                fn_cost=fn_cost,
                fp_cost=fp_cost,
                n_bins=n_bins,
            )
            rows.append(
                {
                    "analysis_type": "subgroup_reliability",
                    "analysis_note": SUBGROUP_RELIABILITY_NOTE,
                    "subgroup_column": column,
                    "subgroup_value": str(value),
                    "subgroup_label": subgroup_display_label(column, value),
                    "threshold": float(threshold),
                    "fn_cost": float(fn_cost),
                    "fp_cost": float(fp_cost),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def plot_subgroup_recall_fnr(table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot subgroup recall and FNR side by side."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values(["subgroup_column", "subgroup_value"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(ordered) * 0.55), 5.5))
    ax.bar(x - width / 2, ordered["recall"], width=width, label="Recall", color="#2f6fbb")
    ax.bar(x + width / 2, ordered["fnr"], width=width, label="FNR", color="#b9473f")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Subgroup Reliability: Recall and False Negative Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["subgroup_label"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_subgroup_calibration(table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot subgroup Brier score and ECE."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values(["subgroup_column", "subgroup_value"]).reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(ordered) * 0.55), 5.5))
    ax.bar(x - width / 2, ordered["brier_score"], width=width, label="Brier score", color="#437f6d")
    ax.bar(x + width / 2, ordered["ece"], width=width, label="ECE", color="#c88f2d")
    ax.set_ylabel("Calibration metric")
    ax.set_title("Subgroup Reliability: Calibration Error")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["subgroup_label"], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_subgroup_cost(table: pd.DataFrame, output_path: str | Path) -> None:
    """Plot expected cost per 1000 observations for each subgroup."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.sort_values("expected_cost_per_1000", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(9, len(ordered) * 0.55), 5.5))
    ax.bar(ordered["subgroup_label"], ordered["expected_cost_per_1000"], color="#6b5aa6")
    ax.set_ylabel("Expected cost per 1000")
    ax.set_title("Subgroup Reliability: Expected Cost")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


__all__ = [
    "AGE_GROUP_COLUMN",
    "SUBGROUP_RELIABILITY_NOTE",
    "TAIWAN_SUBGROUP_VALUE_LABELS",
    "add_taiwan_age_group",
    "expected_calibration_error",
    "expected_cost_from_counts",
    "plot_subgroup_calibration",
    "plot_subgroup_cost",
    "plot_subgroup_recall_fnr",
    "subgroup_display_label",
    "subgroup_metric_row",
    "subgroup_reliability_table",
]
