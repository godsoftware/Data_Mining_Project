"""Faithfulness sanity checks for model explanations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score


TAIWAN_FEATURE_GROUPS = {
    "PAY_variables": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    "engineered_delay_variables": [
        "delay_count",
        "severe_delay_count",
        "max_delay",
        "avg_delay",
        "recent_delay",
    ],
    "BILL_variables": [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "total_bill_amt",
        "avg_bill_amt",
        "max_bill_amt",
        "bill_trend",
        "bill_volatility",
        "utilization_proxy",
    ],
    "PAY_AMT_variables": [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "total_pay_amt",
        "avg_pay_amt",
        "max_pay_amt",
        "payment_to_bill_ratio",
        "recent_payment_intensity",
        "payment_volatility",
    ],
    "demographic_variables": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
}


def baseline_feature_values(X_train: pd.DataFrame, categorical_columns: list[str] | None = None) -> dict[str, float]:
    """Return train-derived replacement values for perturbation checks."""

    categorical = set(categorical_columns or [])
    values: dict[str, float] = {}
    for column in X_train.columns:
        series = pd.to_numeric(X_train[column], errors="coerce")
        if column in categorical:
            mode = series.dropna().mode()
            values[column] = float(mode.iloc[0]) if not mode.empty else 0.0
        else:
            median = series.median()
            values[column] = float(median) if pd.notna(median) else 0.0
    return values


def replace_features_with_baseline(
    X: pd.DataFrame,
    features: list[str],
    replacements: dict[str, float],
) -> pd.DataFrame:
    """Return a copy of X with selected features replaced by train baseline values."""

    perturbed = X.copy()
    for feature in features:
        if feature in perturbed.columns:
            perturbed[feature] = replacements[feature]
    return perturbed


def baseline_frame_like(X: pd.DataFrame, replacements: dict[str, float]) -> pd.DataFrame:
    """Create a baseline-only frame with every feature set to its train replacement value."""

    baseline = pd.DataFrame(index=X.index)
    for column in X.columns:
        baseline[column] = replacements[column]
    return baseline


def restore_features(
    baseline: pd.DataFrame,
    original: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Restore selected features from the original data into a baseline frame."""

    restored = baseline.copy()
    for feature in features:
        if feature in restored.columns:
            restored[feature] = original[feature]
    return restored


def expected_cost(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> float:
    """Compute expected cost from probability decisions."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    return float(fn_cost * false_negative + fp_cost * false_positive)


def faithfulness_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict[str, float]:
    """Compute the metrics used in faithfulness comparisons."""

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "expected_cost": expected_cost(y_true, y_proba, threshold, fn_cost=fn_cost, fp_cost=fp_cost),
    }


def compare_to_baseline_metrics(
    baseline_metrics: dict[str, float],
    baseline_proba: np.ndarray,
    perturbed_metrics: dict[str, float],
    perturbed_proba: np.ndarray,
) -> dict[str, float]:
    """Return drop/change metrics relative to the unperturbed model behavior."""

    return {
        "roc_auc_drop": baseline_metrics["roc_auc"] - perturbed_metrics["roc_auc"],
        "pr_auc_drop": baseline_metrics["pr_auc"] - perturbed_metrics["pr_auc"],
        "probability_shift": float(np.mean(np.abs(baseline_proba - perturbed_proba))),
        "recall_change": perturbed_metrics["recall"] - baseline_metrics["recall"],
        "cost_change": perturbed_metrics["expected_cost"] - baseline_metrics["expected_cost"],
        "perturbed_roc_auc": perturbed_metrics["roc_auc"],
        "perturbed_pr_auc": perturbed_metrics["pr_auc"],
        "perturbed_recall": perturbed_metrics["recall"],
        "perturbed_expected_cost": perturbed_metrics["expected_cost"],
    }


def evaluate_feature_set_perturbation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    replacements: dict[str, float],
    threshold: float,
    baseline_proba: np.ndarray,
    baseline_metrics: dict[str, float],
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict[str, float]:
    """Replace selected features and compare perturbed behavior with baseline."""

    perturbed = replace_features_with_baseline(X, features, replacements)
    perturbed_proba = model.predict_proba(perturbed)[:, 1]
    perturbed_metrics = faithfulness_metrics(y, perturbed_proba, threshold, fn_cost=fn_cost, fp_cost=fp_cost)
    return compare_to_baseline_metrics(
        baseline_metrics=baseline_metrics,
        baseline_proba=baseline_proba,
        perturbed_metrics=perturbed_metrics,
        perturbed_proba=perturbed_proba,
    )


def single_feature_perturbation_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    replacements: dict[str, float],
    threshold: float,
    baseline_proba: np.ndarray,
    baseline_metrics: dict[str, float],
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate single-feature replacement faithfulness checks."""

    rows = []
    for rank, feature in enumerate(features, start=1):
        if feature not in X.columns:
            continue
        row = evaluate_feature_set_perturbation(
            model,
            X,
            y,
            [feature],
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        rows.append(
            {
                "test_type": "single_feature_perturbation",
                "step": rank,
                "feature": feature,
                "feature_set": feature,
                "n_features": 1,
                **row,
            }
        )
    return pd.DataFrame(rows)


def topk_deletion_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    replacements: dict[str, float],
    threshold: float,
    baseline_proba: np.ndarray,
    baseline_metrics: dict[str, float],
    max_k: int = 10,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate cumulative top-k deletion."""

    rows = []
    for k in range(1, min(max_k, len(ranked_features)) + 1):
        selected = ranked_features[:k]
        row = evaluate_feature_set_perturbation(
            model,
            X,
            y,
            selected,
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        rows.append(
            {
                "test_type": "topk_deletion",
                "step": k,
                "feature": selected[-1],
                "feature_set": ", ".join(selected),
                "n_features": len(selected),
                **row,
            }
        )
    return pd.DataFrame(rows)


def topk_insertion_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    ranked_features: list[str],
    replacements: dict[str, float],
    threshold: float,
    baseline_proba: np.ndarray,
    baseline_metrics: dict[str, float],
    max_k: int = 10,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate cumulative top-k insertion into a baseline-only frame."""

    rows = []
    empty_frame = baseline_frame_like(X, replacements)
    for k in range(1, min(max_k, len(ranked_features)) + 1):
        selected = ranked_features[:k]
        inserted = restore_features(empty_frame, X, selected)
        inserted_proba = model.predict_proba(inserted)[:, 1]
        inserted_metrics = faithfulness_metrics(y, inserted_proba, threshold, fn_cost=fn_cost, fp_cost=fp_cost)
        row = compare_to_baseline_metrics(
            baseline_metrics=baseline_metrics,
            baseline_proba=baseline_proba,
            perturbed_metrics=inserted_metrics,
            perturbed_proba=inserted_proba,
        )
        rows.append(
            {
                "test_type": "topk_insertion",
                "step": k,
                "feature": selected[-1],
                "feature_set": ", ".join(selected),
                "n_features": len(selected),
                **row,
            }
        )
    return pd.DataFrame(rows)


def random_feature_deletion_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    candidate_features: list[str],
    replacements: dict[str, float],
    threshold: float,
    baseline_proba: np.ndarray,
    baseline_metrics: dict[str, float],
    random_seed: int = 42,
    max_k: int = 10,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate cumulative random deletion as a sanity baseline."""

    rng = np.random.default_rng(random_seed)
    available = [feature for feature in candidate_features if feature in X.columns]
    random_order = list(rng.permutation(available))[:max_k]
    table = topk_deletion_table(
        model,
        X,
        y,
        random_order,
        replacements,
        threshold,
        baseline_proba,
        baseline_metrics,
        max_k=max_k,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    table["test_type"] = "random_feature_deletion_baseline"
    return table


def group_deletion_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_groups: dict[str, list[str]],
    replacements: dict[str, float],
    threshold: float,
    baseline_proba: np.ndarray,
    baseline_metrics: dict[str, float],
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate correlated feature-group deletion checks."""

    rows = []
    for group_name, features in feature_groups.items():
        present_features = [feature for feature in features if feature in X.columns]
        if not present_features:
            continue
        row = evaluate_feature_set_perturbation(
            model,
            X,
            y,
            present_features,
            replacements,
            threshold,
            baseline_proba,
            baseline_metrics,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        rows.append(
            {
                "test_type": "correlated_feature_group_deletion",
                "group": group_name,
                "feature_set": ", ".join(present_features),
                "n_features": len(present_features),
                **row,
            }
        )
    return pd.DataFrame(rows)


def plot_deletion_curve(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot ROC-AUC and PR-AUC drop for top-k deletion versus random deletion."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for test_type, label in [
        ("topk_deletion", "Top-k deletion"),
        ("random_feature_deletion_baseline", "Random deletion"),
    ]:
        subset = results.loc[results["test_type"] == test_type].sort_values("step")
        if subset.empty:
            continue
        axes[0].plot(subset["step"], subset["roc_auc_drop"], marker="o", label=label)
        axes[1].plot(subset["step"], subset["pr_auc_drop"], marker="o", label=label)
    axes[0].set_title("Deletion ROC-AUC Drop")
    axes[1].set_title("Deletion PR-AUC Drop")
    for ax in axes:
        ax.set_xlabel("Number of Deleted Features")
        ax.set_ylabel("Drop vs Full Model")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_insertion_curve(results: pd.DataFrame, output_path: str | Path) -> None:
    """Plot top-k insertion recovery curve."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = results.loc[results["test_type"] == "topk_insertion"].sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(subset["step"], subset["perturbed_roc_auc"], marker="o", color="#2f6f73")
    axes[1].plot(subset["step"], subset["perturbed_pr_auc"], marker="o", color="#7b5ea7")
    axes[0].set_title("Insertion ROC-AUC")
    axes[1].set_title("Insertion PR-AUC")
    for ax in axes:
        ax.set_xlabel("Number of Restored Top Features")
        ax.set_ylabel("Metric Value")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_auc_drop_by_feature(results: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    """Plot single-feature ROC-AUC drop."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = (
        results.loc[results["test_type"] == "single_feature_perturbation"]
        .sort_values("roc_auc_drop", ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(subset["feature"][::-1], subset["roc_auc_drop"][::-1], color="#b84a39")
    ax.set_xlabel("ROC-AUC Drop")
    ax.set_title("Single Feature Perturbation Faithfulness")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "TAIWAN_FEATURE_GROUPS",
    "baseline_feature_values",
    "faithfulness_metrics",
    "group_deletion_table",
    "plot_auc_drop_by_feature",
    "plot_deletion_curve",
    "plot_insertion_curve",
    "random_feature_deletion_table",
    "single_feature_perturbation_table",
    "topk_deletion_table",
    "topk_insertion_table",
]
