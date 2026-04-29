"""Bootstrap and paired statistical tests for model comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar


BOOTSTRAP_METRICS = ["roc_auc", "pr_auc", "recall", "f1", "brier_score", "expected_cost"]
HIGHER_IS_BETTER_METRICS = {"roc_auc", "pr_auc", "recall", "f1"}
LOWER_IS_BETTER_METRICS = {"brier_score", "expected_cost", "classification_error"}


@dataclass(frozen=True)
class ModelPrediction:
    """Per-row probabilities and decision threshold for one model."""

    model: str
    y_proba: np.ndarray
    threshold: float


def metric_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    threshold: float,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> float:
    """Compute one scalar metric from probabilities and a decision threshold."""

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    if metric == "roc_auc":
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_proba))
    if metric == "pr_auc":
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_proba))
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "brier_score":
        return float(brier_score_loss(y_true, y_proba))
    if metric == "expected_cost":
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return float(fn_cost * fn + fp_cost * fp)
    if metric == "classification_error":
        return float(np.mean(y_true != y_pred))
    raise ValueError(f"Unsupported metric: {metric}")


def bootstrap_metric_values(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str,
    threshold: float,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> np.ndarray:
    """Return bootstrap draws for one metric."""

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(random_seed)
    n_rows = len(y_true)
    values: list[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 5, n_bootstrap + 100)

    while len(values) < n_bootstrap and attempts < max_attempts:
        indices = rng.integers(0, n_rows, size=n_rows)
        value = metric_value(
            y_true[indices],
            y_proba[indices],
            metric=metric,
            threshold=threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        if np.isfinite(value):
            values.append(value)
        attempts += 1

    if not values:
        raise ValueError(f"No valid bootstrap samples for metric {metric!r}.")
    return np.asarray(values, dtype=float)


def confidence_interval(values: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float, float]:
    """Return mean and percentile confidence interval."""

    values = np.asarray(values, dtype=float)
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(values, alpha / 2.0))
    upper = float(np.quantile(values, 1.0 - alpha / 2.0))
    return float(np.mean(values)), lower, upper


def bootstrap_confidence_interval_row(
    dataset: str,
    prediction: ModelPrediction,
    y_true: np.ndarray,
    metric: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> tuple[dict, np.ndarray]:
    """Create one bootstrap confidence interval row and return raw draws."""

    values = bootstrap_metric_values(
        y_true,
        prediction.y_proba,
        metric=metric,
        threshold=prediction.threshold,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    mean, lower_ci, upper_ci = confidence_interval(values, confidence_level=confidence_level)
    return (
        {
            "dataset": dataset,
            "model": prediction.model,
            "metric": metric,
            "mean": mean,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "p_value": np.nan,
            "n_bootstrap": int(len(values)),
            "confidence_level": confidence_level,
            "threshold": prediction.threshold,
            "split": "test",
        },
        values,
    )


def paired_bootstrap_difference(
    y_true: np.ndarray,
    scre: ModelPrediction,
    comparator: ModelPrediction,
    metric: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> tuple[float, float, float, float, np.ndarray]:
    """Bootstrap paired metric difference: SCRE-Credit minus comparator."""

    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(random_seed)
    n_rows = len(y_true)
    differences: list[float] = []
    attempts = 0
    max_attempts = max(n_bootstrap * 5, n_bootstrap + 100)

    while len(differences) < n_bootstrap and attempts < max_attempts:
        indices = rng.integers(0, n_rows, size=n_rows)
        scre_value = metric_value(
            y_true[indices],
            scre.y_proba[indices],
            metric=metric,
            threshold=scre.threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        comparator_value = metric_value(
            y_true[indices],
            comparator.y_proba[indices],
            metric=metric,
            threshold=comparator.threshold,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        if np.isfinite(scre_value) and np.isfinite(comparator_value):
            differences.append(float(scre_value - comparator_value))
        attempts += 1

    if not differences:
        raise ValueError(f"No valid paired bootstrap samples for metric {metric!r}.")

    diff_array = np.asarray(differences, dtype=float)
    mean, lower_ci, upper_ci = confidence_interval(diff_array, confidence_level=confidence_level)
    p_value = float(2.0 * min(np.mean(diff_array <= 0.0), np.mean(diff_array >= 0.0)))
    return mean, lower_ci, upper_ci, min(p_value, 1.0), diff_array


def comparison_test_row(
    dataset: str,
    y_true: np.ndarray,
    scre: ModelPrediction,
    comparator: ModelPrediction,
    metric: str,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
    method: str = "paired_bootstrap",
) -> dict:
    """Create one model-comparison test row."""

    mean, lower_ci, upper_ci, p_value, _ = paired_bootstrap_difference(
        y_true=y_true,
        scre=scre,
        comparator=comparator,
        metric=metric,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        confidence_level=confidence_level,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )
    return {
        "dataset": dataset,
        "comparison": f"SCRE-Credit vs {comparator.model}",
        "method": method,
        "metric": f"{metric}_difference",
        "mean": mean,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "direction": "SCRE-Credit minus comparator",
        "split": "test",
        "scre_threshold": scre.threshold,
        "comparator_threshold": comparator.threshold,
    }


def mcnemar_test_row(
    dataset: str,
    y_true: np.ndarray,
    scre: ModelPrediction,
    comparator: ModelPrediction,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    confidence_level: float = 0.95,
) -> dict:
    """Return McNemar p-value plus bootstrap CI for classification-error difference."""

    y_true = np.asarray(y_true).astype(int)
    scre_pred = (scre.y_proba >= scre.threshold).astype(int)
    comparator_pred = (comparator.y_proba >= comparator.threshold).astype(int)
    scre_correct = scre_pred == y_true
    comparator_correct = comparator_pred == y_true
    table = [
        [int(np.sum(scre_correct & comparator_correct)), int(np.sum(scre_correct & ~comparator_correct))],
        [int(np.sum(~scre_correct & comparator_correct)), int(np.sum(~scre_correct & ~comparator_correct))],
    ]
    test_result = mcnemar(table, exact=False, correction=True)
    mean, lower_ci, upper_ci, _, _ = paired_bootstrap_difference(
        y_true=y_true,
        scre=scre,
        comparator=comparator,
        metric="classification_error",
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        confidence_level=confidence_level,
    )
    return {
        "dataset": dataset,
        "comparison": f"SCRE-Credit vs {comparator.model}",
        "method": "mcnemar",
        "metric": "classification_error_difference",
        "mean": mean,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "p_value": float(test_result.pvalue),
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "direction": "SCRE-Credit minus comparator",
        "split": "test",
        "scre_threshold": scre.threshold,
        "comparator_threshold": comparator.threshold,
        "mcnemar_both_correct": table[0][0],
        "mcnemar_scre_correct_comparator_wrong": table[0][1],
        "mcnemar_scre_wrong_comparator_correct": table[1][0],
        "mcnemar_both_wrong": table[1][1],
    }


def plot_bootstrap_metric_distributions(
    distributions: pd.DataFrame,
    output_path,
) -> None:
    """Save bootstrap metric distributions for SCRE-Credit across datasets."""

    metrics = BOOTSTRAP_METRICS
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        subset = distributions.loc[distributions["metric"] == metric]
        for dataset, dataset_df in subset.groupby("dataset"):
            ax.hist(dataset_df["value"], bins=30, alpha=0.45, density=True, label=dataset)
        ax.set_title(metric)
        ax.set_xlabel("Bootstrap value")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("SCRE-Credit Bootstrap Metric Distributions", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metric_direction(metric_name: str) -> str:
    """Return the optimization direction for a metric or metric difference."""

    base_metric = metric_name.replace("_difference", "")
    if base_metric in HIGHER_IS_BETTER_METRICS:
        return "higher_is_better"
    if base_metric in LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    return "unknown"


def clean_model_comparison_tests(comparison_table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add explicit metric direction, significance, and winner columns."""

    required = ["comparison", "metric", "mean", "lower_ci", "upper_ci", "p_value"]
    missing = [column for column in required if column not in comparison_table.columns]
    if missing:
        raise ValueError(f"Missing comparison-test columns: {missing}")

    table = comparison_table.copy()
    table["metric_direction"] = table["metric"].map(metric_direction)
    table["comparison_baseline"] = table["comparison"].astype(str).str.replace(
        "SCRE-Credit vs ",
        "",
        regex=False,
    )
    table["significant_at_0_05"] = pd.to_numeric(table["p_value"], errors="coerce") < alpha

    winner_by_mean = []
    winner_if_significant = []
    interpretation = []
    for row in table.itertuples(index=False):
        mean = float(row.mean)
        direction = row.metric_direction
        comparator = row.comparison_baseline
        if direction == "higher_is_better":
            mean_winner = "SCRE-Credit" if mean > 0 else comparator if mean < 0 else "tie"
        elif direction == "lower_is_better":
            mean_winner = "SCRE-Credit" if mean < 0 else comparator if mean > 0 else "tie"
        else:
            mean_winner = "undetermined"

        significant_winner = mean_winner if row.significant_at_0_05 and mean_winner != "tie" else "not_significant"
        winner_by_mean.append(mean_winner)
        winner_if_significant.append(significant_winner)
        interpretation.append(
            (
                f"{mean_winner} has the favorable mean direction for {row.metric}; "
                f"significant={bool(row.significant_at_0_05)}."
            )
        )

    table["winner_by_mean_direction"] = winner_by_mean
    table["winner_if_significant"] = winner_if_significant
    table["winner"] = table["winner_if_significant"]
    table["directional_interpretation"] = interpretation
    return table


__all__ = [
    "BOOTSTRAP_METRICS",
    "ModelPrediction",
    "bootstrap_confidence_interval_row",
    "bootstrap_metric_values",
    "comparison_test_row",
    "clean_model_comparison_tests",
    "metric_direction",
    "mcnemar_test_row",
    "metric_value",
    "paired_bootstrap_difference",
    "plot_bootstrap_metric_distributions",
]
