"""Reliability, cost, subgroup, and statistical comparison utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config.settings import RANDOM_SEED
from evaluation import binary_classification_metrics


def bootstrap_metric_ci(
    y_true,
    y_proba,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Return percentile bootstrap confidence interval for a probability metric."""

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    values = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(float(metric_fn(y_true[idx], y_proba[idx])))

    if not values:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan}

    alpha = 1.0 - confidence
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, alpha / 2)),
        "upper": float(np.quantile(values, 1 - alpha / 2)),
    }


def metric_ci_table(
    y_true,
    y_proba,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create bootstrap CIs for ROC-AUC, PR-AUC, and Brier score."""

    metrics = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier_score": brier_score_loss,
    }
    rows = []
    for name, fn in metrics.items():
        ci = bootstrap_metric_ci(
            y_true,
            y_proba,
            fn,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        rows.append({"metric": name, **ci})
    return pd.DataFrame(rows)


def paired_bootstrap_metric_difference(
    y_true,
    proba_a,
    proba_b,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Paired bootstrap difference for two models on the same observations."""

    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    diffs = []

    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(float(metric_fn(y_true[idx], proba_a[idx]) - metric_fn(y_true[idx], proba_b[idx])))

    return {
        "mean_difference": float(np.mean(diffs)),
        "lower": float(np.quantile(diffs, 0.025)),
        "upper": float(np.quantile(diffs, 0.975)),
        "p_model_a_le_model_b": float(np.mean(np.asarray(diffs) <= 0)),
    }


def cost_curve_table(
    y_true,
    y_proba,
    fn_costs: list[float] | None = None,
    fp_cost: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evaluate expected cost over several FN cost assumptions."""

    if fn_costs is None:
        fn_costs = [1, 2, 3, 5, 10]
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    rows = []
    for fn_cost in fn_costs:
        for threshold in thresholds:
            metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
            rows.append(
                {
                    "fn_cost": float(fn_cost),
                    "fp_cost": float(fp_cost),
                    "threshold": float(threshold),
                    "expected_cost": fn_cost * metrics["fn"] + fp_cost * metrics["fp"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )
    return pd.DataFrame(rows).sort_values(["fn_cost", "expected_cost", "threshold"])


def manual_review_band_table(
    y_true,
    y_proba,
    threshold: float,
    band_widths: list[float] | None = None,
) -> pd.DataFrame:
    """Quantify reject/manual-review bands around a decision threshold."""

    if band_widths is None:
        band_widths = [0.02, 0.05, 0.10, 0.15]

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    rows = []

    for width in band_widths:
        lower = max(0.0, threshold - width)
        upper = min(1.0, threshold + width)
        reviewed = (y_proba >= lower) & (y_proba <= upper)
        auto = ~reviewed
        auto_metrics = (
            binary_classification_metrics(y_true[auto], y_proba[auto], threshold=threshold)
            if auto.sum() and len(np.unique(y_true[auto])) > 1
            else {}
        )
        rows.append(
            {
                "threshold": threshold,
                "band_width": width,
                "lower": lower,
                "upper": upper,
                "review_count": int(reviewed.sum()),
                "review_rate": float(reviewed.mean()),
                "review_default_rate": float(y_true[reviewed].mean()) if reviewed.sum() else np.nan,
                "auto_count": int(auto.sum()),
                "auto_recall": auto_metrics.get("recall", np.nan),
                "auto_precision": auto_metrics.get("precision", np.nan),
                "auto_f1": auto_metrics.get("f1", np.nan),
            }
        )
    return pd.DataFrame(rows)


def top_k_overlap_table(rankings: dict[str | int, list[str]], k: int = 10) -> pd.DataFrame:
    """Compute pairwise top-k Jaccard overlap across ranking runs."""

    rows = []
    keys = list(rankings)
    for i, key_a in enumerate(keys):
        set_a = set(rankings[key_a][:k])
        for key_b in keys[i + 1 :]:
            set_b = set(rankings[key_b][:k])
            union = set_a | set_b
            rows.append(
                {
                    "run_a": key_a,
                    "run_b": key_b,
                    "k": k,
                    "overlap_count": len(set_a & set_b),
                    "jaccard": len(set_a & set_b) / len(union) if union else np.nan,
                }
            )
    return pd.DataFrame(rows)


def kendalls_w(rankings: dict[str | int, list[str]]) -> float:
    """Compute Kendall's W agreement coefficient for ranked feature lists."""

    if len(rankings) < 2:
        return np.nan

    all_features = sorted({feature for ranking in rankings.values() for feature in ranking})
    n_items = len(all_features)
    n_raters = len(rankings)
    if n_items < 2:
        return np.nan

    default_rank = n_items + 1
    rank_matrix = []
    for ranking in rankings.values():
        rank_map = {feature: index + 1 for index, feature in enumerate(ranking)}
        rank_matrix.append([rank_map.get(feature, default_rank) for feature in all_features])

    rank_sums = np.asarray(rank_matrix).sum(axis=0)
    mean_rank_sum = np.mean(rank_sums)
    s_value = float(np.sum((rank_sums - mean_rank_sum) ** 2))
    denominator = (n_raters**2) * (n_items**3 - n_items)
    return float(12 * s_value / denominator) if denominator else np.nan


def subgroup_metric_table(
    X: pd.DataFrame,
    y_true,
    y_proba,
    subgroup_columns: list[str],
    threshold: float = 0.5,
    min_group_size: int = 100,
) -> pd.DataFrame:
    """Compute metrics by demographic or risk subgroups."""

    rows = []
    y_series = pd.Series(y_true, index=X.index)
    proba_series = pd.Series(y_proba, index=X.index)

    for column in subgroup_columns:
        if column not in X.columns:
            continue
        for value, group in X.groupby(column):
            if len(group) < min_group_size:
                continue
            y_group = y_series.loc[group.index]
            p_group = proba_series.loc[group.index]
            if y_group.nunique() < 2:
                continue
            metrics = binary_classification_metrics(y_group, p_group, threshold=threshold)
            rows.append(
                {
                    "subgroup_column": column,
                    "subgroup_value": value,
                    "count": int(len(group)),
                    "default_rate": float(y_group.mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
