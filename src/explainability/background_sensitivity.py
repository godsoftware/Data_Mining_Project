"""SHAP background-set sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans

from src.explainability.shap_analysis import (
    fitted_model_from_pipeline,
    raw_feature_from_transformed,
    transform_pipeline_features,
)


BACKGROUND_ORDER = [
    "random_sample",
    "stratified_sample",
    "low_risk_sample",
    "high_risk_sample",
    "kmeans_summary",
]


@dataclass(frozen=True)
class BackgroundSet:
    """Named transformed background data used by SHAP."""

    name: str
    X_background_transformed: pd.DataFrame
    selection_policy: str


def stratified_background_sample(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample rows proportionally within target classes."""

    sample_size = min(sample_size, len(X))
    class_counts = y.value_counts(normalize=True).to_dict()
    pieces = []
    allocated = 0
    classes = sorted(class_counts)
    for class_value in classes:
        class_index = y.loc[y == class_value].index
        if class_value == classes[-1]:
            n_class = sample_size - allocated
        else:
            n_class = int(round(sample_size * class_counts[class_value]))
            allocated += n_class
        n_class = max(1, min(n_class, len(class_index)))
        pieces.append(X.loc[class_index].sample(n_class, random_state=random_seed + int(class_value)))
    return pd.concat(pieces).sample(frac=1.0, random_state=random_seed)


def risk_background_sample(
    X: pd.DataFrame,
    y_proba: np.ndarray,
    sample_size: int,
    high_risk: bool,
) -> pd.DataFrame:
    """Select lowest-risk or highest-risk rows according to model probability."""

    sample_size = min(sample_size, len(X))
    order = np.argsort(y_proba)
    if high_risk:
        order = order[::-1]
    return X.iloc[order[:sample_size]].copy()


def kmeans_background_summary(
    X_transformed: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
) -> pd.DataFrame:
    """Summarize transformed features with KMeans cluster centers."""

    n_clusters = min(n_clusters, len(X_transformed))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    centers = kmeans.fit(X_transformed).cluster_centers_
    return pd.DataFrame(centers, columns=X_transformed.columns)


def build_background_sets(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_size: int,
    kmeans_size: int,
    random_seed: int,
) -> list[BackgroundSet]:
    """Build all required SHAP background sets in transformed feature space."""

    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    transformed_train = transform_pipeline_features(pipeline, X_train)
    raw_sets = [
        (
            "random_sample",
            X_train.sample(min(sample_size, len(X_train)), random_state=random_seed),
            "uniform random sample from train split",
        ),
        (
            "stratified_sample",
            stratified_background_sample(X_train, y_train, sample_size, random_seed),
            "target-stratified sample from train split",
        ),
        (
            "low_risk_sample",
            risk_background_sample(X_train, y_proba_train, sample_size, high_risk=False),
            "lowest predicted-risk train rows",
        ),
        (
            "high_risk_sample",
            risk_background_sample(X_train, y_proba_train, sample_size, high_risk=True),
            "highest predicted-risk train rows",
        ),
    ]
    backgrounds = [
        BackgroundSet(
            name=name,
            X_background_transformed=transform_pipeline_features(pipeline, raw_background),
            selection_policy=policy,
        )
        for name, raw_background, policy in raw_sets
    ]
    backgrounds.append(
        BackgroundSet(
            name="kmeans_summary",
            X_background_transformed=kmeans_background_summary(transformed_train, kmeans_size, random_seed),
            selection_policy="KMeans cluster centers in transformed feature space",
        )
    )
    return backgrounds


def compute_background_shap_importance(
    pipeline,
    background: BackgroundSet,
    X_explain: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    """Compute global SHAP ranking for one background set."""

    import shap

    X_explain_transformed = transform_pipeline_features(pipeline, X_explain)
    model = fitted_model_from_pipeline(pipeline)
    explainer = shap.TreeExplainer(
        model,
        data=background.X_background_transformed,
        feature_perturbation="interventional",
    )
    try:
        shap_values = explainer(X_explain_transformed, check_additivity=False)
    except TypeError:
        shap_values = explainer(X_explain_transformed)

    values = np.asarray(getattr(shap_values, "values", shap_values))
    if values.ndim == 3:
        values = values[:, :, 1]
    feature_names = list(X_explain_transformed.columns)
    table = pd.DataFrame(
        {
            "background": background.name,
            "selection_policy": background.selection_policy,
            "feature": feature_names,
            "raw_feature": [raw_feature_from_transformed(feature, raw_features) for feature in feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "background_size": int(len(background.X_background_transformed)),
        }
    ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    table["rank"] = np.arange(1, len(table) + 1)
    table["is_top_10"] = table["rank"] <= 10
    return table.reset_index(drop=True)


def compute_all_background_importance(
    pipeline,
    backgrounds: list[BackgroundSet],
    X_explain: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    """Compute SHAP importance tables for every background set."""

    rows = []
    for background in backgrounds:
        print(f"Computing SHAP background sensitivity: {background.name}")
        rows.append(
            compute_background_shap_importance(
                pipeline=pipeline,
                background=background,
                X_explain=X_explain,
                raw_features=raw_features,
            )
        )
    return pd.concat(rows, ignore_index=True)


def pairwise_background_comparisons(
    importance: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare background rankings with overlap, correlation, and magnitude drift."""

    rows = []
    background_names = [name for name in BACKGROUND_ORDER if name in set(importance["background"])]
    rank_table = importance.pivot(index="feature", columns="background", values="rank")
    shap_table = importance.pivot(index="feature", columns="background", values="mean_abs_shap")

    for i, background_a in enumerate(background_names):
        for background_b in background_names[i + 1 :]:
            ranks_a = rank_table[background_a]
            ranks_b = rank_table[background_b]
            spearman_rho, spearman_p = spearmanr(ranks_a, ranks_b)
            kendall_tau, kendall_p = kendalltau(ranks_a, ranks_b)
            top_a = set(ranks_a.sort_values().head(top_k).index)
            top_b = set(ranks_b.sort_values().head(top_k).index)
            union = top_a | top_b
            drift = (shap_table[background_a] - shap_table[background_b]).abs()
            denominator = shap_table[background_a].replace(0, np.nan)
            relative_drift = (drift / denominator).replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "result_type": "background_comparison",
                    "background": f"{background_a} vs {background_b}",
                    "background_a": background_a,
                    "background_b": background_b,
                    "top_k": top_k,
                    "top_k_overlap_count": int(len(top_a & top_b)),
                    "top_k_overlap_rate": float(len(top_a & top_b) / top_k),
                    "top_k_jaccard": float(len(top_a & top_b) / len(union)) if union else np.nan,
                    "spearman_rho": float(spearman_rho),
                    "spearman_p_value": float(spearman_p),
                    "kendall_tau": float(kendall_tau),
                    "kendall_p_value": float(kendall_p),
                    "mean_abs_shap_drift": float(drift.mean()),
                    "median_abs_shap_drift": float(drift.median()),
                    "mean_relative_shap_drift": float(relative_drift.mean(skipna=True)),
                }
            )
    return pd.DataFrame(rows)


def combined_background_sensitivity_results(
    importance: pd.DataFrame,
    comparisons: pd.DataFrame,
    dataset: str,
    model: str,
) -> pd.DataFrame:
    """Combine feature-importance rows and pairwise-comparison rows for one CSV."""

    feature_rows = importance.copy()
    feature_rows.insert(0, "result_type", "feature_importance")
    feature_rows.insert(0, "model", model)
    feature_rows.insert(0, "dataset", dataset)
    comparison_rows = comparisons.copy()
    comparison_rows.insert(0, "model", model)
    comparison_rows.insert(0, "dataset", dataset)
    return pd.concat([feature_rows, comparison_rows], ignore_index=True, sort=False)


def plot_background_rank_shift(
    importance: pd.DataFrame,
    output_path: str | Path,
    reference_background: str = "random_sample",
    top_n: int = 15,
) -> None:
    """Plot rank shifts across background choices for reference top features."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_table = importance.pivot(index="feature", columns="background", values="rank")
    reference = importance.loc[importance["background"] == reference_background].sort_values("rank").head(top_n)
    selected = reference["feature"].tolist()
    backgrounds = [name for name in BACKGROUND_ORDER if name in rank_table.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in selected:
        ax.plot(backgrounds, rank_table.loc[feature, backgrounds], marker="o", linewidth=1.5, label=feature)
    ax.invert_yaxis()
    ax.set_ylabel("SHAP Rank (1 = most important)")
    ax.set_xlabel("Background Set")
    ax.set_title("SHAP Rank Shift Across Background Sets")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_background_shap_drift(
    importance: pd.DataFrame,
    output_path: str | Path,
    reference_background: str = "random_sample",
    top_n: int = 15,
) -> None:
    """Plot absolute mean-|SHAP| drift from a reference background."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap_table = importance.pivot(index="feature", columns="background", values="mean_abs_shap")
    reference = importance.loc[importance["background"] == reference_background].sort_values("rank").head(top_n)
    selected = reference["feature"].tolist()
    backgrounds = [name for name in BACKGROUND_ORDER if name in shap_table.columns and name != reference_background]
    drift = shap_table.loc[selected, backgrounds].subtract(shap_table.loc[selected, reference_background], axis=0).abs()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(selected))
    width = 0.8 / max(len(backgrounds), 1)
    for offset, background in enumerate(backgrounds):
        ax.bar(x + offset * width, drift[background], width=width, label=background)
    ax.set_xticks(x + width * (len(backgrounds) - 1) / 2)
    ax.set_xticklabels(selected, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Absolute Mean |SHAP| Drift")
    ax.set_title(f"SHAP Magnitude Drift vs {reference_background}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "BACKGROUND_ORDER",
    "BackgroundSet",
    "build_background_sets",
    "combined_background_sensitivity_results",
    "compute_all_background_importance",
    "compute_background_shap_importance",
    "kmeans_background_summary",
    "pairwise_background_comparisons",
    "plot_background_rank_shift",
    "plot_background_shap_drift",
    "risk_background_sample",
    "stratified_background_sample",
]
