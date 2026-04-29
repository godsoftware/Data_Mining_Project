"""SHAP rank-stability experiments across random seeds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau, spearmanr

from scre_credit import make_model_pipeline
from src.explainability.shap_analysis import (
    fitted_model_from_pipeline,
    raw_feature_from_transformed,
    transform_pipeline_features,
)

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


@dataclass(frozen=True)
class StabilityDatasetConfig:
    """Dataset configuration for a SHAP stability run."""

    dataset: str
    scope: str
    target: str
    categorical_columns: list[str]
    feature_subset: list[str] | None = None


def make_stability_lightgbm(random_seed: int):
    """Create the LightGBM final-family model used for rank stability."""

    if LGBMClassifier is None:
        raise ImportError("lightgbm is required for SHAP stability.")
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_seed,
        n_jobs=-1,
        verbose=-1,
    )


def _positive_class_shap_values(shap_values) -> np.ndarray:
    """Return positive-class SHAP values as a 2D array."""

    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values, dtype=float)


def shap_importance_for_seed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_explain: pd.DataFrame,
    categorical_columns: list[str],
    random_seed: int,
) -> pd.DataFrame:
    """Train one LightGBM pipeline and return full SHAP feature ranking."""

    import shap

    model = make_stability_lightgbm(random_seed)
    pipeline = make_model_pipeline(model, X_train, categorical_columns=categorical_columns)
    pipeline.fit(X_train, y_train)
    X_transformed = transform_pipeline_features(pipeline, X_explain)
    final_model = fitted_model_from_pipeline(pipeline)
    explainer = shap.TreeExplainer(final_model)
    try:
        shap_values = explainer(X_transformed, check_additivity=False)
    except TypeError:
        shap_values = explainer(X_transformed)

    values = _positive_class_shap_values(shap_values)
    feature_names = list(X_transformed.columns)
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance["shap_sign"] = np.sign(importance["mean_shap"])
    return importance.reset_index(drop=True)


def run_seed_stability(
    df: pd.DataFrame,
    config: StabilityDatasetConfig,
    seeds: list[int],
    explain_rows: int,
    base_random_seed: int,
) -> pd.DataFrame:
    """Run SHAP rank extraction for all seeds for one dataset."""

    from data.split_leakage import stratified_train_validation_test_split

    if config.feature_subset:
        columns = [column for column in config.feature_subset if column in df.columns]
        missing = sorted(set(config.feature_subset) - set(columns))
        if missing:
            raise KeyError(f"Missing reduced-stability feature columns for {config.dataset}: {missing}")
        df = df[columns + [config.target]].copy()

    split = stratified_train_validation_test_split(df, target_column=config.target, random_state=base_random_seed)
    X_train = split.train.drop(columns=[config.target])
    y_train = split.train[config.target]
    X_explain = split.validation.drop(columns=[config.target])
    X_explain = X_explain.sample(min(explain_rows, len(X_explain)), random_state=base_random_seed)
    categorical_columns = [column for column in config.categorical_columns if column in X_train.columns]

    rows = []
    raw_features = list(X_train.columns)
    for seed_index, seed in enumerate(seeds, start=1):
        print(f"{config.dataset} {config.scope} SHAP stability seed {seed_index}/{len(seeds)}: {seed}")
        importance = shap_importance_for_seed(
            X_train=X_train,
            y_train=y_train,
            X_explain=X_explain,
            categorical_columns=categorical_columns,
            random_seed=seed,
        )
        importance.insert(0, "random_seed", seed)
        importance.insert(0, "seed_index", seed_index)
        importance.insert(0, "model_family", "lightgbm")
        importance.insert(0, "stability_scope", config.scope)
        importance.insert(0, "dataset", config.dataset)
        importance["raw_feature"] = importance["feature"].map(
            lambda feature: raw_feature_from_transformed(feature, raw_features)
        )
        importance["in_top_5"] = importance["rank"] <= 5
        importance["in_top_10"] = importance["rank"] <= 10
        rows.append(importance)

    return pd.concat(rows, ignore_index=True)


def _rank_matrix(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Create seed-by-feature rank matrix."""

    return seed_results.pivot_table(
        index="random_seed",
        columns="feature",
        values="rank",
        aggfunc="first",
    )


def kendalls_w(rank_matrix: pd.DataFrame) -> float:
    """Compute Kendall's coefficient of concordance W from a seed-by-feature rank matrix."""

    matrix = rank_matrix.to_numpy(dtype=float)
    n_raters, n_items = matrix.shape
    if n_raters < 2 or n_items < 2:
        return np.nan
    rank_sums = matrix.sum(axis=0)
    centered = rank_sums - rank_sums.mean()
    s_stat = float(np.sum(centered**2))
    denominator = float(n_raters**2 * (n_items**3 - n_items))
    return float(12.0 * s_stat / denominator) if denominator else np.nan


def rerank_subset(rank_matrix: pd.DataFrame) -> pd.DataFrame:
    """Rerank a selected feature subset within each seed before concordance testing."""

    if rank_matrix.empty:
        return rank_matrix.copy()
    return rank_matrix.rank(axis=1, method="average", ascending=True)


def kendalls_w_with_inference(rank_matrix: pd.DataFrame) -> dict[str, float | int]:
    """Compute Kendall's W plus large-sample chi-square inference."""

    ranks = rerank_subset(rank_matrix.dropna(axis=1, how="any"))
    n_seeds, n_features = ranks.shape
    if n_seeds < 2 or n_features < 2:
        return {
            "kendalls_w": np.nan,
            "chi_square": np.nan,
            "degrees_of_freedom": max(0, int(n_features - 1)),
            "p_value": np.nan,
            "n_seeds": int(n_seeds),
            "n_features": int(n_features),
        }
    w_value = kendalls_w(ranks)
    chi_square = float(n_seeds * (n_features - 1) * w_value)
    degrees_of_freedom = int(n_features - 1)
    p_value = float(chi2.sf(chi_square, degrees_of_freedom))
    return {
        "kendalls_w": float(w_value),
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "p_value": p_value,
        "n_seeds": int(n_seeds),
        "n_features": int(n_features),
    }


def feature_subset_by_importance(seed_results: pd.DataFrame, subset: str) -> list[str]:
    """Select feature names for all, top-k, or mid-importance Kendall's W."""

    ranks = (
        seed_results.groupby("feature", as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "mean_rank"})
        .sort_values(["mean_rank", "feature"], ascending=[True, True])
    )
    if subset == "all_features":
        return ranks["feature"].tolist()
    if subset == "top_5_features":
        return ranks.head(min(5, len(ranks)))["feature"].tolist()
    if subset == "top_10_features":
        return ranks.head(min(10, len(ranks)))["feature"].tolist()
    if subset == "mid_importance_features":
        mid = ranks.loc[ranks["mean_rank"] > 10].head(6)
        if len(mid) >= 2:
            return mid["feature"].tolist()
        return []
    raise ValueError(f"Unknown feature subset: {subset}")


def kendalls_w_by_feature_subset(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute Kendall's W for all, top-5, top-10, and mid-importance features."""

    rows = []
    subsets = ["all_features", "top_5_features", "top_10_features", "mid_importance_features"]
    for (dataset, scope, model_family), group in seed_results.groupby(
        ["dataset", "stability_scope", "model_family"]
    ):
        ranks = _rank_matrix(group)
        for subset in subsets:
            selected_features = feature_subset_by_importance(group, subset)
            if len(selected_features) < 2:
                rows.append(
                    {
                        "dataset": dataset,
                        "stability_scope": scope,
                        "model_family": model_family,
                        "feature_subset": subset,
                        "kendalls_w": np.nan,
                        "chi_square": np.nan,
                        "degrees_of_freedom": np.nan,
                        "p_value": np.nan,
                        "n_seeds": int(ranks.shape[0]),
                        "n_features": int(len(selected_features)),
                        "features_used": "",
                        "status": "not_available_insufficient_mid_importance_features",
                    }
                )
                continue
            inference = kendalls_w_with_inference(ranks[selected_features])
            rows.append(
                {
                    "dataset": dataset,
                    "stability_scope": scope,
                    "model_family": model_family,
                    "feature_subset": subset,
                    **inference,
                    "features_used": "; ".join(selected_features),
                    "status": "computed_from_existing_seed_results",
                }
            )
    return pd.DataFrame(rows)


def pairwise_rank_correlations(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman/Kendall rank correlations and top-k overlaps."""

    rows = []
    for (dataset, scope), group in seed_results.groupby(["dataset", "stability_scope"]):
        ranks = _rank_matrix(group)
        w_value = kendalls_w(ranks)
        seed_values = list(ranks.index)
        for i, seed_a in enumerate(seed_values):
            for seed_b in seed_values[i + 1 :]:
                rank_a = ranks.loc[seed_a]
                rank_b = ranks.loc[seed_b]
                spearman_rho, spearman_p = spearmanr(rank_a, rank_b)
                kendall_tau, kendall_p = kendalltau(rank_a, rank_b)
                top5_a = set(rank_a.sort_values().head(5).index)
                top5_b = set(rank_b.sort_values().head(5).index)
                top10_a = set(rank_a.sort_values().head(10).index)
                top10_b = set(rank_b.sort_values().head(10).index)
                rows.append(
                    {
                        "dataset": dataset,
                        "stability_scope": scope,
                        "model_family": "lightgbm",
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "spearman_rho": float(spearman_rho),
                        "spearman_p_value": float(spearman_p),
                        "kendall_tau": float(kendall_tau),
                        "kendall_p_value": float(kendall_p),
                        "kendalls_w": w_value,
                        "top5_overlap_count": int(len(top5_a & top5_b)),
                        "top5_overlap_rate": float(len(top5_a & top5_b) / 5.0),
                        "top10_overlap_count": int(len(top10_a & top10_b)),
                        "top10_overlap_rate": float(len(top10_a & top10_b) / 10.0),
                    }
                )
    return pd.DataFrame(rows)


def kendalls_w_summary_table(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute one Kendall's W concordance row per dataset/scope."""

    rows = []
    for (dataset, scope), group in seed_results.groupby(["dataset", "stability_scope"]):
        ranks = _rank_matrix(group)
        rows.append(
            {
                "dataset": dataset,
                "stability_scope": scope,
                "model_family": "lightgbm",
                "kendalls_w": kendalls_w(ranks),
                "n_seeds": int(ranks.shape[0]),
                "n_features": int(ranks.shape[1]),
                "metric_direction": "higher_is_better",
                "interpretation": "Kendall's W measures agreement among seed-specific SHAP rankings.",
            }
        )
    return pd.DataFrame(rows)


def feature_frequency_table(seed_results: pd.DataFrame) -> pd.DataFrame:
    """Compute feature rank standard deviation and top-k frequencies."""

    rows = []
    for (dataset, scope, feature), group in seed_results.groupby(["dataset", "stability_scope", "feature"]):
        ranks = group["rank"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "stability_scope": scope,
                "model_family": "lightgbm",
                "feature": feature,
                "raw_feature": group["raw_feature"].iloc[0],
                "mean_rank": float(ranks.mean()),
                "median_rank": float(ranks.median()),
                "rank_std": float(ranks.std(ddof=0)),
                "min_rank": int(ranks.min()),
                "max_rank": int(ranks.max()),
                "top5_count": int(group["in_top_5"].sum()),
                "top5_frequency": float(group["in_top_5"].mean()),
                "top10_count": int(group["in_top_10"].sum()),
                "top10_frequency": float(group["in_top_10"].mean()),
                "mean_abs_shap": float(group["mean_abs_shap"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "stability_scope", "top10_frequency", "mean_rank"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def plot_rank_stability_heatmap(correlations: pd.DataFrame, output_path: str | Path) -> None:
    """Save a heatmap of pairwise Spearman rank correlations."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(correlations.groupby(["dataset", "stability_scope"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)

    for ax, ((dataset, scope), group) in zip(axes.ravel(), groups):
        seeds = sorted(set(group["seed_a"]) | set(group["seed_b"]))
        matrix = pd.DataFrame(np.eye(len(seeds)), index=seeds, columns=seeds)
        for row in group.itertuples(index=False):
            matrix.loc[row.seed_a, row.seed_b] = row.spearman_rho
            matrix.loc[row.seed_b, row.seed_a] = row.spearman_rho
        im = ax.imshow(matrix.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{dataset} {scope}\nSpearman SHAP Rank Stability")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Seed")
        tick_step = max(1, len(seeds) // 10)
        tick_positions = np.arange(0, len(seeds), tick_step)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels([seeds[i] for i in tick_positions], rotation=90, fontsize=7)
        ax.set_yticklabels([seeds[i] for i in tick_positions], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_feature_frequency(frequency: pd.DataFrame, output_path: str | Path, top_n: int = 15) -> None:
    """Save a bar chart of top-10 feature frequencies."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = list(frequency.groupby(["dataset", "stability_scope"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(6.5 * len(groups), 5), squeeze=False)

    for ax, ((dataset, scope), group) in zip(axes.ravel(), groups):
        top = group.sort_values(["top10_frequency", "mean_rank"], ascending=[False, True]).head(top_n)
        ax.barh(top["feature"][::-1], top["top10_frequency"][::-1], color="#3b7a78")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Top-10 Frequency")
        ax.set_title(f"{dataset} {scope} Top Feature Frequency")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "StabilityDatasetConfig",
    "feature_frequency_table",
    "kendalls_w",
    "kendalls_w_summary_table",
    "pairwise_rank_correlations",
    "plot_rank_stability_heatmap",
    "plot_top_feature_frequency",
    "run_seed_stability",
    "shap_importance_for_seed",
]
