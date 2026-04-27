"""SHAP, LIME, stability, and faithfulness helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from config.settings import RANDOM_SEED


def get_fitted_model(pipeline):
    """Return the final estimator from a sklearn/imblearn pipeline."""

    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        return pipeline.named_steps["model"]
    return pipeline


def transformed_feature_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return preprocessed features as a DataFrame with transformed names."""

    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        return X.copy()

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def compute_shap_values(pipeline, X: pd.DataFrame, max_rows: int = 1000, random_state: int = RANDOM_SEED):
    """Compute SHAP values for a fitted pipeline on a manageable sample."""

    import shap

    X_sample = X.sample(min(len(X), max_rows), random_state=random_state)
    X_transformed = transformed_feature_frame(pipeline, X_sample)
    model = get_fitted_model(pipeline)

    is_permutation = False
    try:
        explainer = shap.Explainer(model, X_transformed)
    except TypeError:
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            masker = shap.maskers.Independent(
                X_transformed,
                max_samples=min(100, len(X_transformed)),
            )
            predict_fn = lambda values: model.predict_proba(values)[:, 1]
            explainer = shap.Explainer(
                predict_fn,
                masker,
                algorithm="permutation",
                feature_names=list(X_transformed.columns),
            )
            is_permutation = True

    try:
        if is_permutation:
            shap_values = explainer(
                X_transformed,
                max_evals=2 * X_transformed.shape[1] + 1,
                silent=True,
            )
        else:
            shap_values = explainer(X_transformed)
    except Exception as exc:
        message = str(exc)
        if "Additivity check failed" in message:
            shap_values = explainer(X_transformed, check_additivity=False)
        elif "could not convert string to float" in message:
            masker = shap.maskers.Independent(
                X_transformed,
                max_samples=min(100, len(X_transformed)),
            )
            predict_fn = lambda values: model.predict_proba(values)[:, 1]
            explainer = shap.Explainer(
                predict_fn,
                masker,
                algorithm="permutation",
                feature_names=list(X_transformed.columns),
            )
            shap_values = explainer(
                X_transformed,
                max_evals=2 * X_transformed.shape[1] + 1,
                silent=True,
            )
        else:
            raise

    return X_transformed, shap_values


def save_shap_summary_plots(
    X_transformed: pd.DataFrame,
    shap_values,
    output_dir: str | Path,
) -> None:
    """Save SHAP beeswarm and bar plots."""

    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def top_shap_features(shap_values, top_n: int = 10) -> pd.DataFrame:
    """Return mean absolute SHAP importance as a ranked table."""

    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    importance = np.abs(values).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": shap_values.feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    return table.head(top_n)


def ranking_stability_table(rankings: dict[int, list[str]]) -> pd.DataFrame:
    """Compute pairwise Spearman correlation between SHAP feature rankings."""

    all_features = sorted({feature for ranking in rankings.values() for feature in ranking})
    default_rank = len(all_features) + 1
    rows = []

    for seed_a, ranking_a in rankings.items():
        for seed_b, ranking_b in rankings.items():
            if seed_a >= seed_b:
                continue

            rank_a = {feature: i + 1 for i, feature in enumerate(ranking_a)}
            rank_b = {feature: i + 1 for i, feature in enumerate(ranking_b)}
            vector_a = [rank_a.get(feature, default_rank) for feature in all_features]
            vector_b = [rank_b.get(feature, default_rank) for feature in all_features]
            rho, p_value = spearmanr(vector_a, vector_b)
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)


def top_k_frequency(rankings: dict[int, list[str]], k: int = 5) -> pd.DataFrame:
    """Report how often each feature appears in the top-k SHAP list."""

    counts: dict[str, int] = {}
    for ranking in rankings.values():
        for feature in ranking[:k]:
            counts[feature] = counts.get(feature, 0) + 1

    table = pd.DataFrame(
        [{"feature": feature, "top_k_count": count} for feature, count in counts.items()]
    )
    table["top_k_rate"] = table["top_k_count"] / max(len(rankings), 1)
    return table.sort_values(["top_k_count", "feature"], ascending=[False, True])


def faithfulness_permutation_test(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Permute selected raw features and measure AUC/probability degradation."""

    rng = np.random.default_rng(random_state)
    baseline_proba = pipeline.predict_proba(X)[:, 1]
    baseline_auc = roc_auc_score(y, baseline_proba)
    rows = []

    for feature in features:
        if feature not in X.columns:
            continue

        perturbed = X.copy()
        perturbed[feature] = rng.permutation(perturbed[feature].to_numpy())
        perturbed_proba = pipeline.predict_proba(perturbed)[:, 1]
        perturbed_auc = roc_auc_score(y, perturbed_proba)

        rows.append(
            {
                "perturbed_feature": feature,
                "baseline_auc": baseline_auc,
                "perturbed_auc": perturbed_auc,
                "auc_drop": baseline_auc - perturbed_auc,
                "mean_probability_shift": float(
                    np.abs(baseline_proba - perturbed_proba).mean()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False)


def build_lime_explainer(
    X_train: pd.DataFrame,
    categorical_columns: list[str],
    class_names: tuple[str, str] = ("non_default", "default"),
):
    """Create a LIME tabular explainer for raw pipeline inputs."""

    from lime.lime_tabular import LimeTabularExplainer

    categorical_indexes = [
        X_train.columns.get_loc(column)
        for column in categorical_columns
        if column in X_train.columns
    ]

    return LimeTabularExplainer(
        training_data=X_train.to_numpy(),
        feature_names=list(X_train.columns),
        class_names=list(class_names),
        categorical_features=categorical_indexes,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_SEED,
    )
