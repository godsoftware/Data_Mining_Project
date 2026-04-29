"""SHAP analysis utilities for the credit-risk model pool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ShapComputation:
    """Container for transformed data and positive-class SHAP values."""

    X_transformed: pd.DataFrame
    shap_values: object
    explainer_type: str
    model_name: str


def fitted_model_from_pipeline(pipeline):
    """Return the final estimator from a sklearn/imblearn pipeline."""

    if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
        return pipeline.named_steps["model"]
    return pipeline


def transform_pipeline_features(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Transform raw model inputs and return a named feature frame."""

    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        return X.copy()

    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=X.index)


def raw_feature_from_transformed(feature: str, raw_features: list[str]) -> str:
    """Map a transformed one-hot/scaled feature name back to a raw feature where possible."""

    if feature in raw_features:
        return feature
    for raw_feature in sorted(raw_features, key=len, reverse=True):
        if feature.startswith(f"{raw_feature}_"):
            return raw_feature
    return feature


def _positive_class_explanation(shap_values):
    """Convert SHAP output to a single positive-class Explanation object."""

    import shap

    values = getattr(shap_values, "values", shap_values)
    data = getattr(shap_values, "data", None)
    feature_names = getattr(shap_values, "feature_names", None)
    base_values = getattr(shap_values, "base_values", None)

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
        if base_values is not None:
            base_values = np.asarray(base_values)
            if base_values.ndim == 2:
                base_values = base_values[:, 1]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=feature_names,
    )


def _prediction_fn_for_transformed_model(model, columns: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    """Build a model-agnostic prediction function over transformed features."""

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        return model.predict_proba(frame)[:, 1]

    return predict


def compute_shap_analysis(
    pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    model_name: str,
    max_background_rows: int = 200,
    random_seed: int = 42,
) -> ShapComputation:
    """Compute TreeExplainer SHAP values, falling back to permutation SHAP if needed."""

    import shap

    if len(X_background) > max_background_rows:
        X_background = X_background.sample(max_background_rows, random_state=random_seed)

    X_background_transformed = transform_pipeline_features(pipeline, X_background)
    X_explain_transformed = transform_pipeline_features(pipeline, X_explain)
    model = fitted_model_from_pipeline(pipeline)

    try:
        explainer = shap.TreeExplainer(model)
        try:
            shap_values = explainer(X_explain_transformed, check_additivity=False)
        except TypeError:
            shap_values = explainer(X_explain_transformed)
        explainer_type = "tree"
    except Exception:
        masker = shap.maskers.Independent(
            X_background_transformed,
            max_samples=min(max_background_rows, len(X_background_transformed)),
        )
        explainer = shap.Explainer(
            _prediction_fn_for_transformed_model(model, list(X_explain_transformed.columns)),
            masker,
            algorithm="permutation",
            feature_names=list(X_explain_transformed.columns),
        )
        shap_values = explainer(
            X_explain_transformed,
            max_evals=2 * X_explain_transformed.shape[1] + 1,
            silent=True,
        )
        explainer_type = "permutation_fallback"

    return ShapComputation(
        X_transformed=X_explain_transformed,
        shap_values=_positive_class_explanation(shap_values),
        explainer_type=explainer_type,
        model_name=model_name,
    )


def save_global_shap_plots(computation: ShapComputation, output_dir: str | Path) -> None:
    """Save global SHAP bar and beeswarm plots."""

    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap.plots.bar(computation.shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "global_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.plots.beeswarm(computation.shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "global_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()


def shap_top_features_table(
    computation: ShapComputation,
    raw_features: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Return global mean absolute SHAP feature importance."""

    values = np.asarray(computation.shap_values.values)
    feature_names = list(computation.shap_values.feature_names)
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "raw_feature": [raw_feature_from_transformed(feature, raw_features) for feature in feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    table["rank"] = np.arange(1, len(table) + 1)
    table["shap_direction"] = np.sign(table["mean_shap"])
    return table.head(top_n).reset_index(drop=True)


def save_dependence_plots(
    computation: ShapComputation,
    top_features: list[str],
    output_dir: str | Path,
) -> None:
    """Save SHAP dependence plots for selected transformed features."""

    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(computation.shap_values.values)

    for feature in top_features:
        if feature not in computation.X_transformed.columns:
            continue
        shap.dependence_plot(
            feature,
            values,
            computation.X_transformed,
            show=False,
            interaction_index=None,
        )
        plt.tight_layout()
        safe_feature = feature.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plt.savefig(output_dir / f"{safe_feature}.png", dpi=300, bbox_inches="tight")
        plt.close()


def local_shap_feature_table(
    computation: ShapComputation,
    case_metadata: pd.DataFrame,
    raw_features: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top local SHAP contributions for each selected case."""

    values = np.asarray(computation.shap_values.values)
    feature_names = list(computation.shap_values.feature_names)
    rows = []
    for position, case in case_metadata.reset_index(drop=True).iterrows():
        row_values = values[position]
        order = np.argsort(np.abs(row_values))[::-1][:top_k]
        for rank, feature_index in enumerate(order, start=1):
            feature = feature_names[feature_index]
            rows.append(
                {
                    "case_type": case["case_type"],
                    "row_id": int(case["row_id"]),
                    "predicted_probability": float(case["predicted_probability"]),
                    "threshold": float(case["threshold"]),
                    "rank": rank,
                    "feature": feature,
                    "raw_feature": raw_feature_from_transformed(feature, raw_features),
                    "shap_value": float(row_values[feature_index]),
                    "abs_shap_value": float(abs(row_values[feature_index])),
                    "shap_sign": float(np.sign(row_values[feature_index])),
                }
            )
    return pd.DataFrame(rows)


def save_local_waterfall_plots(
    computation: ShapComputation,
    case_metadata: pd.DataFrame,
    output_dir: str | Path,
    max_display: int = 15,
) -> None:
    """Save SHAP local waterfall plots for selected cases."""

    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for position, case in case_metadata.reset_index(drop=True).iterrows():
        shap.plots.waterfall(
            computation.shap_values[position],
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        filename = f"{case['case_type']}_row_{int(case['row_id'])}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


__all__ = [
    "ShapComputation",
    "compute_shap_analysis",
    "fitted_model_from_pipeline",
    "local_shap_feature_table",
    "raw_feature_from_transformed",
    "save_dependence_plots",
    "save_global_shap_plots",
    "save_local_waterfall_plots",
    "shap_top_features_table",
    "transform_pipeline_features",
]
