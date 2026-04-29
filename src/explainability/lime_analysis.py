"""LIME and SHAP-LIME agreement utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_reliability import normalize_lime_rule_feature


def build_lime_tabular_explainer(
    X_train: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    class_names: tuple[str, str] = ("non_default", "default"),
    random_seed: int = 42,
):
    """Build a LIME tabular explainer over raw model inputs."""

    from lime.lime_tabular import LimeTabularExplainer

    categorical_columns = categorical_columns or []
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
        random_state=random_seed,
    )


def dataframe_prediction_fn(pipeline, columns: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap pipeline.predict_proba so LIME can pass numpy arrays safely."""

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return pipeline.predict_proba(frame)

    return predict


def explain_lime_cases(
    pipeline,
    X_cases: pd.DataFrame,
    case_metadata: pd.DataFrame,
    X_train: pd.DataFrame,
    output_dir: str | Path,
    categorical_columns: list[str] | None = None,
    num_features: int = 10,
    num_samples: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate LIME explanations for selected local cases and save plots/HTML."""

    explainer = build_lime_tabular_explainer(
        X_train,
        categorical_columns=categorical_columns,
        random_seed=random_seed,
    )
    predict_fn = dataframe_prediction_fn(pipeline, list(X_train.columns))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, case in case_metadata.reset_index(drop=True).iterrows():
        explanation = explainer.explain_instance(
            X_cases.iloc[position].to_numpy(),
            predict_fn,
            labels=(1,),
            num_features=num_features,
            num_samples=num_samples,
        )
        case_name = f"{case['case_type']}_row_{int(case['row_id'])}"
        explanation.save_to_file(str(output_dir / f"{case_name}.html"))
        fig = explanation.as_pyplot_figure(label=1)
        fig.tight_layout()
        fig.savefig(output_dir / f"{case_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        for rank, (feature_rule, lime_weight) in enumerate(explanation.as_list(label=1), start=1):
            raw_feature = normalize_lime_rule_feature(feature_rule, list(X_train.columns))
            rows.append(
                {
                    "case_type": case["case_type"],
                    "row_id": int(case["row_id"]),
                    "predicted_probability": float(case["predicted_probability"]),
                    "threshold": float(case["threshold"]),
                    "rank": rank,
                    "feature_rule": feature_rule,
                    "raw_feature": raw_feature,
                    "lime_weight": float(lime_weight),
                    "abs_lime_weight": float(abs(lime_weight)),
                    "lime_sign": float(np.sign(lime_weight)),
                    "explanation_png": str(output_dir / f"{case_name}.png"),
                    "explanation_html": str(output_dir / f"{case_name}.html"),
                }
            )
    return pd.DataFrame(rows)


def shap_lime_agreement_table(
    shap_local_features: pd.DataFrame,
    lime_local_features: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compute top-k overlap, sign agreement, and local consistency per case."""

    rows = []
    case_keys = sorted(set(shap_local_features["case_type"]) & set(lime_local_features["case_type"]))
    for case_type in case_keys:
        shap_case = (
            shap_local_features.loc[shap_local_features["case_type"] == case_type]
            .sort_values("abs_shap_value", ascending=False)
            .head(top_k)
        )
        lime_case = (
            lime_local_features.loc[lime_local_features["case_type"] == case_type]
            .sort_values("abs_lime_weight", ascending=False)
            .head(top_k)
        )
        shap_set = set(shap_case["raw_feature"])
        lime_set = set(lime_case["raw_feature"])
        overlap = sorted(shap_set & lime_set)
        union = shap_set | lime_set

        shap_sign = shap_case.groupby("raw_feature")["shap_sign"].first()
        lime_sign = lime_case.groupby("raw_feature")["lime_sign"].first()
        sign_matches = [
            float(np.sign(shap_sign.loc[feature]) == np.sign(lime_sign.loc[feature]))
            for feature in overlap
        ]
        sign_agreement = float(np.mean(sign_matches)) if sign_matches else np.nan
        jaccard = float(len(overlap) / len(union)) if union else np.nan
        local_consistency = float(jaccard * sign_agreement) if np.isfinite(sign_agreement) else np.nan

        rows.append(
            {
                "case_type": case_type,
                "row_id": int(shap_case["row_id"].iloc[0]) if not shap_case.empty else np.nan,
                "top_k": top_k,
                "shap_top_k_count": int(len(shap_set)),
                "lime_top_k_count": int(len(lime_set)),
                "overlap_count": int(len(overlap)),
                "top_k_overlap": ", ".join(overlap),
                "jaccard_overlap": jaccard,
                "sign_agreement": sign_agreement,
                "local_explanation_consistency": local_consistency,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "build_lime_tabular_explainer",
    "dataframe_prediction_fn",
    "explain_lime_cases",
    "shap_lime_agreement_table",
]
