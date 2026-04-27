"""Model factory for dataset-specific SCRE-Credit pools."""

from __future__ import annotations

from scre_credit import build_candidate_models


def make_candidate_pool(dataset: str, X, categorical_columns=None, random_state: int = 42, include_catboost: bool = True):
    """Build the configured candidate pool for Taiwan or HELOC."""

    profile = "heloc_reduced" if dataset == "heloc" else "taiwan_full"
    return build_candidate_models(
        X,
        categorical_columns=categorical_columns,
        random_state=random_state,
        include_catboost=include_catboost,
        model_profile=profile,
    )

