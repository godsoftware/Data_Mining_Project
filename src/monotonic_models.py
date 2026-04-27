"""Monotonic constraint helpers for credit-risk boosting models."""

from __future__ import annotations

from features.feature_registry import HELOC_MONOTONIC_PRIORS, TAIWAN_MONOTONIC_PRIORS

def monotonic_constraint_vector(feature_names: list[str], priors: dict[str, int]) -> list[int]:
    """Return model-ready monotonic constraints for a feature list."""

    return [int(priors.get(feature, 0)) for feature in feature_names]


def xgboost_constraint_string(feature_names: list[str], priors: dict[str, int]) -> str:
    """Return XGBoost's tuple-string monotonic constraint format."""

    values = monotonic_constraint_vector(feature_names, priors)
    return "(" + ",".join(str(value) for value in values) + ")"
