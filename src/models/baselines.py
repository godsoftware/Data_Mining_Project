"""Baseline model builders."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from run_baseline_models import build_baseline_models
from scre_credit import make_model_pipeline

__all__ = ["build_baseline_models", "logistic_regression_pipeline"]


def logistic_regression_pipeline(X, categorical_columns=None, class_weight=None, random_state: int = 42):
    """Create a logistic-regression baseline pipeline."""

    return make_model_pipeline(
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight=class_weight,
            random_state=random_state,
        ),
        X,
        categorical_columns,
    )
