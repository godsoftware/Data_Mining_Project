"""Common feature transformation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def signed_log1p(values: pd.Series) -> pd.Series:
    """Signed log transform for variables that can be negative."""

    return np.sign(values) * np.log1p(np.abs(values))


def safe_ratio(numerator: pd.Series, denominator: pd.Series, offset: float = 1.0) -> pd.Series:
    """Compute a stable ratio with absolute denominator protection."""

    return numerator / (np.abs(denominator) + offset)


def one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible one-hot encoder for categorical features."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build the train-fitted numeric/categorical preprocessing transformer."""

    categorical_features = categorical_features or []
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric_features)]
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers=transformers, remainder="drop")
