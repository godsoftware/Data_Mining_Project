"""Scorecard-style WOE logistic baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Encode tabular variables with weight of evidence learned on the train set.

    This is intentionally lightweight: it gives the project a transparent
    scorecard-style baseline without introducing a heavy binning dependency.
    """

    def __init__(
        self,
        n_bins: int = 5,
        smoothing: float = 0.5,
        categorical_columns: list[str] | None = None,
        min_unique_for_numeric_bins: int = 10,
    ) -> None:
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.categorical_columns = categorical_columns
        self.min_unique_for_numeric_bins = min_unique_for_numeric_bins

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn bin edges and WOE maps."""

        X_df = pd.DataFrame(X).copy()
        y_series = pd.Series(y).astype(int).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

        self.feature_names_in_ = list(X_df.columns)
        self.bin_edges_: dict[str, np.ndarray | None] = {}
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.default_woe_: dict[str, float] = {}

        categorical = set(self.categorical_columns or [])
        total_bad = float((y_series == 1).sum())
        total_good = float((y_series == 0).sum())

        for column in self.feature_names_in_:
            series = X_df[column]
            use_numeric_bins = (
                column not in categorical
                and pd.api.types.is_numeric_dtype(series)
                and series.nunique(dropna=True) >= self.min_unique_for_numeric_bins
            )

            if use_numeric_bins:
                edges = self._quantile_edges(series)
                labels = self._apply_edges(series, edges)
                self.bin_edges_[column] = edges
            else:
                labels = self._category_labels(series)
                self.bin_edges_[column] = None

            grouped = pd.DataFrame({"label": labels, "target": y_series}).groupby("label")
            n_groups = max(grouped.ngroups, 1)
            mapping: dict[str, float] = {}

            for label, group in grouped:
                bad = float((group["target"] == 1).sum())
                good = float((group["target"] == 0).sum())
                bad_rate = (bad + self.smoothing) / (total_bad + self.smoothing * n_groups)
                good_rate = (good + self.smoothing) / (total_good + self.smoothing * n_groups)
                mapping[str(label)] = float(np.log(bad_rate / good_rate))

            self.woe_maps_[column] = mapping
            self.default_woe_[column] = float(np.mean(list(mapping.values()))) if mapping else 0.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WOE values."""

        X_df = pd.DataFrame(X).copy()
        transformed = pd.DataFrame(index=X_df.index)

        for column in self.feature_names_in_:
            series = X_df[column]
            edges = self.bin_edges_[column]
            labels = self._apply_edges(series, edges) if edges is not None else self._category_labels(series)
            mapping = self.woe_maps_[column]
            transformed[column] = labels.astype(str).map(mapping).fillna(self.default_woe_[column])

        return transformed

    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""

        return np.asarray(self.feature_names_in_, dtype=object)

    def _quantile_edges(self, series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return np.asarray([-np.inf, np.inf])
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.nanquantile(values, quantiles))
        if len(edges) <= 2:
            edges = np.asarray([values.min(), values.max()])
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def _apply_edges(self, series: pd.Series, edges: np.ndarray) -> pd.Series:
        labels = pd.cut(pd.to_numeric(series, errors="coerce"), bins=edges, include_lowest=True)
        return labels.astype("object").where(pd.notna(labels), "__MISSING__").astype(str)

    def _category_labels(self, series: pd.Series) -> pd.Series:
        return series.astype("object").where(pd.notna(series), "__MISSING__").astype(str)


def make_scorecard_pipeline(
    categorical_columns: list[str] | None = None,
    class_weight: str | dict | None = "balanced",
) -> Pipeline:
    """Create a WOE + logistic-regression scorecard baseline."""

    return Pipeline(
        steps=[
            ("woe", WOEEncoder(categorical_columns=categorical_columns)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    class_weight=class_weight,
                ),
            ),
        ]
    )

