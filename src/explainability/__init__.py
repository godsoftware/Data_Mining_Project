"""Explainability package with compatibility exports."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "explainability.py"
_SPEC = importlib.util.spec_from_file_location("_legacy_explainability", _LEGACY_PATH)
_legacy = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_legacy)

build_lime_explainer = _legacy.build_lime_explainer
compute_shap_values = _legacy.compute_shap_values
faithfulness_permutation_test = _legacy.faithfulness_permutation_test
get_fitted_model = _legacy.get_fitted_model
ranking_stability_table = _legacy.ranking_stability_table
save_shap_summary_plots = _legacy.save_shap_summary_plots
top_k_frequency = _legacy.top_k_frequency
top_shap_features = _legacy.top_shap_features
transformed_feature_frame = _legacy.transformed_feature_frame

__all__ = [
    "build_lime_explainer",
    "compute_shap_values",
    "faithfulness_permutation_test",
    "get_fitted_model",
    "ranking_stability_table",
    "save_shap_summary_plots",
    "top_k_frequency",
    "top_shap_features",
    "transformed_feature_frame",
]
