"""Evaluation package with compatibility exports."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_LEGACY_PATH = Path(__file__).resolve().parents[1] / "evaluation.py"
_SPEC = importlib.util.spec_from_file_location("_legacy_evaluation", _LEGACY_PATH)
_legacy = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_legacy)

binary_classification_metrics = _legacy.binary_classification_metrics
calibration_slope_intercept = _legacy.calibration_slope_intercept
expected_calibration_error = _legacy.expected_calibration_error
plot_calibration_curve = _legacy.plot_calibration_curve
plot_roc_and_pr_curves = _legacy.plot_roc_and_pr_curves
save_model_comparison = _legacy.save_model_comparison
threshold_optimization_table = _legacy.threshold_optimization_table

__all__ = [
    "binary_classification_metrics",
    "calibration_slope_intercept",
    "expected_calibration_error",
    "plot_calibration_curve",
    "plot_roc_and_pr_curves",
    "save_model_comparison",
    "threshold_optimization_table",
]
