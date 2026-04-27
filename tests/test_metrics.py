"""Metric tests."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.evaluation.metrics import binary_classification_metrics


def test_binary_metrics_include_extended_fields():
    metrics = binary_classification_metrics([0, 1, 1, 0], [0.1, 0.8, 0.3, 0.2], threshold=0.5)
    for key in ["roc_auc", "pr_auc", "brier_score", "ece", "calibration_slope", "g_mean"]:
        assert key in metrics

