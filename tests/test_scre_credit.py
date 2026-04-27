"""SCRE-Credit utility tests."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scre_credit import three_way_policy_metrics


def test_three_way_policy_metrics_runs():
    metrics = three_way_policy_metrics([0, 1, 1, 0], [0.1, 0.3, 0.8, 0.6], 0.2, 0.5)
    assert metrics["manual_review_count"] == 1
    assert metrics["high_count"] == 2

