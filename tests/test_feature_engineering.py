"""Feature-engineering smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import PROJECT_ROOT, TAIWAN_MODEL_READY


TAIWAN_FEATURES = [
    "delay_count",
    "severe_delay_count",
    "max_delay",
    "avg_delay",
    "recent_delay",
    "total_bill_amt",
    "avg_bill_amt",
    "max_bill_amt",
    "bill_trend",
    "total_pay_amt",
    "avg_pay_amt",
    "max_pay_amt",
    "payment_to_bill_ratio",
    "utilization_proxy",
    "recent_payment_intensity",
    "bill_volatility",
    "payment_volatility",
]


def test_taiwan_engineered_features_exist():
    df = pd.read_csv(TAIWAN_MODEL_READY)
    for column in TAIWAN_FEATURES:
        assert column in df.columns


def test_taiwan_model_ready_excludes_id():
    df = pd.read_csv(TAIWAN_MODEL_READY)
    assert "ID" not in df.columns


def test_taiwan_feature_dictionary_is_target_free():
    dictionary = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "feature_dictionary_taiwan.csv")
    assert set(TAIWAN_FEATURES).issubset(set(dictionary["feature"]))
    assert not dictionary["uses_target"].astype(bool).any()
