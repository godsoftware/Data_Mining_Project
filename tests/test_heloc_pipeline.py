"""HELOC data-pipeline smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heloc_preprocessing import HELOC_SPECIAL_CODES, HELOC_TARGET
from src.config.paths import HELOC_CLEANED, HELOC_MODEL_READY, HELOC_RAW_CSV, PROJECT_ROOT


HELOC_ENGINEERED_FEATURES = [
    "delinquency_intensity",
    "trade_activity_ratio",
    "recent_inquiry_pressure",
    "revolving_burden_proxy",
    "installment_burden_proxy",
    "credit_history_length_proxy",
    "negative_trade_signal",
    "high_utilization_signal",
]


def test_heloc_raw_file_exists():
    assert HELOC_RAW_CSV.exists()


def test_heloc_target_is_binary_and_original_target_removed():
    df = pd.read_csv(HELOC_CLEANED)
    assert HELOC_TARGET in df.columns
    assert "RiskPerformance" not in df.columns
    assert set(df[HELOC_TARGET].unique()).issubset({0, 1})


def test_heloc_special_codes_are_not_left_as_feature_values():
    df = pd.read_csv(HELOC_CLEANED)
    features = df.drop(columns=[HELOC_TARGET])
    for code in HELOC_SPECIAL_CODES:
        assert not (features == code).any().any()


def test_heloc_model_ready_matches_metric_interface():
    df = pd.read_csv(HELOC_MODEL_READY)
    assert HELOC_TARGET in df.columns
    assert set(df[HELOC_TARGET].unique()).issubset({0, 1})


def test_heloc_engineered_features_exist():
    df = pd.read_csv(HELOC_MODEL_READY)
    for column in HELOC_ENGINEERED_FEATURES:
        assert column in df.columns


def test_heloc_feature_dictionary_is_target_free():
    dictionary = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "feature_dictionary_heloc.csv")
    assert set(HELOC_ENGINEERED_FEATURES).issubset(set(dictionary["feature"]))
    assert not dictionary["uses_target"].astype(bool).any()
