"""Leakage guard tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import TAIWAN_MODEL_READY
from src.data.validation_checks import assert_no_target_leakage_columns


def test_no_obvious_target_leakage_columns():
    df = pd.read_csv(TAIWAN_MODEL_READY)
    assert_no_target_leakage_columns(df, target="default_next_month")

