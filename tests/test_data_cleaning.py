"""Data-cleaning smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import TAIWAN_CLEANED


def test_taiwan_cleaned_categories_are_valid():
    df = pd.read_csv(TAIWAN_CLEANED)
    assert set(df["EDUCATION"].unique()).issubset({1, 2, 3, 4})
    assert set(df["MARRIAGE"].unique()).issubset({1, 2, 3})

