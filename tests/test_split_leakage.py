"""Split and leakage-control output tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import TABLES_DIR


def test_split_summaries_are_60_20_20_and_stratified():
    for name in ["taiwan", "heloc"]:
        summary = pd.read_csv(TABLES_DIR / f"split_summary_{name}.csv")
        fractions = dict(zip(summary["split"], summary["row_fraction"]))
        assert abs(fractions["train"] - 0.60) < 0.01
        assert abs(fractions["validation"] - 0.20) < 0.01
        assert abs(fractions["test"] - 0.20) < 0.01
        assert summary["stratified"].astype(bool).all()
        assert summary["target_1_rate"].max() - summary["target_1_rate"].min() < 0.002


def test_leakage_checklist_all_pass():
    checklist = pd.read_csv(TABLES_DIR / "leakage_checklist.csv")
    assert len(checklist) >= 10
    assert set(checklist["status"]) == {"pass"}


def test_duplicate_split_overlap_audit_exists_for_both_datasets():
    audit = pd.read_csv(TABLES_DIR / "duplicate_split_overlap_audit.csv")
    required = {
        "dataset",
        "split_left",
        "split_right",
        "overlap_feature_keys",
        "duplicate_pair_count",
        "same_target_pair_count",
        "different_target_pair_count",
        "policy",
    }
    assert required.issubset(audit.columns)
    assert set(audit["dataset"]) == {"taiwan", "heloc"}
    assert set(audit["split_left"] + "-" + audit["split_right"]) == {
        "train-validation",
        "train-test",
        "validation-test",
    }
    assert audit["duplicate_pair_count"].ge(0).all()
