"""FICO HELOC dataset loader."""

from __future__ import annotations

from heloc_preprocessing import load_heloc_dataset
from src.config.paths import HELOC_RAW_CSV


def load_heloc(path=HELOC_RAW_CSV):
    """Load the raw HELOC external-validation dataset."""

    return load_heloc_dataset(path)

