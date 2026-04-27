"""Taiwan default dataset loader."""

from __future__ import annotations

from data_preprocessing import load_raw_dataset
from src.config.paths import TAIWAN_RAW_XLS


def load_taiwan(path=TAIWAN_RAW_XLS):
    """Load the raw UCI Taiwan default dataset."""

    return load_raw_dataset(path)

