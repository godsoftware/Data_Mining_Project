"""Runtime settings loaded from the project YAML config."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from config.paths import PROJECT_ROOT


CONFIG_PATH = Path(__file__).with_name("experiment_config.yaml")


@lru_cache(maxsize=1)
def load_experiment_config() -> dict[str, Any]:
    """Load the central experiment configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def project_setting(key: str, default: Any = None) -> Any:
    """Return a top-level project setting."""

    return load_experiment_config().get("project", {}).get(key, default)


def cost_setting(key: str, default: Any = None) -> Any:
    """Return a top-level cost setting."""

    return load_experiment_config().get("costs", {}).get(key, default)


def split_setting(key: str, default: Any = None) -> Any:
    """Return a split-policy setting."""

    return load_experiment_config().get("splits", {}).get(key, default)


PROJECT_NAME = str(project_setting("name", "SCRE-Credit"))
RANDOM_SEED = int(project_setting("random_seed", 42))
FN_COST = float(cost_setting("false_negative", 5.0))
FP_COST = float(cost_setting("false_positive", 1.0))
MANUAL_REVIEW_COST = float(cost_setting("manual_review", 0.5))
TRAIN_SIZE = float(split_setting("train", 0.60))
VALIDATION_SIZE = float(split_setting("validation", 0.20))
TEST_SIZE = float(split_setting("test", 0.20))


__all__ = [
    "CONFIG_PATH",
    "PROJECT_ROOT",
    "PROJECT_NAME",
    "RANDOM_SEED",
    "FN_COST",
    "FP_COST",
    "MANUAL_REVIEW_COST",
    "TRAIN_SIZE",
    "VALIDATION_SIZE",
    "TEST_SIZE",
    "load_experiment_config",
]
