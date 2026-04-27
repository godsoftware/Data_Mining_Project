"""Prepare Taiwan raw, cleaned, and model-ready data."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_preprocessing import main as clean_taiwan
from experiment_registry import finish_run, start_run
from feature_engineering import main as build_taiwan_features
from src.config.paths import FIGURES_DIR, TABLES_DIR, TAIWAN_CLEANED, TAIWAN_MODEL_READY


def main() -> None:
    run_id = start_run("prepare_taiwan", dataset="taiwan", tags=["phase-2", "data-pipeline"])
    try:
        clean_taiwan()
        build_taiwan_features()
        finish_run(
            run_id,
            artifacts={
                "cleaned_data": TAIWAN_CLEANED,
                "model_ready_data": TAIWAN_MODEL_READY,
                "data_audit": TABLES_DIR / "taiwan_data_audit.csv",
                "target_distribution": FIGURES_DIR / "taiwan_target_distribution.png",
                "feature_dictionary": TABLES_DIR / "feature_dictionary_taiwan.csv",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
