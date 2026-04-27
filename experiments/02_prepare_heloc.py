"""Prepare HELOC cleaned and model-ready data."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from heloc_preprocessing import main as prepare_heloc
from experiment_registry import finish_run, start_run
from src.config.paths import FIGURES_DIR, HELOC_CLEANED, HELOC_MODEL_READY, TABLES_DIR


if __name__ == "__main__":
    run_id = start_run("prepare_heloc", dataset="heloc", tags=["phase-3", "external-data-pipeline"])
    try:
        prepare_heloc()
        finish_run(
            run_id,
            artifacts={
                "cleaned_data": HELOC_CLEANED,
                "model_ready_data": HELOC_MODEL_READY,
                "data_audit": TABLES_DIR / "heloc_data_audit.csv",
                "data_dictionary": TABLES_DIR / "heloc_data_dictionary.csv",
                "feature_dictionary": TABLES_DIR / "feature_dictionary_heloc.csv",
                "target_distribution": FIGURES_DIR / "heloc_target_distribution.png",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise
