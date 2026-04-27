"""Run the legacy Taiwan base-model comparison."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_enhanced_modeling import main
from experiment_registry import finish_run, start_run


if __name__ == "__main__":
    run_id = start_run("train_base_models_taiwan", dataset="taiwan", tags=["phase-2", "baseline-models"])
    try:
        main()
        finish_run(run_id)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise
