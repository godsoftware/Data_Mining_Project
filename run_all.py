"""Run the reproducible SCRE-Credit workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_registry import finish_run, start_run

PYTHON = sys.executable

STEPS = [
    "experiments/00_environment_check.py",
    "experiments/01_prepare_taiwan.py",
    "experiments/02_prepare_heloc.py",
    "experiments/03_feature_engineering.py",
    "experiments/04_create_splits.py",
    "experiments/05_train_baselines.py",
    "experiments/06_train_scorecard.py",
    "experiments/07_train_boosting_models.py",
    "experiments/04_train_scre_credit_taiwan.py",
    "experiments/05_external_validation_heloc.py",
]


def main() -> None:
    run_id = start_run("run_all", dataset="both", tags=["phase-1", "core-workflow"])
    try:
        for step in STEPS:
            print(f"\n=== {step} ===")
            subprocess.run([PYTHON, str(PROJECT_ROOT / step)], check=True, cwd=PROJECT_ROOT)
        finish_run(run_id, artifacts={"steps": ", ".join(STEPS)})
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
