"""Generate the current core SCRE-Credit outputs."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_scre_credit_experiments import run_one_dataset
from experiment_registry import finish_run, start_run


def main() -> None:
    run_id = start_run("generate_all_outputs", dataset="both", tags=["phase-2", "phase-3", "scre-credit"])
    try:
        for dataset in ("taiwan", "heloc"):
            print(f"Running {dataset}...")
            run_one_dataset(dataset, n_bootstraps=300, quick=False)
        finish_run(run_id)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
