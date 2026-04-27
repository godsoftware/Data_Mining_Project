"""Train SCRE-Credit on the Taiwan dataset."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_scre_credit_experiments import run_one_dataset
from experiment_registry import finish_run, start_run


def main() -> None:
    run_id = start_run("train_scre_credit_taiwan_entrypoint", dataset="taiwan", tags=["phase-2", "scre-credit"])
    try:
        result = run_one_dataset("taiwan", n_bootstraps=300, quick=False)
        print(result)
        finish_run(run_id, metrics={"best_threshold": result["best_threshold"]})
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
