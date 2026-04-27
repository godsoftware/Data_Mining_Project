"""Run SCRE-Credit external robustness validation on HELOC."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_scre_credit_experiments import run_one_dataset
from experiment_registry import finish_run, start_run


def main() -> None:
    run_id = start_run("external_validation_heloc_entrypoint", dataset="heloc", tags=["phase-3", "external-validation"])
    try:
        result = run_one_dataset("heloc", n_bootstraps=300, quick=False)
        print(result)
        finish_run(run_id, metrics={"best_threshold": result["best_threshold"]})
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
