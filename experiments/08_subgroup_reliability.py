"""Placeholder entry point for subgroup reliability experiments."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_registry import finish_run, start_run


def main() -> None:
    run_id = start_run("subgroup_reliability_placeholder", dataset="taiwan", tags=["phase-4", "planned"])
    print("TODO: Run subgroup reliability for Taiwan demographics and HELOC risk bands.")
    finish_run(run_id, status="planned", notes="Placeholder only; subgroup reliability is planned, not complete.")


if __name__ == "__main__":
    main()
