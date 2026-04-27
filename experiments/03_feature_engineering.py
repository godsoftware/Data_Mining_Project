"""Build Taiwan and HELOC model-ready feature tables."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment_registry import finish_run, start_run
from feature_engineering import main as build_taiwan_features
from heloc_preprocessing import main as build_heloc_features


def main() -> None:
    """Run the row-wise feature-engineering phase for both active datasets."""

    run_id = start_run("feature_engineering", dataset="both", tags=["phase-4", "features"])
    try:
        build_taiwan_features()
        build_heloc_features()
        finish_run(run_id)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
