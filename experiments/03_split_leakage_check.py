"""Create split summaries and leakage-control checklist."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import build_leakage_checklist, build_split_summary
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_TARGET
from src.config.paths import HELOC_MODEL_READY, TABLES_DIR, TAIWAN_MODEL_READY


def main() -> None:
    """Generate Phase 5 reproducibility outputs."""

    run_id = start_run("split_leakage_check", dataset="both", tags=["phase-5", "leakage-control"])
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        taiwan = pd.read_csv(TAIWAN_MODEL_READY)
        heloc = pd.read_csv(HELOC_MODEL_READY)

        taiwan_summary = build_split_summary("taiwan", taiwan, TARGET_COLUMN)
        heloc_summary = build_split_summary("heloc", heloc, HELOC_TARGET)
        leakage = build_leakage_checklist()

        taiwan_summary.to_csv(TABLES_DIR / "split_summary_taiwan.csv", index=False)
        heloc_summary.to_csv(TABLES_DIR / "split_summary_heloc.csv", index=False)
        leakage.to_csv(TABLES_DIR / "leakage_checklist.csv", index=False)

        print(taiwan_summary.to_string(index=False))
        print(heloc_summary.to_string(index=False))
        print(leakage.to_string(index=False))

        finish_run(
            run_id,
            metrics={
                "taiwan_train_rows": int(taiwan_summary.loc[taiwan_summary["split"] == "train", "rows"].iloc[0]),
                "heloc_train_rows": int(heloc_summary.loc[heloc_summary["split"] == "train", "rows"].iloc[0]),
                "leakage_checks": int(len(leakage)),
                "failed_leakage_checks": int((leakage["status"] != "pass").sum()),
            },
            artifacts={
                "split_summary_taiwan": TABLES_DIR / "split_summary_taiwan.csv",
                "split_summary_heloc": TABLES_DIR / "split_summary_heloc.csv",
                "leakage_checklist": TABLES_DIR / "leakage_checklist.csv",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
