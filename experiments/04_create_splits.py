"""Create stratified split summaries and leakage checklist."""

from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the canonical split/leakage-check script."""

    runpy.run_path(str(PROJECT_ROOT / "experiments" / "03_split_leakage_check.py"), run_name="__main__")


if __name__ == "__main__":
    main()
