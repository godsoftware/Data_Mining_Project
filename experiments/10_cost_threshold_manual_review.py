"""Run Phase 10 cost-sensitive threshold and manual-review-band analysis."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_cost_threshold_analysis import main


if __name__ == "__main__":
    main()
