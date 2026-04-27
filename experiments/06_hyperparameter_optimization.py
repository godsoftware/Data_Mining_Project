"""Run Phase 8 Optuna hyperparameter optimization."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_hyperparameter_optimization import main


if __name__ == "__main__":
    main()
