"""Train Phase 7 black-box and constrained models."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_blackbox_constrained_models import main


if __name__ == "__main__":
    main()
