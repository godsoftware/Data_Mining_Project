"""Check the SCRE-Credit environment and folder layout."""

from __future__ import annotations

import importlib
from importlib import metadata
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import TABLES_DIR, ensure_project_dirs
from experiment_registry import finish_run, start_run


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",
    "imblearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "shap",
    "lime",
    "yaml",
    "statsmodels",
    "pytest",
]


def package_version(package: str) -> str:
    """Return a package version from the imported module or package metadata."""

    module = importlib.import_module(package)
    version = getattr(module, "__version__", None)
    if version:
        return str(version)

    metadata_name = {"sklearn": "scikit-learn", "yaml": "PyYAML"}.get(package, package)
    try:
        return metadata.version(metadata_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    run_id = start_run("environment_check", dataset="environment", tags=["phase-1", "reproducibility"])
    ensure_project_dirs()
    rows = [{"package": "python", "version": sys.version, "import_name": "python"}]
    try:
        print(f"Python: {sys.version}")
        for package in REQUIRED_PACKAGES:
            version = package_version(package)
            rows.append({"package": package, "version": version, "import_name": package})
            print(f"{package}: {version}")
        output_path = TABLES_DIR / "environment_versions.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        finish_run(
            run_id,
            metrics={"package_count": len(REQUIRED_PACKAGES)},
            artifacts={"environment_versions": output_path},
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
