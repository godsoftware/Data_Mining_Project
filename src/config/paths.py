"""Central filesystem paths for the SCRE-Credit project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

TAIWAN_RAW_DIR = RAW_DIR / "taiwan_default"
TAIWAN_RAW_XLS = TAIWAN_RAW_DIR / "default_credit_card_clients.xls"
TAIWAN_RAW_CSV = TAIWAN_RAW_DIR / "default_credit_card_clients.csv"
TAIWAN_CLEANED = INTERIM_DIR / "taiwan_cleaned.csv"
TAIWAN_MODEL_READY = PROCESSED_DIR / "taiwan_model_ready.csv"

HELOC_RAW_DIR = RAW_DIR / "heloc"
HELOC_RAW_CSV = HELOC_RAW_DIR / "heloc_dataset.csv"
HELOC_CLEANED = INTERIM_DIR / "heloc_cleaned.csv"
HELOC_MODEL_READY = PROCESSED_DIR / "heloc_model_ready.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baselines"
BOOSTING_MODELS_DIR = MODELS_DIR / "boosting"
MONOTONIC_MODELS_DIR = MODELS_DIR / "monotonic"
SCORECARD_MODELS_DIR = MODELS_DIR / "scorecard"
CALIBRATED_MODELS_DIR = MODELS_DIR / "calibrated"
SHAP_DIR = OUTPUTS_DIR / "shap"
LIME_DIR = OUTPUTS_DIR / "lime"
STABILITY_DIR = OUTPUTS_DIR / "stability"
FAITHFULNESS_DIR = OUTPUTS_DIR / "faithfulness"
CALIBRATION_DIR = OUTPUTS_DIR / "calibration"
EXPERIMENT_LOGS_DIR = OUTPUTS_DIR / "experiment_logs"


def ensure_project_dirs() -> None:
    """Create the standard directory tree."""

    for path in [
        TAIWAN_RAW_DIR,
        HELOC_RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        MODELS_DIR,
        BASELINE_MODELS_DIR,
        BOOSTING_MODELS_DIR,
        MONOTONIC_MODELS_DIR,
        SCORECARD_MODELS_DIR,
        CALIBRATED_MODELS_DIR,
        SHAP_DIR,
        LIME_DIR,
        STABILITY_DIR,
        FAITHFULNESS_DIR,
        CALIBRATION_DIR,
        EXPERIMENT_LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
