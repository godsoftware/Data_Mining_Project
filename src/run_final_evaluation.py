"""Evaluate the selected final model, thresholds, and calibration."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt

from config.settings import FN_COST, FP_COST, RANDOM_SEED
from data_preprocessing import PROJECT_ROOT, TARGET_COLUMN
from evaluation import (
    binary_classification_metrics,
    plot_roc_and_pr_curves,
    threshold_optimization_table,
)
from model_training import split_features_target


PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "taiwan_model_ready.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "final_model.joblib"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def main() -> None:
    """Save final test metrics, threshold table, and calibration comparison."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split = split_features_target(df, target_column=TARGET_COLUMN, random_state=RANDOM_SEED)
    final_model = joblib.load(MODEL_PATH)

    final_proba = final_model.predict_proba(split.X_test)[:, 1]
    final_metrics = binary_classification_metrics(split.y_test, final_proba, threshold=0.5)
    final_metrics["model"] = "final_model_uncalibrated"

    threshold_table = threshold_optimization_table(
        split.y_test,
        final_proba,
        false_negative_cost=FN_COST,
        false_positive_cost=FP_COST,
    )
    threshold_table.to_csv(TABLES_DIR / "threshold_comparison.csv", index=False)

    calibration_rows = [final_metrics]
    calibrated_probabilities = {"uncalibrated_final_model": final_proba}

    for method in ("sigmoid", "isotonic"):
        print(f"Fitting {method} calibration...")
        calibrated = CalibratedClassifierCV(final_model, method=method, cv=3)
        calibrated.fit(split.X_train, split.y_train)
        y_proba = calibrated.predict_proba(split.X_test)[:, 1]
        metrics = binary_classification_metrics(split.y_test, y_proba, threshold=0.5)
        metrics["model"] = f"final_model_calibrated_{method}"
        calibration_rows.append(metrics)
        calibrated_probabilities[f"calibrated_{method}"] = y_proba
        joblib.dump(calibrated, MODELS_DIR / f"final_model_calibrated_{method}.joblib")

    pd.DataFrame([final_metrics]).to_csv(TABLES_DIR / "final_test_metrics.csv", index=False)
    calibration_results = pd.DataFrame(calibration_rows).sort_values("brier_score")
    calibration_results.to_csv(TABLES_DIR / "calibration_results.csv", index=False)

    plot_roc_and_pr_curves(calibrated_probabilities, split.y_test, FIGURES_DIR)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, y_proba in calibrated_probabilities.items():
        CalibrationDisplay.from_predictions(
            split.y_test,
            y_proba,
            n_bins=10,
            name=label,
            ax=ax,
        )
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300)
    plt.close(fig)

    print(calibration_results[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"]
    ].to_string(index=False))
    print("Best threshold rows by expected cost:")
    print(threshold_table[
        ["threshold", "precision", "recall", "f1", "fp", "fn", "expected_cost"]
    ].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
