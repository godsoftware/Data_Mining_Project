"""Evaluation, threshold optimization, and calibration helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins[1:-1], right=True)
    ece = 0.0

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_proba[mask]))
        observed = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(observed - confidence)

    return float(ece)


def calibration_slope_intercept(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Estimate calibration slope/intercept by regressing outcome on logit score."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    clipped = np.clip(y_proba, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y_true)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def binary_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute metrics that matter for imbalanced default prediction."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0)
    g_mean = float(np.sqrt(recall * specificity))
    calibration_slope, calibration_intercept = calibration_slope_intercept(y_true, y_proba)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "g_mean": g_mean,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "ece": expected_calibration_error(y_true, y_proba),
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_optimization_table(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    thresholds: np.ndarray | None = None,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> pd.DataFrame:
    """Evaluate precision/recall/F1 and expected cost over many thresholds."""

    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

    rows = []
    for threshold in thresholds:
        metrics = binary_classification_metrics(y_true, y_proba, threshold)
        metrics["fn_cost_component"] = false_negative_cost * metrics["fn"]
        metrics["fp_cost_component"] = false_positive_cost * metrics["fp"]
        metrics["expected_cost"] = metrics["fn_cost_component"] + metrics["fp_cost_component"]
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values(["expected_cost", "threshold"])


def save_model_comparison(rows: list[dict], path: str | Path) -> pd.DataFrame:
    """Save a model comparison table and return it."""

    table = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def plot_roc_and_pr_curves(
    model_probabilities: dict[str, np.ndarray],
    y_true: np.ndarray | pd.Series,
    output_dir: str | Path,
) -> None:
    """Save ROC and precision-recall curves for multiple models."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    for model_name, y_proba in model_probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    for model_name, y_proba in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, label=f"{model_name} (AP={auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curves.png", dpi=300)
    plt.close()


def plot_calibration_curve(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str | Path,
    model_name: str = "Final model",
    n_bins: int = 10,
) -> None:
    """Save a calibration curve for a fitted probability model."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    CalibrationDisplay.from_estimator(
        estimator,
        X_test,
        y_test,
        n_bins=n_bins,
        name=model_name,
        ax=ax,
    )
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
