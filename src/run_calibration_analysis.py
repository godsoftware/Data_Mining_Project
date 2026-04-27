"""Phase 9 calibration analysis for the SCRE-Credit model pool."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from config.settings import FN_COST, FP_COST, RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET
from monotonic_models import HELOC_MONOTONIC_PRIORS, TAIWAN_MONOTONIC_PRIORS
from run_baseline_models import build_baseline_models
from run_blackbox_constrained_models import build_phase7_models
from run_hyperparameter_optimization import build_model_from_best_params
from scre_credit import make_model_pipeline
from src.config.paths import CALIBRATED_MODELS_DIR, FIGURES_DIR, HELOC_MODEL_READY, TABLES_DIR, TAIWAN_MODEL_READY


warnings.filterwarnings("ignore", category=UserWarning)


def _safe_name(name: str) -> str:
    """Return a filesystem-friendly model or calibration name."""

    return name.lower().replace(" + ", "_").replace(" / ", "_").replace("/", "_").replace(" ", "_").replace("-", "_")


DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "categorical_columns": ["SEX", "EDUCATION", "MARRIAGE"],
        "monotonic_priors": TAIWAN_MONOTONIC_PRIORS,
        "output": TABLES_DIR / "calibration_results_taiwan.csv",
        "figure": FIGURES_DIR / "calibration_curve_taiwan.png",
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "categorical_columns": HELOC_CATEGORICAL_COLUMNS,
        "monotonic_priors": HELOC_MONOTONIC_PRIORS,
        "output": TABLES_DIR / "calibration_results_heloc.csv",
        "figure": FIGURES_DIR / "calibration_curve_heloc.png",
    },
}


@dataclass
class CalibrationModelSpec:
    """Model definition used by the Phase 9 calibration experiment."""

    name: str
    estimator: object
    source_phase: str
    model_family: str
    resampling: str = "none"
    monotonic_constraints: bool = False
    description: str = ""


def _positive_class_probability(estimator, X: pd.DataFrame) -> np.ndarray:
    """Return the positive-class probability for fitted binary classifiers."""

    probabilities = estimator.predict_proba(X)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Estimator did not return two-column class probabilities.")
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return probabilities[:, -1]


def _load_tuned_spec(
    dataset: str,
    X_train: pd.DataFrame,
    categorical_columns: list[str],
) -> CalibrationModelSpec | None:
    """Build the Phase 8 best model from outputs/tables/best_params.csv when available."""

    best_params_path = TABLES_DIR / "best_params.csv"
    if not best_params_path.exists():
        return None

    best_params = pd.read_csv(best_params_path)
    matching = best_params.loc[best_params["dataset"] == dataset]
    if matching.empty:
        return None

    row = matching.iloc[0]
    params = json.loads(row["best_params_json"])
    model = build_model_from_best_params(params, random_state=RANDOM_SEED)
    estimator = make_model_pipeline(model, X_train, categorical_columns)
    return CalibrationModelSpec(
        name=f"optuna_best_{params['model_type']}",
        estimator=estimator,
        source_phase="phase8_hyperparameter_optimization",
        model_family="tuned_boosting",
        description=(
            f"Best Phase 8 Optuna model for {dataset}; "
            f"trial={row['best_trial_number']}, objective=PR-AUC."
        ),
    )


def build_calibration_model_specs(
    dataset: str,
    X_train: pd.DataFrame,
    categorical_columns: list[str],
) -> list[CalibrationModelSpec]:
    """Create the full model list for Phase 9 calibration."""

    config = DATASET_CONFIGS[dataset]
    specs: list[CalibrationModelSpec] = []

    for name, estimator, description in build_baseline_models(
        X_train,
        categorical_columns=categorical_columns,
        random_state=RANDOM_SEED,
    ):
        specs.append(
            CalibrationModelSpec(
                name=name,
                estimator=estimator,
                source_phase="phase6_baseline",
                model_family="baseline",
                description=description,
            )
        )

    for spec in build_phase7_models(
        dataset,
        X_train,
        categorical_columns=categorical_columns,
        monotonic_priors=config["monotonic_priors"],
        random_state=RANDOM_SEED,
    ):
        specs.append(
            CalibrationModelSpec(
                name=spec.name,
                estimator=spec.estimator,
                source_phase="phase7_model_pool",
                model_family=spec.family,
                resampling=spec.resampling,
                monotonic_constraints=spec.monotonic,
                description=spec.constraint_summary,
            )
        )

    tuned_spec = _load_tuned_spec(dataset, X_train, categorical_columns)
    if tuned_spec is not None:
        specs.append(tuned_spec)

    return specs


def _fit_probability_model(estimator, calibration_method: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit an uncalibrated or train-CV calibrated probability model."""

    base_estimator = clone(estimator)
    if calibration_method == "uncalibrated":
        model = base_estimator
    else:
        model = CalibratedClassifierCV(
            estimator=base_estimator,
            method=calibration_method,
            cv=3,
            n_jobs=1,
        )
    model.fit(X_train, y_train)
    return model


def _metric_row(
    dataset: str,
    spec: CalibrationModelSpec,
    calibration_method: str,
    y_validation: pd.Series,
    y_proba: np.ndarray,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> dict:
    """Build one calibration-result table row."""

    metrics = binary_classification_metrics(y_validation, y_proba, threshold=0.5)
    return {
        "dataset": dataset,
        "model": spec.name,
        "source_phase": spec.source_phase,
        "model_family": spec.model_family,
        "calibration_method": calibration_method,
        "calibration_fit_policy": (
            "uncalibrated_train_fit"
            if calibration_method == "uncalibrated"
            else f"{calibration_method}_calibrated_classifier_cv_on_train_only"
        ),
        "split": "validation",
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "test_rows_reserved": int(test_rows),
        "test_set_used": False,
        "calibration_cv_folds": 0 if calibration_method == "uncalibrated" else 3,
        "resampling": spec.resampling,
        "monotonic_constraints": spec.monotonic_constraints,
        "expected_cost": float(FN_COST * metrics["fn"] + FP_COST * metrics["fp"]),
        "description": spec.description,
        **metrics,
    }


def plot_calibration_curves(
    results: pd.DataFrame,
    probability_records: dict[tuple[str, str], np.ndarray],
    y_validation: pd.Series,
    output_path: Path,
    dataset: str,
    n_bins: int = 10,
    max_models: int = 5,
) -> None:
    """Save a readable multi-panel calibration curve for the strongest models."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    uncalibrated = results.loc[results["calibration_method"] == "uncalibrated"].copy()
    selected_models = (
        uncalibrated.sort_values(["pr_auc", "roc_auc"], ascending=False)["model"]
        .head(max_models)
        .tolist()
    )

    methods = ["uncalibrated", "sigmoid", "isotonic"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, method in zip(axes, methods):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#707070", linewidth=1, label="Perfect")
        for model_name in selected_models:
            key = (model_name, method)
            if key not in probability_records:
                continue
            prob_true, prob_pred = calibration_curve(
                y_validation,
                probability_records[key],
                n_bins=n_bins,
                strategy="uniform",
            )
            ax.plot(prob_pred, prob_true, marker="o", linewidth=1.5, markersize=4, label=model_name)
        ax.set_title(method.title())
        ax.set_xlabel("Mean Predicted Probability")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Observed Default Rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(f"{dataset.upper()} Calibration Curves - Top Validation Models", y=1.02)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_calibration_for_dataset(dataset: str) -> pd.DataFrame:
    """Run uncalibrated, sigmoid, and isotonic calibration for one dataset."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])

    X_train = split.train.drop(columns=[config["target"]])
    y_train = split.train[config["target"]]
    X_validation = split.validation.drop(columns=[config["target"]])
    y_validation = split.validation[config["target"]]

    specs = build_calibration_model_specs(
        dataset,
        X_train,
        categorical_columns=config["categorical_columns"],
    )

    rows: list[dict] = []
    probability_records: dict[tuple[str, str], np.ndarray] = {}
    model_dir = CALIBRATED_MODELS_DIR / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for calibration_method in ["uncalibrated", "sigmoid", "isotonic"]:
            print(f"Calibrating {dataset}: {spec.name} [{calibration_method}]")
            fitted = _fit_probability_model(spec.estimator, calibration_method, X_train, y_train)
            model_path = model_dir / f"{_safe_name(spec.name)}_{calibration_method}.joblib"
            joblib.dump(fitted, model_path)
            y_proba = _positive_class_probability(fitted, X_validation)
            probability_records[(spec.name, calibration_method)] = y_proba
            row = _metric_row(
                dataset=dataset,
                spec=spec,
                calibration_method=calibration_method,
                y_validation=y_validation,
                y_proba=y_proba,
                train_rows=len(split.train),
                validation_rows=len(split.validation),
                test_rows=len(split.test),
            )
            row["model_path"] = str(model_path)
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(
        ["calibration_method", "brier_score", "ece", "pr_auc"],
        ascending=[True, True, True, False],
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config["output"], index=False)
    plot_calibration_curves(
        results,
        probability_records,
        y_validation,
        config["figure"],
        dataset=dataset,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    args = parser.parse_args()

    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    run_id = start_run("calibration_analysis", dataset=args.dataset, tags=["phase-9", "calibration"])
    try:
        artifacts = {}
        metrics = {}
        for dataset in datasets:
            results = run_calibration_for_dataset(dataset)
            output_path = DATASET_CONFIGS[dataset]["output"]
            figure_path = DATASET_CONFIGS[dataset]["figure"]
            artifacts[f"{dataset}_calibration_results"] = output_path
            artifacts[f"{dataset}_calibration_curve"] = figure_path
            best_brier = results.sort_values("brier_score").iloc[0]
            best_ece = results.sort_values("ece").iloc[0]
            metrics[f"{dataset}_best_brier_score"] = float(best_brier["brier_score"])
            metrics[f"{dataset}_best_brier_model"] = str(best_brier["model"])
            metrics[f"{dataset}_best_brier_calibration"] = str(best_brier["calibration_method"])
            metrics[f"{dataset}_best_ece"] = float(best_ece["ece"])
            print(
                results[
                    [
                        "dataset",
                        "model",
                        "calibration_method",
                        "brier_score",
                        "ece",
                        "calibration_slope",
                        "calibration_intercept",
                        "pr_auc",
                    ]
                ]
                .sort_values(["brier_score", "ece"])
                .head(12)
                .to_string(index=False)
            )

        finish_run(run_id, metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
