"""Retrain boosting models with controlled class-imbalance strategies.

This experiment does not create new features. It fits CatBoost, LightGBM, and
XGBoost variants on the frozen train split, calibrates probabilities using the
validation split, selects threshold policies on validation only, and evaluates
the selected policies once on the held-out test split.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.split_leakage import stratified_train_validation_test_split  # noqa: E402
from data_preprocessing import TARGET_COLUMN  # noqa: E402
from evaluation import expected_calibration_error  # noqa: E402
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET  # noqa: E402
from scre_credit import make_model_pipeline  # noqa: E402
from src.config.paths import HELOC_MODEL_READY, OUTPUTS_DIR, TAIWAN_MODEL_READY  # noqa: E402

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("CatBoost is required for Prompt 5 imbalance revision.") from exc

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("LightGBM is required for Prompt 5 imbalance revision.") from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("XGBoost is required for Prompt 5 imbalance revision.") from exc


DECISION_REVISION_DIR = OUTPUTS_DIR / "decision_revision"
RANDOM_SEED = 42
FN_COST = 5.0
FP_COST = 1.0
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


@dataclass(frozen=True)
class VariantSpec:
    """One model/imbalance strategy configuration."""

    model_family: str
    variant_name: str
    imbalance_strategy: str
    estimator: Any
    fit_sample_weight_positive: float | None = None
    experimental: bool = False


def catboost_estimator(**params: Any) -> CatBoostClassifier:
    """Build a CatBoost model with the project's standard capacity."""

    base_params = {
        "iterations": 400,
        "learning_rate": 0.04,
        "depth": 5,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    base_params.update(params)
    return CatBoostClassifier(**base_params)


def lightgbm_estimator(**params: Any) -> LGBMClassifier:
    """Build a LightGBM model with the project's standard capacity."""

    base_params = {
        "n_estimators": 500,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 40,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    base_params.update(params)
    return LGBMClassifier(**base_params)


def xgboost_estimator(**params: Any) -> XGBClassifier:
    """Build an XGBoost model with the project's standard capacity."""

    base_params = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    base_params.update(params)
    return XGBClassifier(**base_params)


def build_variant_specs() -> list[VariantSpec]:
    """Return all requested model imbalance variants."""

    specs: list[VariantSpec] = [
        VariantSpec("CatBoost", "catboost_baseline", "baseline CatBoost", catboost_estimator()),
        VariantSpec(
            "CatBoost",
            "catboost_auto_none",
            "auto_class_weights=None",
            catboost_estimator(auto_class_weights=None),
        ),
        VariantSpec(
            "CatBoost",
            "catboost_auto_balanced",
            "auto_class_weights='Balanced'",
            catboost_estimator(auto_class_weights="Balanced"),
        ),
        VariantSpec(
            "CatBoost",
            "catboost_auto_sqrtbalanced",
            "auto_class_weights='SqrtBalanced'",
            catboost_estimator(auto_class_weights="SqrtBalanced"),
        ),
    ]
    for weight in [1.5, 2.0, 3.0, 4.0, 5.0]:
        specs.append(
            VariantSpec(
                "CatBoost",
                f"catboost_class_weight_pos_{weight:g}",
                f"custom class_weights positive_weight={weight:g}",
                catboost_estimator(class_weights=[1.0, weight]),
            )
        )

    specs.extend(
        [
            VariantSpec("LightGBM", "lightgbm_baseline", "baseline LightGBM", lightgbm_estimator()),
            VariantSpec("LightGBM", "lightgbm_is_unbalance", "is_unbalance=True", lightgbm_estimator(is_unbalance=True)),
            VariantSpec(
                "LightGBM",
                "lightgbm_class_weight_balanced",
                "class_weight='balanced'",
                lightgbm_estimator(class_weight="balanced"),
            ),
        ]
    )
    for weight in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        specs.append(
            VariantSpec(
                "LightGBM",
                f"lightgbm_scale_pos_weight_{weight:g}",
                f"scale_pos_weight={weight:g}",
                lightgbm_estimator(scale_pos_weight=weight),
            )
        )

    specs.extend(
        [
            VariantSpec("XGBoost", "xgboost_baseline", "baseline XGBoost", xgboost_estimator()),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_1", "scale_pos_weight=1", xgboost_estimator(scale_pos_weight=1.0)),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_1_5", "scale_pos_weight=1.5", xgboost_estimator(scale_pos_weight=1.5)),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_2", "scale_pos_weight=2", xgboost_estimator(scale_pos_weight=2.0)),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_2_5", "scale_pos_weight=2.5", xgboost_estimator(scale_pos_weight=2.5)),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_3", "scale_pos_weight=3", xgboost_estimator(scale_pos_weight=3.0)),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_3_5", "scale_pos_weight=3.5", xgboost_estimator(scale_pos_weight=3.5)),
            VariantSpec("XGBoost", "xgboost_scale_pos_weight_4", "scale_pos_weight=4", xgboost_estimator(scale_pos_weight=4.0)),
            VariantSpec("XGBoost", "xgboost_max_delta_step_1", "max_delta_step=1", xgboost_estimator(max_delta_step=1)),
            VariantSpec("XGBoost", "xgboost_max_delta_step_3", "max_delta_step=3", xgboost_estimator(max_delta_step=3)),
            VariantSpec("XGBoost", "xgboost_max_delta_step_5", "max_delta_step=5", xgboost_estimator(max_delta_step=5)),
        ]
    )
    for weight in [1.5, 2.0, 3.0]:
        specs.append(
            VariantSpec(
                "XGBoost",
                f"xgboost_sample_weight_pos_{weight:g}",
                f"sample_weight positive class={weight:g}",
                xgboost_estimator(),
                fit_sample_weight_positive=weight,
            )
        )
    return specs


def positive_class_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class probability from a fitted estimator."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        probabilities = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"Estimator does not expose class 1: {classes}")
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def fit_pipeline(spec: VariantSpec, X_train: pd.DataFrame, y_train: pd.Series, categorical_columns: list[str]) -> Any:
    """Fit one model pipeline with optional class-specific sample weights."""

    pipeline = make_model_pipeline(spec.estimator, X_train, categorical_columns)
    fit_kwargs: dict[str, Any] = {}
    if spec.fit_sample_weight_positive is not None:
        fit_kwargs["model__sample_weight"] = np.where(
            np.asarray(y_train, dtype=int) == 1,
            float(spec.fit_sample_weight_positive),
            1.0,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pipeline.fit(X_train, y_train, **fit_kwargs)
    return pipeline


def safe_logit(probability: np.ndarray) -> np.ndarray:
    """Return clipped logit values as a calibration feature."""

    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def calibrated_probabilities(
    calibration_type: str,
    y_validation: pd.Series,
    validation_probability: np.ndarray,
    test_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit calibration on validation probabilities and transform validation/test."""

    validation_probability = np.asarray(validation_probability, dtype=float)
    test_probability = np.asarray(test_probability, dtype=float)
    if calibration_type == "uncalibrated":
        return validation_probability, test_probability
    if calibration_type == "sigmoid":
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibrator.fit(safe_logit(validation_probability), y_validation)
        return (
            calibrator.predict_proba(safe_logit(validation_probability))[:, 1],
            calibrator.predict_proba(safe_logit(test_probability))[:, 1],
        )
    if calibration_type == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(validation_probability, y_validation)
        return calibrator.transform(validation_probability), calibrator.transform(test_probability)
    raise ValueError(f"Unknown calibration type: {calibration_type}")


def threshold_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
    brier: float | None = None,
    ece: float | None = None,
) -> dict[str, float | int]:
    """Compute binary metrics quickly at one threshold."""

    prediction = np.asarray(probability) >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    tn = int(np.sum((~prediction) & negative))
    fn = int(np.sum((~prediction) & positive))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc if roc_auc is not None else roc_auc_score(y_true, probability)),
        "pr_auc": float(pr_auc if pr_auc is not None else average_precision_score(y_true, probability)),
        "brier": float(brier if brier is not None else brier_score_loss(y_true, probability)),
        "ece": float(ece if ece is not None else expected_calibration_error(y_true, probability)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "expected_cost_FN5_FP1": float(FN_COST * fn + FP_COST * fp),
    }


def threshold_grid(y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Evaluate all threshold policies on validation."""

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    roc_auc = float(roc_auc_score(y, probability))
    pr_auc = float(average_precision_score(y, probability))
    brier = float(brier_score_loss(y, probability))
    ece = float(expected_calibration_error(y, probability))
    rows = [
        threshold_metrics(
            y,
            probability,
            threshold=float(threshold),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            brier=brier,
            ece=ece,
        )
        for threshold in THRESHOLDS
    ]
    return pd.DataFrame(rows)


def select_threshold(policy: str, validation_grid: pd.DataFrame) -> pd.Series | None:
    """Select a threshold on validation under the requested policy."""

    if policy == "default_0_50":
        idx = (validation_grid["threshold"] - 0.50).abs().idxmin()
        return validation_grid.loc[idx]
    if policy == "cost_optimal_FN5_FP1":
        candidates = validation_grid
    elif policy == "precision_0_40":
        candidates = validation_grid.loc[validation_grid["precision"] >= 0.40]
    elif policy == "precision_0_45":
        candidates = validation_grid.loc[validation_grid["precision"] >= 0.45]
    elif policy == "specificity_0_65":
        candidates = validation_grid.loc[validation_grid["specificity"] >= 0.65]
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["expected_cost_FN5_FP1", "f1", "pr_auc", "threshold"],
        ascending=[True, False, False, False],
    ).iloc[0]


def calibration_comment(calibration_type: str, uncalibrated: dict[str, float], calibrated: dict[str, float]) -> str:
    """Summarize validation calibration movement."""

    if calibration_type == "uncalibrated":
        return "Reference uncalibrated probability."
    brier_delta = calibrated["brier"] - uncalibrated["brier"]
    ece_delta = calibrated["ece"] - uncalibrated["ece"]
    if brier_delta < 0 and ece_delta < 0:
        return "Calibration improves validation Brier and ECE."
    if ece_delta < 0:
        return "Calibration improves ECE but not Brier."
    if brier_delta < 0:
        return "Calibration improves Brier but not ECE."
    return "Calibration does not improve validation Brier/ECE."


def operational_comment(row: dict[str, Any]) -> str:
    """Summarize the operational trade-off versus baseline CatBoost."""

    precision_delta = float(row["delta_precision_vs_baseline_catboost"])
    recall_delta = float(row["delta_recall_vs_baseline_catboost"])
    fp_delta = float(row["delta_fp_vs_baseline_catboost"])
    cost_delta = float(row["delta_cost_vs_baseline_catboost"])
    if precision_delta > 0 and fp_delta < 0 and recall_delta > -0.20:
        return "More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss."
    if precision_delta > 0 and fp_delta < 0:
        return "Precision/FP improve but recall loss is material."
    if recall_delta > 0 and precision_delta < 0:
        return "Weighting mainly increases recall and may worsen FP burden."
    if cost_delta < 0:
        return "Cost improves, but precision/recall trade-off needs review."
    return "No clear operational improvement over baseline CatBoost."


def dataset_config(dataset: str) -> dict[str, Any]:
    """Return dataset path/target/categorical columns."""

    if dataset == "taiwan":
        return {
            "path": TAIWAN_MODEL_READY,
            "target": TARGET_COLUMN,
            "categorical_columns": ["SEX", "EDUCATION", "MARRIAGE"],
        }
    if dataset == "heloc":
        return {
            "path": HELOC_MODEL_READY,
            "target": HELOC_TARGET,
            "categorical_columns": HELOC_CATEGORICAL_COLUMNS,
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def build_dataset_results(dataset: str) -> pd.DataFrame:
    """Train and evaluate all imbalance strategy variants for one dataset."""

    config = dataset_config(dataset)
    frame = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(frame, target_column=config["target"])
    X_train = split.train.drop(columns=[config["target"]])
    y_train = split.train[config["target"]].astype(int)
    X_validation = split.validation.drop(columns=[config["target"]])
    y_validation = split.validation[config["target"]].astype(int)
    X_test = split.test.drop(columns=[config["target"]])
    y_test = split.test[config["target"]].astype(int)

    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(build_variant_specs(), start=1):
        print(f"[{dataset}] Training {index:02d}/33: {spec.variant_name}")
        fitted = fit_pipeline(spec, X_train, y_train, config["categorical_columns"])
        base_validation_probability = positive_class_probability(fitted, X_validation)
        base_test_probability = positive_class_probability(fitted, X_test)

        uncalibration_reference = {
            "brier": float(brier_score_loss(y_validation, base_validation_probability)),
            "ece": float(expected_calibration_error(y_validation, base_validation_probability)),
        }

        for calibration_type in ["uncalibrated", "sigmoid", "isotonic"]:
            validation_probability, test_probability = calibrated_probabilities(
                calibration_type,
                y_validation,
                base_validation_probability,
                base_test_probability,
            )
            validation_grid = threshold_grid(y_validation, validation_probability)
            calibrated_reference = {
                "brier": float(validation_grid.iloc[0]["brier"]),
                "ece": float(validation_grid.iloc[0]["ece"]),
            }
            for policy in [
                "default_0_50",
                "cost_optimal_FN5_FP1",
                "precision_0_40",
                "precision_0_45",
                "specificity_0_65",
            ]:
                selected = select_threshold(policy, validation_grid)
                row = {
                    "dataset": dataset,
                    "model_family": spec.model_family,
                    "variant_name": spec.variant_name,
                    "imbalance_strategy": spec.imbalance_strategy,
                    "experimental": spec.experimental,
                    "calibration_type": calibration_type,
                    "threshold_policy": policy,
                    "selection_split": "validation",
                    "test_set_used_for_calibration_or_threshold_selection": False,
                    "train_rows": int(len(X_train)),
                    "validation_rows": int(len(X_validation)),
                    "test_rows": int(len(X_test)),
                    "validation_roc_auc": float(validation_grid.iloc[0]["roc_auc"]),
                    "validation_pr_auc": float(validation_grid.iloc[0]["pr_auc"]),
                    "validation_brier": float(validation_grid.iloc[0]["brier"]),
                    "validation_ece": float(validation_grid.iloc[0]["ece"]),
                    "probability_calibration_comment": calibration_comment(
                        calibration_type,
                        uncalibration_reference,
                        calibrated_reference,
                    ),
                }
                if selected is None:
                    for column in [
                        "selected_threshold_validation",
                        "validation_precision",
                        "validation_recall",
                        "validation_specificity",
                        "validation_f1",
                        "validation_expected_cost_FN5_FP1",
                        "test_accuracy",
                        "test_precision",
                        "test_recall",
                        "test_specificity",
                        "test_f1",
                        "test_roc_auc",
                        "test_pr_auc",
                        "test_brier",
                        "test_ece",
                        "test_tn",
                        "test_fp",
                        "test_fn",
                        "test_tp",
                        "test_expected_cost_FN5_FP1",
                    ]:
                        row[column] = np.nan
                    rows.append(row)
                    continue

                test_metrics = threshold_metrics(
                    np.asarray(y_test, dtype=int),
                    np.asarray(test_probability, dtype=float),
                    threshold=float(selected["threshold"]),
                )
                row.update(
                    {
                        "selected_threshold_validation": float(selected["threshold"]),
                        "validation_precision": float(selected["precision"]),
                        "validation_recall": float(selected["recall"]),
                        "validation_specificity": float(selected["specificity"]),
                        "validation_f1": float(selected["f1"]),
                        "validation_expected_cost_FN5_FP1": float(selected["expected_cost_FN5_FP1"]),
                        "test_accuracy": float(test_metrics["accuracy"]),
                        "test_precision": float(test_metrics["precision"]),
                        "test_recall": float(test_metrics["recall"]),
                        "test_specificity": float(test_metrics["specificity"]),
                        "test_f1": float(test_metrics["f1"]),
                        "test_roc_auc": float(test_metrics["roc_auc"]),
                        "test_pr_auc": float(test_metrics["pr_auc"]),
                        "test_brier": float(test_metrics["brier"]),
                        "test_ece": float(test_metrics["ece"]),
                        "test_tn": int(test_metrics["tn"]),
                        "test_fp": int(test_metrics["fp"]),
                        "test_fn": int(test_metrics["fn"]),
                        "test_tp": int(test_metrics["tp"]),
                        "test_expected_cost_FN5_FP1": float(test_metrics["expected_cost_FN5_FP1"]),
                    }
                )
                rows.append(row)

    result = pd.DataFrame(rows)
    return add_baseline_deltas(result)


def add_baseline_deltas(result: pd.DataFrame) -> pd.DataFrame:
    """Add deltas versus retrained baseline CatBoost cost-optimal row."""

    baseline = result.loc[
        result["variant_name"].eq("catboost_baseline")
        & result["calibration_type"].eq("uncalibrated")
        & result["threshold_policy"].eq("cost_optimal_FN5_FP1")
    ].iloc[0]
    rows: list[dict[str, Any]] = []
    for row in result.to_dict(orient="records"):
        row["baseline_catboost_reference_variant"] = "catboost_baseline|uncalibrated|cost_optimal_FN5_FP1"
        row["baseline_catboost_precision"] = float(baseline["test_precision"])
        row["baseline_catboost_recall"] = float(baseline["test_recall"])
        row["baseline_catboost_fp"] = int(baseline["test_fp"])
        row["baseline_catboost_cost"] = float(baseline["test_expected_cost_FN5_FP1"])
        row["delta_fp_vs_baseline_catboost"] = int(row["test_fp"]) - int(baseline["test_fp"]) if pd.notna(row["test_fp"]) else np.nan
        row["delta_precision_vs_baseline_catboost"] = (
            float(row["test_precision"]) - float(baseline["test_precision"])
            if pd.notna(row["test_precision"])
            else np.nan
        )
        row["delta_recall_vs_baseline_catboost"] = (
            float(row["test_recall"]) - float(baseline["test_recall"])
            if pd.notna(row["test_recall"])
            else np.nan
        )
        row["delta_cost_vs_baseline_catboost"] = (
            float(row["test_expected_cost_FN5_FP1"]) - float(baseline["test_expected_cost_FN5_FP1"])
            if pd.notna(row["test_expected_cost_FN5_FP1"])
            else np.nan
        )
        row["operational_comment"] = operational_comment(row) if pd.notna(row["test_precision"]) else "No feasible validation threshold."
        rows.append(row)
    return pd.DataFrame(rows)


def best_variants_table(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> pd.DataFrame:
    """Select best variants by validation-only operational criteria."""

    combined = pd.concat([taiwan, heloc], ignore_index=True)
    feasible = combined.dropna(subset=["selected_threshold_validation"]).copy()
    feasible["meets_precision_0_40"] = feasible["validation_precision"] >= 0.40
    feasible["meets_specificity_0_60"] = feasible["validation_specificity"] >= 0.60
    feasible["meets_recall_0_55"] = feasible["validation_recall"] >= 0.55
    feasible["selection_score_group"] = (
        feasible["meets_precision_0_40"].astype(int)
        + feasible["meets_specificity_0_60"].astype(int)
        + feasible["meets_recall_0_55"].astype(int)
    )
    feasible = feasible.sort_values(
        [
            "dataset",
            "model_family",
            "selection_score_group",
            "validation_expected_cost_FN5_FP1",
            "validation_pr_auc",
            "validation_brier",
        ],
        ascending=[True, True, False, True, False, True],
    )
    per_family = feasible.groupby(["dataset", "model_family"], as_index=False).head(1).copy()
    per_family["best_scope"] = "best_per_dataset_model_family"
    overall = (
        feasible.sort_values(
            [
                "dataset",
                "selection_score_group",
                "validation_expected_cost_FN5_FP1",
                "validation_pr_auc",
                "validation_brier",
            ],
            ascending=[True, False, True, False, True],
        )
        .groupby("dataset", as_index=False)
        .head(1)
        .copy()
    )
    overall["best_scope"] = "best_overall_dataset"
    return pd.concat([overall, per_family], ignore_index=True)


def fmt(value: float) -> str:
    """Format numeric values for markdown."""

    if pd.isna(value):
        return "NA"
    if abs(float(value)) >= 100:
        return f"{float(value):.0f}"
    return f"{float(value):.4f}"


def write_summary(dataset: str, table: pd.DataFrame, best: pd.DataFrame) -> Path:
    """Write dataset-level imbalance training summary."""

    dataset_best = best.loc[
        best["dataset"].eq(dataset) & best["best_scope"].eq("best_per_dataset_model_family")
    ].sort_values("test_expected_cost_FN5_FP1")
    overall = best.loc[
        best["dataset"].eq(dataset) & best["best_scope"].eq("best_overall_dataset")
    ].iloc[0]
    baseline = table.loc[
        table["variant_name"].eq("catboost_baseline")
        & table["calibration_type"].eq("uncalibrated")
        & table["threshold_policy"].eq("cost_optimal_FN5_FP1")
    ].iloc[0]

    weighted = table.loc[~table["imbalance_strategy"].str.contains("baseline", case=False, regex=False)].copy()
    precision_help = int((weighted["delta_precision_vs_baseline_catboost"] > 0).sum())
    recall_over = int((weighted["delta_recall_vs_baseline_catboost"] > 0.10).sum())
    calibration_improved = int(weighted["probability_calibration_comment"].str.contains("improves", case=False).sum())
    fp_reduced = int((weighted["delta_fp_vs_baseline_catboost"] < 0).sum())
    precision_band = int((weighted["test_precision"].between(0.40, 0.45, inclusive="both")).sum())

    lines = [
        f"# Training-Level Imbalance Revision Summary: {dataset.upper()}",
        "",
        "All variants are trained on the train split. Calibration and threshold policies are selected on validation only. Test metrics are held-out final evaluations.",
        "",
        f"Baseline reference: catboost_baseline + uncalibrated + cost_optimal_FN5_FP1, precision={baseline['test_precision']:.4f}, recall={baseline['test_recall']:.4f}, FP={int(baseline['test_fp'])}, cost={baseline['test_expected_cost_FN5_FP1']:.0f}.",
        "",
        "## Best Variant Per Model Family",
        "",
        "| Model Variant | Calibration | Threshold Policy | Precision | Recall | Specificity | FP | FN | Cost | Comment |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in dataset_best.itertuples(index=False):
        lines.append(
            f"| {row.variant_name} | {row.calibration_type} | {row.threshold_policy} | "
            f"{fmt(row.test_precision)} | {fmt(row.test_recall)} | {fmt(row.test_specificity)} | "
            f"{fmt(row.test_fp)} | {fmt(row.test_fn)} | {fmt(row.test_expected_cost_FN5_FP1)} | "
            f"{row.operational_comment} |"
        )
    lines.extend(
        [
            "",
            "## Audit Answers",
            "",
            f"1. Class weighting precision improvement rows: {precision_help}/{len(weighted)}.",
            f"2. Rows where recall increased by more than 0.10 versus baseline CatBoost: {recall_over}/{len(weighted)}.",
            "3. Weighted models can distort probability quality; inspect Brier/ECE columns by calibration_type.",
            f"4. Calibration-improvement comments observed: {calibration_improved}/{len(weighted)}.",
            "5. CatBoost Balanced vs SqrtBalanced should be judged by validation-selected test rows in the CSV, not by name alone.",
            "6. XGBoost scale_pos_weight best point is listed in the best-variant table when it survives validation criteria.",
            "7. LightGBM is_unbalance probability quality is visible through validation/test Brier and ECE.",
            f"8. Best overall variant by validation-only criteria: {overall['variant_name']} / {overall['calibration_type']} / {overall['threshold_policy']}.",
            f"9. Rows with FP reduction versus baseline CatBoost: {fp_reduced}/{len(weighted)}.",
            f"10. Rows landing in precision 0.40-0.45 band: {precision_band}/{len(weighted)}.",
            "",
        ]
    )
    path = DECISION_REVISION_DIR / f"imbalance_training_summary_{dataset}.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def plot_precision_recall(taiwan: pd.DataFrame) -> Path:
    """Plot Taiwan precision/recall for operationally feasible rows."""

    path = DECISION_REVISION_DIR / "imbalance_variant_precision_recall_taiwan.png"
    rows = taiwan.loc[taiwan["threshold_policy"].isin(["cost_optimal_FN5_FP1", "precision_0_40", "precision_0_45"])].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for family, group in rows.groupby("model_family"):
        ax.scatter(group["test_recall"], group["test_precision"], label=family, alpha=0.65, s=28)
    ax.axhline(0.40, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.45, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Test recall")
    ax.set_ylabel("Test precision")
    ax.set_title("Taiwan Imbalance Variants: Precision vs Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_cost_fp(taiwan: pd.DataFrame) -> Path:
    """Plot Taiwan cost versus FP for selected operational rows."""

    path = DECISION_REVISION_DIR / "imbalance_variant_cost_fp_taiwan.png"
    rows = taiwan.loc[taiwan["threshold_policy"].isin(["cost_optimal_FN5_FP1", "precision_0_40", "precision_0_45"])].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    for family, group in rows.groupby("model_family"):
        ax.scatter(group["test_fp"], group["test_expected_cost_FN5_FP1"], label=family, alpha=0.65, s=28)
    ax.set_xlabel("Test FP")
    ax.set_ylabel("Test expected cost FN=5, FP=1")
    ax.set_title("Taiwan Imbalance Variants: Cost vs False Positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_outputs(taiwan: pd.DataFrame, heloc: pd.DataFrame, best: pd.DataFrame) -> list[Path]:
    """Write all requested outputs."""

    DECISION_REVISION_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        DECISION_REVISION_DIR / "imbalance_training_variants_taiwan.csv",
        DECISION_REVISION_DIR / "imbalance_training_variants_heloc.csv",
        DECISION_REVISION_DIR / "imbalance_training_best_variants.csv",
    ]
    taiwan.to_csv(paths[0], index=False)
    heloc.to_csv(paths[1], index=False)
    best.to_csv(paths[2], index=False)
    paths.extend(
        [
            write_summary("taiwan", taiwan, best),
            write_summary("heloc", heloc, best),
            plot_precision_recall(taiwan),
            plot_cost_fp(taiwan),
        ]
    )
    return paths


def main() -> None:
    """Run the imbalance training strategy revision."""

    taiwan = build_dataset_results("taiwan")
    heloc = build_dataset_results("heloc")
    best = best_variants_table(taiwan, heloc)
    paths = write_outputs(taiwan, heloc, best)
    for path in paths:
        print(path)

    for dataset, table in [("Taiwan", taiwan), ("HELOC", heloc)]:
        print(f"\n{dataset} best variants:")
        best_rows = best.loc[
            best["dataset"].eq(dataset.lower()) & best["best_scope"].eq("best_per_dataset_model_family")
        ].sort_values("test_expected_cost_FN5_FP1")
        print(
            best_rows[
                [
                    "variant_name",
                    "calibration_type",
                    "threshold_policy",
                    "test_precision",
                    "test_recall",
                    "test_specificity",
                    "test_fp",
                    "test_fn",
                    "test_expected_cost_FN5_FP1",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
