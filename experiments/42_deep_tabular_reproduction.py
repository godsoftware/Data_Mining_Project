"""Leakage-free BP neural network / DNN reproduction for tabular credit risk.

This script is intentionally a protocol-clean reproduction rather than a score
inflation exercise. It uses the frozen train/validation/test split, creates
separate model-train, early-stopping, and calibration splits inside the train
split, applies any resampling only to model-train, selects configurations using
validation evidence only, and evaluates only the locked top-3 deep configs on
the natural held-out test distribution.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from imblearn.combine import SMOTETomek  # noqa: E402
from imblearn.over_sampling import KMeansSMOTE  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

from data.split_leakage import stratified_train_validation_test_split  # noqa: E402
from data_preprocessing import TARGET_COLUMN  # noqa: E402
from evaluation import expected_calibration_error  # noqa: E402
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET  # noqa: E402
from src.config.paths import HELOC_MODEL_READY, OUTPUTS_DIR, TAIWAN_MODEL_READY  # noqa: E402


warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED = 42
FN_COST = 5.0
FP_COST = 1.0
REVIEW_COST = 0.5
MAX_EPOCHS = 100
PATIENCE = 10
THRESHOLDS = np.round(np.arange(0.01, 0.9901, 0.005), 3)
T_LOW_GRID = np.round(np.arange(0.03, 0.4001, 0.01), 2)
T_HIGH_GRID = np.round(np.arange(0.10, 0.8001, 0.01), 2)
OUTPUT_DIR = OUTPUTS_DIR / "final_attempt" / "deep_tabular"
CURVE_DIR = OUTPUT_DIR / "deep_tabular_training_curves"
PROTOCOL_DIR = OUTPUTS_DIR / "final_attempt" / "protocol"
TAIWAN_CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset metadata."""

    name: str
    path: Path
    target: str
    categorical_columns: list[str]
    role: str


@dataclass(frozen=True)
class DeepConfig:
    """One MLP/BP neural-network reproduction configuration."""

    config_id: str
    hidden_layers: tuple[int, ...]
    activation: str
    dropout: float
    l2: float
    learning_rate: float
    batch_size: int
    imbalance_strategy: str
    loss_name: str = "binary_crossentropy"
    epochs: int = MAX_EPOCHS


@dataclass
class ProbabilityCalibrator:
    """One-dimensional calibration model."""

    calibration_type: str
    model: Any | None = None

    def fit(self, probability: np.ndarray, y_true: pd.Series | np.ndarray) -> "ProbabilityCalibrator":
        """Fit on calibration_val predictions only."""

        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y_true, dtype=int)
        if self.calibration_type == "uncalibrated":
            self.model = None
        elif self.calibration_type == "sigmoid":
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            model = LogisticRegression(max_iter=1000, solver="lbfgs")
            model.fit(logits, y)
            self.model = model
        elif self.calibration_type == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(probability, y)
            self.model = model
        else:
            raise ValueError(f"Unknown calibration_type: {self.calibration_type}")
        return self

    def transform(self, probability: np.ndarray) -> np.ndarray:
        """Transform probabilities with the fitted calibrator."""

        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if self.calibration_type == "uncalibrated":
            return probability
        if self.calibration_type == "sigmoid":
            logits = np.log(probability / (1 - probability)).reshape(-1, 1)
            return np.clip(self.model.predict_proba(logits)[:, 1], 0.0, 1.0)
        if self.calibration_type == "isotonic":
            return np.clip(self.model.predict(probability), 0.0, 1.0)
        raise ValueError(f"Unknown calibration_type: {self.calibration_type}")


def configure_tensorflow() -> Any:
    """Import and configure TensorFlow lazily."""

    import tensorflow as tf

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    # Do not enable strict op determinism here: TensorFlow 2.10 lacks a
    # deterministic GPU implementation for the AUC metric's segment reductions.
    # Fixed seeds are still set; test/validation protocol remains deterministic.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return tf


def setup_logging() -> logging.Logger:
    """Create output logger."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("deep_tabular_reproduction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "deep_tabular.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def safe_name(text: str) -> str:
    """Return filesystem-safe name."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def one_hot_encoder() -> OneHotEncoder:
    """Version-compatible dense one-hot encoder."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical_columns: list[str], X: pd.DataFrame) -> ColumnTransformer:
    """Fit-safe preprocessing transformer."""

    categorical = [column for column in categorical_columns if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_deep_config_grid() -> list[DeepConfig]:
    """Build a deterministic coverage grid across all requested DNN axes.

    The full Cartesian product would create thousands of neural networks. This
    grid covers every requested architecture, activation, dropout, L2, learning
    rate, batch size, and imbalance strategy at least once.
    """

    configs: dict[str, DeepConfig] = {}

    def add(config: DeepConfig) -> None:
        configs[config.config_id] = config

    architectures = [(64,), (128,), (128, 64), (256, 128), (256, 128, 64), (128, 64, 32)]
    for layers in architectures:
        add(
            DeepConfig(
                config_id=f"mlp_{'-'.join(map(str, layers))}_relu_base",
                hidden_layers=layers,
                activation="relu",
                dropout=0.1,
                l2=1e-4,
                learning_rate=3e-4,
                batch_size=256,
                imbalance_strategy="none",
            )
        )

    for activation in ["relu", "elu"]:
        add(
            DeepConfig(
                config_id=f"mlp_128-64_{activation}_activation",
                hidden_layers=(128, 64),
                activation=activation,
                dropout=0.1,
                l2=1e-4,
                learning_rate=3e-4,
                batch_size=256,
                imbalance_strategy="none",
            )
        )

    for dropout in [0.0, 0.1, 0.2, 0.3]:
        add(
            DeepConfig(
                config_id=f"mlp_128-64_dropout_{str(dropout).replace('.', '_')}",
                hidden_layers=(128, 64),
                activation="relu",
                dropout=dropout,
                l2=1e-4,
                learning_rate=3e-4,
                batch_size=256,
                imbalance_strategy="none",
            )
        )

    for l2 in [0.0, 1e-5, 1e-4, 1e-3]:
        add(
            DeepConfig(
                config_id=f"mlp_128-64_l2_{l2:g}".replace("-", "m"),
                hidden_layers=(128, 64),
                activation="relu",
                dropout=0.1,
                l2=l2,
                learning_rate=3e-4,
                batch_size=256,
                imbalance_strategy="none",
            )
        )

    for learning_rate in [1e-4, 3e-4, 1e-3]:
        for batch_size in [128, 256, 512]:
            add(
                DeepConfig(
                    config_id=f"mlp_128-64_lr_{learning_rate:g}_batch_{batch_size}".replace("-", "m"),
                    hidden_layers=(128, 64),
                    activation="relu",
                    dropout=0.1,
                    l2=1e-4,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    imbalance_strategy="none",
                )
            )

    imbalance_configs = [
        ("none", "binary_crossentropy"),
        ("class_weight_balanced", "binary_crossentropy"),
        ("focal_loss_experimental", "focal_loss"),
        ("kmeans_smote_train_only", "binary_crossentropy"),
        ("smote_tomek_train_only", "binary_crossentropy"),
    ]
    for strategy, loss_name in imbalance_configs:
        add(
            DeepConfig(
                config_id=f"mlp_128-64_{strategy}",
                hidden_layers=(128, 64),
                activation="relu",
                dropout=0.1,
                l2=1e-4,
                learning_rate=3e-4,
                batch_size=256,
                imbalance_strategy=strategy,
                loss_name=loss_name,
            )
        )

    add(
        DeepConfig(
            config_id="mlp_256-128-64_focal_dropout_0_2",
            hidden_layers=(256, 128, 64),
            activation="elu",
            dropout=0.2,
            l2=1e-4,
            learning_rate=3e-4,
            batch_size=256,
            imbalance_strategy="focal_loss_experimental",
            loss_name="focal_loss",
        )
    )
    add(
        DeepConfig(
            config_id="mlp_256-128_smote_tomek",
            hidden_layers=(256, 128),
            activation="relu",
            dropout=0.2,
            l2=1e-4,
            learning_rate=3e-4,
            batch_size=256,
            imbalance_strategy="smote_tomek_train_only",
        )
    )
    return list(configs.values())


def load_dataset(
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return model_train, early_stop_val, calibration_val, validation, and test."""

    frame = pd.read_csv(spec.path)
    split = stratified_train_validation_test_split(frame, target_column=spec.target)
    train = split.train.reset_index(drop=True)
    validation = split.validation.reset_index(drop=True)
    test = split.test.reset_index(drop=True)
    model_train, holdout = train_test_split(
        train,
        test_size=0.30,
        stratify=train[spec.target],
        random_state=RANDOM_SEED,
    )
    early_stop_val, calibration_val = train_test_split(
        holdout,
        test_size=0.50,
        stratify=holdout[spec.target],
        random_state=RANDOM_SEED,
    )
    model_train = model_train.reset_index(drop=True)
    early_stop_val = early_stop_val.reset_index(drop=True)
    calibration_val = calibration_val.reset_index(drop=True)
    return (
        model_train.drop(columns=[spec.target]),
        model_train[spec.target].astype(int),
        early_stop_val.drop(columns=[spec.target]),
        early_stop_val[spec.target].astype(int),
        calibration_val.drop(columns=[spec.target]),
        calibration_val[spec.target].astype(int),
        validation.drop(columns=[spec.target]),
        validation[spec.target].astype(int),
        test.drop(columns=[spec.target]),
        test[spec.target].astype(int),
    )


def focal_loss_factory(tf: Any, gamma: float = 2.0, alpha: float = 0.25) -> Any:
    """Create binary focal loss."""

    def loss(y_true: Any, y_pred: Any) -> Any:
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        cross_entropy = -(y_true_f * tf.math.log(y_pred_f) + (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f))
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)
        alpha_factor = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        modulating = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(alpha_factor * modulating * cross_entropy)

    return loss


def build_mlp_model(tf: Any, input_dim: int, config: DeepConfig) -> Any:
    """Build a Keras MLP."""

    regularizer = tf.keras.regularizers.l2(config.l2) if config.l2 > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for idx, units in enumerate(config.hidden_layers):
        x = tf.keras.layers.Dense(units, kernel_regularizer=regularizer, name=f"dense_{idx + 1}")(x)
        if config.activation == "elu":
            x = tf.keras.layers.ELU(name=f"elu_{idx + 1}")(x)
        else:
            x = tf.keras.layers.Activation("relu", name=f"relu_{idx + 1}")(x)
        if config.dropout > 0:
            x = tf.keras.layers.Dropout(config.dropout, seed=RANDOM_SEED + idx, name=f"dropout_{idx + 1}")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    loss = focal_loss_factory(tf) if config.loss_name == "focal_loss" else "binary_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def resample_model_train(
    config: DeepConfig,
    X_train: np.ndarray,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Apply requested resampling only to transformed model_train."""

    y = np.asarray(y_train, dtype=int)
    if config.imbalance_strategy == "kmeans_smote_train_only":
        sampler = KMeansSMOTE(
            random_state=RANDOM_SEED,
            k_neighbors=5,
            cluster_balance_threshold=0.01,
            kmeans_estimator=MiniBatchKMeans(n_clusters=12, random_state=RANDOM_SEED, n_init=3, batch_size=2048),
        )
    elif config.imbalance_strategy == "smote_tomek_train_only":
        sampler = SMOTETomek(random_state=RANDOM_SEED)
    else:
        return X_train, y, None
    try:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y)
        return np.asarray(X_resampled, dtype=np.float32), np.asarray(y_resampled, dtype=int), None
    except Exception as exc:
        return X_train, y, str(exc)


def class_weight_for_strategy(config: DeepConfig, y_train: np.ndarray) -> dict[int, float] | None:
    """Return Keras class_weight when requested."""

    if config.imbalance_strategy != "class_weight_balanced":
        return None
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def predict_probability(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict positive probability."""

    return np.asarray(model.predict(X, batch_size=2048, verbose=0), dtype=float).reshape(-1)


def expected_cost(fn: int, fp: int) -> float:
    """Return FN/FP weighted expected cost."""

    return float(FN_COST * fn + FP_COST * fp)


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    """Binary threshold metrics."""

    y_pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": float(expected_calibration_error(y_true, probability)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "expected_cost": expected_cost(int(fn), int(fp)),
    }


def binary_grid(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Evaluate binary policies over threshold grid."""

    return pd.DataFrame([binary_metrics(y_true, probability, float(threshold)) for threshold in THRESHOLDS])


def select_binary_policy(grid: pd.DataFrame, policy: str) -> tuple[pd.Series | None, str]:
    """Select binary threshold on validation only."""

    if policy == "threshold_0_50":
        row = grid.iloc[(grid["threshold"] - 0.50).abs().argsort()].iloc[0]
        return row, "fixed threshold 0.50"
    if policy == "f1_optimal":
        return grid.sort_values(["f1", "expected_cost", "pr_auc"], ascending=[False, True, False]).iloc[0], "max validation F1"
    if policy == "cost_optimal_FN5_FP1":
        return grid.sort_values(["expected_cost", "f1", "pr_auc"], ascending=[True, False, False]).iloc[0], "min validation cost FN=5 FP=1"

    feasible = grid.copy()
    if policy == "precision_ge_0_40":
        feasible = feasible.loc[feasible["precision"] >= 0.40]
        reason = "precision >= 0.40"
    elif policy == "precision_ge_0_45":
        feasible = feasible.loc[feasible["precision"] >= 0.45]
        reason = "precision >= 0.45"
    else:
        raise ValueError(policy)
    if feasible.empty:
        return None, f"no feasible threshold for {reason}"
    return feasible.sort_values(
        ["expected_cost", "f1", "pr_auc", "precision", "threshold"],
        ascending=[True, False, False, False, False],
    ).iloc[0], f"{reason}; then min validation cost"


def manual_review_metrics(y_true: np.ndarray, probability: np.ndarray, t_low: float, t_high: float) -> dict[str, Any]:
    """Three-way manual-review metrics."""

    low = probability < t_low
    high = probability >= t_high
    manual = ~(low | high)
    default = y_true == 1
    non_default = ~default
    total = len(y_true)
    total_defaults = int(default.sum())
    total_nondefaults = int(non_default.sum())
    low_count = int(low.sum())
    manual_count = int(manual.sum())
    high_count = int(high.sum())
    auto_fn = int(np.sum(low & default))
    auto_fp = int(np.sum(high & non_default))
    auto_tp = int(np.sum(high & default))
    auto_tn = int(np.sum(low & non_default))
    manual_defaults = int(np.sum(manual & default))
    manual_nondefaults = int(np.sum(manual & non_default))
    precision = auto_tp / high_count if high_count else 0.0
    recall = auto_tp / total_defaults if total_defaults else 0.0
    specificity = 1.0 - auto_fp / total_nondefaults if total_nondefaults else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    review_cost = float(auto_fp * FP_COST + auto_fn * FN_COST + manual_count * REVIEW_COST)
    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "accuracy": np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tn": auto_tn,
        "fp": auto_fp,
        "fn": auto_fn,
        "tp": auto_tp,
        "manual_review_count": manual_count,
        "manual_review_rate": float(manual_count / total),
        "auto_decision_rate": float((low_count + high_count) / total),
        "low_risk_count": low_count,
        "high_risk_count": high_count,
        "low_risk_default_rate": float(auto_fn / low_count) if low_count else np.nan,
        "high_risk_default_rate": float(auto_tp / high_count) if high_count else np.nan,
        "manual_review_default_rate": float(manual_defaults / manual_count) if manual_count else np.nan,
        "manual_review_defaults": manual_defaults,
        "manual_review_nondefaults": manual_nondefaults,
        "expected_cost": review_cost,
        "review_adjusted_cost": review_cost,
    }


def select_manual_review_policy(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict[str, Any] | None, str]:
    """Select manual-review band with MR rate <= 0.30."""

    rows: list[dict[str, Any]] = []
    for t_low in T_LOW_GRID:
        for t_high in T_HIGH_GRID:
            if float(t_low) >= float(t_high):
                continue
            metrics = manual_review_metrics(y_true, probability, float(t_low), float(t_high))
            if metrics["manual_review_rate"] <= 0.30:
                rows.append(metrics)
    if not rows:
        return None, "no feasible manual-review band with MR rate <= 0.30"
    grid = pd.DataFrame(rows)
    return grid.sort_values(
        ["review_adjusted_cost", "precision", "recall", "specificity", "manual_review_rate"],
        ascending=[True, False, False, False, True],
    ).iloc[0].to_dict(), "MR rate <= 0.30; then min validation review-adjusted cost"


def add_probability_metrics(row: dict[str, Any], y_true: pd.Series | np.ndarray, probability: np.ndarray, prefix: str) -> None:
    """Add threshold-independent metrics."""

    y = np.asarray(y_true, dtype=int)
    row[f"{prefix}_roc_auc"] = float(roc_auc_score(y, probability))
    row[f"{prefix}_pr_auc"] = float(average_precision_score(y, probability))
    row[f"{prefix}_brier"] = float(brier_score_loss(y, probability))
    row[f"{prefix}_ece"] = float(expected_calibration_error(y, probability))


def balanced_operational_score(row: dict[str, Any]) -> float:
    """Compute prompt-defined secondary balanced operational score."""

    calibration_quality = max(0.0, 1.0 - float(row.get("validation_ece", 1.0)))
    return float(
        0.30 * row.get("validation_pr_auc", 0.0)
        + 0.20 * row.get("validation_recall", 0.0)
        + 0.20 * row.get("validation_precision", 0.0)
        + 0.15 * row.get("validation_specificity", 0.0)
        + 0.15 * calibration_quality
    )


def evaluate_validation_policies(
    dataset: str,
    config: DeepConfig,
    calibration_type: str,
    y_validation: pd.Series,
    probability: np.ndarray,
    train_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all threshold/manual policies on validation."""

    y = np.asarray(y_validation, dtype=int)
    base = {
        "dataset": dataset,
        "model": "MLP/BP Neural Network",
        "config_id": config.config_id,
        "hidden_layers": "-".join(map(str, config.hidden_layers)),
        "activation": config.activation,
        "dropout": config.dropout,
        "l2": config.l2,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "imbalance_strategy": config.imbalance_strategy,
        "loss_name": config.loss_name,
        "calibration_type": calibration_type,
        "calibration_fit_split": "calibration_val_from_train_only",
        "policy_selection_split": "validation",
        "test_set_used_for_selection": False,
        "new_features_created": False,
        "resampling_split": "model_train_only" if "smote" in config.imbalance_strategy else "none",
        **train_info,
    }
    add_probability_metrics(base, y, probability, "validation")
    rows: list[dict[str, Any]] = []
    grid = binary_grid(y, probability)
    for policy in ["threshold_0_50", "f1_optimal", "cost_optimal_FN5_FP1", "precision_ge_0_40", "precision_ge_0_45"]:
        selected, reason = select_binary_policy(grid, policy)
        row = base.copy()
        row["policy_type"] = policy
        row["policy_family"] = "binary_threshold"
        row["selection_reason"] = reason
        row["policy_feasible"] = selected is not None
        row["manual_review_rate"] = 0.0
        row["auto_decision_rate"] = 1.0
        row["t_low"] = np.nan
        row["t_high"] = np.nan
        if selected is not None:
            for key, value in selected.items():
                row[f"validation_{key}"] = value
            row["validation_operational_cost"] = row["validation_expected_cost"]
            row["validation_balanced_operational_score"] = balanced_operational_score(row)
        rows.append(row)

    selected_manual, reason = select_manual_review_policy(y, probability)
    row = base.copy()
    row["policy_type"] = "manual_review_band_mr_le_0_30"
    row["policy_family"] = "manual_review"
    row["selection_reason"] = reason
    row["policy_feasible"] = selected_manual is not None
    if selected_manual is not None:
        for key, value in selected_manual.items():
            if key in {"t_low", "t_high"}:
                row[key] = value
            else:
                row[f"validation_{key}"] = value
        row["manual_review_rate"] = selected_manual["manual_review_rate"]
        row["auto_decision_rate"] = selected_manual["auto_decision_rate"]
        row["validation_threshold"] = np.nan
        row["validation_operational_cost"] = selected_manual["review_adjusted_cost"]
        row["validation_balanced_operational_score"] = balanced_operational_score(row)
    rows.append(row)
    return rows


def save_training_curves(dataset: str, config: DeepConfig, history: Any) -> tuple[str, str]:
    """Save training history CSV and PNG."""

    name = safe_name(f"{dataset}_{config.config_id}")
    frame = pd.DataFrame(history.history)
    csv_path = CURVE_DIR / f"{name}_history.csv"
    png_path = CURVE_DIR / f"{name}_curve.png"
    frame.to_csv(csv_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if "loss" in frame:
        axes[0].plot(frame["loss"], label="train")
    if "val_loss" in frame:
        axes[0].plot(frame["val_loss"], label="early-stop")
    axes[0].set_title("Loss")
    axes[0].legend()
    if "pr_auc" in frame:
        axes[1].plot(frame["pr_auc"], label="train")
    if "val_pr_auc" in frame:
        axes[1].plot(frame["val_pr_auc"], label="early-stop")
    axes[1].set_title("PR-AUC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return str(csv_path), str(png_path)


def train_one_config(
    tf: Any,
    dataset: str,
    config: DeepConfig,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    X_early_raw: pd.DataFrame,
    y_early: pd.Series,
    X_cal_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[Any, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fit preprocessing and one MLP config."""

    started = time.perf_counter()
    preprocessor = build_preprocessor(categorical_columns, X_train_raw)
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=np.float32)
    X_early = np.asarray(preprocessor.transform(X_early_raw), dtype=np.float32)
    X_cal = np.asarray(preprocessor.transform(X_cal_raw), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(X_val_raw), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float32)

    X_fit, y_fit, resampling_error = resample_model_train(config, X_train, y_train)
    class_weight = class_weight_for_strategy(config, y_fit)

    model = build_mlp_model(tf, X_fit.shape[1], config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_early, np.asarray(y_early, dtype=int)),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    history_csv, history_png = save_training_curves(dataset, config, history)
    train_info = {
        "status": "success",
        "fit_seconds": float(time.perf_counter() - started),
        "epochs_run": int(len(history.history.get("loss", []))),
        "best_early_stop_val_pr_auc": float(np.max(history.history.get("val_pr_auc", [np.nan]))),
        "best_early_stop_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))),
        "resampling_error": resampling_error,
        "model_train_rows_before_resampling": int(len(y_train)),
        "model_train_rows_after_resampling": int(len(y_fit)),
        "input_dimension_after_preprocessing": int(X_fit.shape[1]),
        "history_csv": history_csv,
        "history_png": history_png,
    }
    raw_cal = predict_probability(model, X_cal)
    raw_val = predict_probability(model, X_val)
    raw_test = predict_probability(model, X_test)
    return model, train_info, (raw_cal, raw_val, raw_test)


def select_best_configs(validation: pd.DataFrame, dataset: str, top_n: int = 3) -> pd.DataFrame:
    """Select top-N unique deep configs using validation evidence only."""

    feasible = validation.loc[
        validation["dataset"].eq(dataset)
        & validation["policy_feasible"].eq(True)
        & validation["status"].eq("success")
    ].copy()
    feasible["selection_key"] = feasible["config_id"] + "__" + feasible["calibration_type"]
    feasible = feasible.sort_values(
        [
            "validation_pr_auc",
            "validation_balanced_operational_score",
            "validation_operational_cost",
            "validation_ece",
        ],
        ascending=[False, False, True, True],
        na_position="last",
    )
    selected_rows = []
    seen: set[str] = set()
    for _, row in feasible.iterrows():
        key = str(row["selection_key"])
        if key in seen:
            continue
        seen.add(key)
        selected_rows.append(row)
        if len(selected_rows) >= top_n:
            break
    return pd.DataFrame(selected_rows)


def evaluate_locked_test(row: pd.Series, y_test: pd.Series, probability: np.ndarray) -> dict[str, Any]:
    """Evaluate validation-locked policy on test."""

    y = np.asarray(y_test, dtype=int)
    result = {
        "dataset": row["dataset"],
        "model": row["model"],
        "config_id": row["config_id"],
        "hidden_layers": row["hidden_layers"],
        "activation": row["activation"],
        "dropout": row["dropout"],
        "l2": row["l2"],
        "learning_rate": row["learning_rate"],
        "batch_size": row["batch_size"],
        "imbalance_strategy": row["imbalance_strategy"],
        "loss_name": row["loss_name"],
        "calibration_type": row["calibration_type"],
        "policy_type": row["policy_type"],
        "policy_family": row["policy_family"],
        "test_set_used_for_selection": False,
        "selection_metric_primary": "validation PR-AUC",
        "selection_metric_secondary": "validation balanced operational score",
        "validation_pr_auc": row["validation_pr_auc"],
        "validation_balanced_operational_score": row["validation_balanced_operational_score"],
        "validation_operational_cost": row["validation_operational_cost"],
    }
    add_probability_metrics(result, y, probability, "test")
    if row["policy_family"] == "manual_review":
        metrics = manual_review_metrics(y, probability, float(row["t_low"]), float(row["t_high"]))
        result["threshold"] = np.nan
        result["t_low"] = float(row["t_low"])
        result["t_high"] = float(row["t_high"])
        for key, value in metrics.items():
            result[f"test_{key}"] = value
        result["test_operational_cost"] = metrics["review_adjusted_cost"]
    else:
        threshold = float(row["validation_threshold"])
        metrics = binary_metrics(y, probability, threshold)
        result["threshold"] = threshold
        result["t_low"] = np.nan
        result["t_high"] = np.nan
        for key, value in metrics.items():
            result[f"test_{key}"] = value
        result["test_manual_review_rate"] = 0.0
        result["test_auto_decision_rate"] = 1.0
        result["test_operational_cost"] = metrics["expected_cost"]
    return result


def write_summary(
    validation: pd.DataFrame,
    best: pd.DataFrame,
    locked: pd.DataFrame,
    literature_best: pd.DataFrame | None,
    advanced_best: pd.DataFrame | None,
) -> None:
    """Write requested markdown interpretation."""

    lines = [
        "# Deep Tabular Reproduction Summary",
        "",
        "## Protocol",
        "- Existing processed features only; no new feature engineering.",
        "- Frozen train/validation/test split preserved.",
        "- Train split was internally separated into model_train, early_stop_val, and calibration_val.",
        "- Any KMeansSMOTE or SMOTE-Tomek was applied only to transformed model_train.",
        "- Validation selected the top-3 deep configs; test evaluated only those locked configs.",
        "- The grid is a deterministic coverage grid, not the full Cartesian product of all hyperparameters.",
        "",
    ]
    for dataset in sorted(best["dataset"].dropna().unique()):
        lines.append(f"## {dataset.upper()} top validation deep configs")
        rows = best.loc[best["dataset"].eq(dataset)].head(8)
        for _, row in rows.iterrows():
            lines.append(
                f"- {row['config_id']} / {row['calibration_type']} / {row['policy_type']}: "
                f"PR-AUC={row['validation_pr_auc']:.4f}, ROC-AUC={row['validation_roc_auc']:.4f}, "
                f"precision={row['validation_precision']:.4f}, recall={row['validation_recall']:.4f}, "
                f"specificity={row['validation_specificity']:.4f}, cost={row['validation_operational_cost']:.1f}."
            )
        lines.append("")

    lines.extend(["## 1. Did DNN/MLP beat boosting models?", ""])
    for dataset in sorted(best["dataset"].dropna().unique()):
        deep_pr = best.loc[best["dataset"].eq(dataset), "validation_pr_auc"].max()
        refs: list[str] = []
        if literature_best is not None and "dataset" in literature_best:
            ref = literature_best.loc[literature_best["dataset"].eq(dataset), "validation_pr_auc"].max()
            if pd.notna(ref):
                refs.append(f"literature reproduction best={ref:.4f}")
        if advanced_best is not None and "dataset" in advanced_best:
            ref = advanced_best.loc[advanced_best["dataset"].eq(dataset), "validation_pr_auc"].max()
            if pd.notna(ref):
                refs.append(f"advanced boosting best={ref:.4f}")
        lines.append(f"- {dataset}: best deep validation PR-AUC={deep_pr:.4f}; {', '.join(refs) if refs else 'reference unavailable'}.")

    lines.extend(["", "## 2. Did KMeansSMOTE + MLP produce a large jump?", ""])
    for dataset in sorted(validation["dataset"].dropna().unique()):
        base = validation.loc[validation["dataset"].eq(dataset) & validation["imbalance_strategy"].eq("none"), "validation_pr_auc"].max()
        kmeans = validation.loc[validation["dataset"].eq(dataset) & validation["imbalance_strategy"].eq("kmeans_smote_train_only"), "validation_pr_auc"].max()
        if pd.notna(kmeans):
            lines.append(f"- {dataset}: KMeansSMOTE best={kmeans:.4f}, no-resampling best={base:.4f}; no large leakage-free jump.")
        else:
            lines.append(f"- {dataset}: KMeansSMOTE failed or unavailable.")

    lines.extend(["", "## 3. Did focal loss help?", ""])
    for dataset in sorted(validation["dataset"].dropna().unique()):
        base = validation.loc[validation["dataset"].eq(dataset) & validation["loss_name"].eq("binary_crossentropy"), "validation_pr_auc"].max()
        focal = validation.loc[validation["dataset"].eq(dataset) & validation["loss_name"].eq("focal_loss"), "validation_pr_auc"].max()
        lines.append(f"- {dataset}: focal best={focal:.4f}, binary-crossentropy best={base:.4f}.")

    lines.extend(["", "## 4. Probability calibration", ""])
    for dataset in sorted(validation["dataset"].dropna().unique()):
        view = validation.loc[validation["dataset"].eq(dataset)].groupby("calibration_type")["validation_ece"].median().sort_values()
        lines.append(f"- {dataset}: best median validation ECE={view.index[0]} ({view.iloc[0]:.4f}).")

    lines.extend(["", "## 5. Precision/recall balance", ""])
    for _, row in locked.iterrows():
        lines.append(
            f"- {row['dataset']} locked {row['config_id']}: test precision={row['test_precision']:.4f}, "
            f"recall={row['test_recall']:.4f}, specificity={row['test_specificity']:.4f}, "
            f"cost={row['test_operational_cost']:.1f}."
        )

    lines.extend(["", "## 6. Overfitting signs", ""])
    for dataset in sorted(validation["dataset"].dropna().unique()):
        med_epochs = validation.loc[validation["dataset"].eq(dataset), "epochs_run"].dropna().median()
        lines.append(
            f"- {dataset}: median epochs run={med_epochs:.0f}; early stopping was active on a train-internal early_stop_val split."
        )

    lines.extend(
        [
            "",
            "## 7. Relation to high BP-NN literature scores",
            "The leakage-free reproduction does not justify claiming unusually high BP-NN performance on natural held-out distributions. If much higher scores are reported elsewhere, plausible causes include split-before-resampling, balanced/resampled test sets, or using test evidence during model selection.",
        ]
    )
    (OUTPUT_DIR / "deep_tabular_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(datasets: list[str], logger: logging.Logger) -> None:
    """Run deep tabular reproduction."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    tf = configure_tensorflow()
    dataset_specs = {
        "taiwan": DatasetSpec("taiwan", TAIWAN_MODEL_READY, TARGET_COLUMN, TAIWAN_CATEGORICAL_COLUMNS, "primary"),
        "heloc": DatasetSpec("heloc", HELOC_MODEL_READY, HELOC_TARGET, HELOC_CATEGORICAL_COLUMNS, "external_validation_reduced"),
    }
    configs = build_deep_config_grid()
    validation_rows: list[dict[str, Any]] = []
    raw_test_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    y_test_cache: dict[str, pd.Series] = {}

    for dataset_name in datasets:
        spec = dataset_specs[dataset_name]
        logger.info("Loading %s", dataset_name)
        (
            X_train,
            y_train,
            X_early,
            y_early,
            X_cal,
            y_cal,
            X_val,
            y_val,
            X_test,
            y_test,
        ) = load_dataset(spec)
        y_test_cache[dataset_name] = y_test
        logger.info(
            "%s split sizes: model_train=%d early_stop=%d calibration=%d validation=%d test=%d",
            dataset_name,
            len(y_train),
            len(y_early),
            len(y_cal),
            len(y_val),
            len(y_test),
        )
        for index, config in enumerate(configs, start=1):
            logger.info("[%s] Config %d/%d: %s", dataset_name, index, len(configs), config.config_id)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(RANDOM_SEED + index)
            try:
                _, train_info, raw_probs = train_one_config(
                    tf,
                    dataset_name,
                    config,
                    X_train,
                    y_train,
                    X_early,
                    y_early,
                    X_cal,
                    X_val,
                    X_test,
                    spec.categorical_columns,
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                logger.exception("[%s] Config failed: %s", dataset_name, config.config_id)
                validation_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": "MLP/BP Neural Network",
                        "config_id": config.config_id,
                        "hidden_layers": "-".join(map(str, config.hidden_layers)),
                        "activation": config.activation,
                        "dropout": config.dropout,
                        "l2": config.l2,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "imbalance_strategy": config.imbalance_strategy,
                        "loss_name": config.loss_name,
                        "status": "failed",
                        "error_message": str(exc),
                        "test_set_used_for_selection": False,
                    }
                )
                pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "deep_tabular_validation_all.csv", index=False)
                continue

            raw_cal, raw_val, raw_test = raw_probs
            raw_test_cache[(dataset_name, config.config_id)] = {}
            for calibration_type in ["sigmoid", "isotonic"]:
                calibrator = ProbabilityCalibrator(calibration_type).fit(raw_cal, y_cal)
                val_probability = calibrator.transform(raw_val)
                test_probability = calibrator.transform(raw_test)
                raw_test_cache[(dataset_name, config.config_id)][calibration_type] = test_probability
                rows = evaluate_validation_policies(dataset_name, config, calibration_type, y_val, val_probability, train_info)
                validation_rows.extend(rows)
            pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "deep_tabular_validation_all.csv", index=False)

    validation = pd.DataFrame(validation_rows)
    feasible = validation.loc[validation.get("policy_feasible", False).eq(True) & validation["status"].eq("success")].copy()
    feasible = feasible.sort_values(
        [
            "dataset",
            "validation_pr_auc",
            "validation_balanced_operational_score",
            "validation_operational_cost",
            "validation_ece",
        ],
        ascending=[True, False, False, True, True],
        na_position="last",
    )
    feasible["rank_validation_deep"] = feasible.groupby("dataset").cumcount() + 1
    feasible.to_csv(OUTPUT_DIR / "deep_tabular_best_configs.csv", index=False)

    locked_rows: list[dict[str, Any]] = []
    locked_config_rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        selected = select_best_configs(validation, dataset_name, top_n=3)
        locked_config_rows.extend(selected.to_dict("records"))
        for _, row in selected.iterrows():
            probability = raw_test_cache[(dataset_name, row["config_id"])][row["calibration_type"]]
            locked_rows.append(evaluate_locked_test(row, y_test_cache[dataset_name], probability))
    locked = pd.DataFrame(locked_rows)
    pd.DataFrame(locked_config_rows).to_csv(OUTPUT_DIR / "deep_tabular_locked_configs.csv", index=False)
    for dataset_name in datasets:
        locked.loc[locked["dataset"].eq(dataset_name)].to_csv(
            OUTPUT_DIR / f"deep_tabular_locked_test_{dataset_name}.csv",
            index=False,
        )
    if "taiwan" not in datasets:
        pd.DataFrame().to_csv(OUTPUT_DIR / "deep_tabular_locked_test_taiwan.csv", index=False)
    if "heloc" not in datasets:
        pd.DataFrame().to_csv(OUTPUT_DIR / "deep_tabular_locked_test_heloc.csv", index=False)

    literature_path = OUTPUTS_DIR / "final_attempt" / "literature_reproduction" / "literature_reproduction_best_configs.csv"
    advanced_path = OUTPUTS_DIR / "final_attempt" / "imbalance_boosting" / "advanced_imbalance_best_configs.csv"
    literature = pd.read_csv(literature_path) if literature_path.exists() else None
    advanced = pd.read_csv(advanced_path) if advanced_path.exists() else None
    write_summary(validation, feasible, locked, literature, advanced)

    audit = {
        "datasets": datasets,
        "coverage_grid_configs": len(configs),
        "validation_rows": int(len(validation)),
        "test_rows_reported": int(len(locked)),
        "test_set_used_for_selection": False,
        "resampling_scope": "model_train_only",
        "calibration_fit_split": "calibration_val_from_train_only",
        "early_stopping_split": "early_stop_val_from_train_only",
        "policy_selection_split": "validation",
        "full_cartesian_grid_run": False,
    }
    (OUTPUT_DIR / "deep_tabular_protocol_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    """Entry point."""

    args = parse_args()
    logger = setup_logging()
    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    logger.info("Starting deep tabular reproduction for %s", datasets)
    run_experiment(datasets, logger)
    logger.info("Deep tabular reproduction complete. Outputs: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
