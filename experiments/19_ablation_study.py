"""Run Prompt 21 SCRE-Credit ablation study on the Taiwan primary dataset."""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import FN_COST, FP_COST, RANDOM_SEED, load_experiment_config
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from run_calibration_analysis import (
    _fit_probability_model,
    _positive_class_probability,
    build_calibration_model_specs,
)
from src.config.paths import FIGURES_DIR, MODELS_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.explainability.faithfulness import (
    baseline_feature_values,
    faithfulness_metrics,
    topk_deletion_table,
)
from src.explainability.shap_stability import (
    StabilityDatasetConfig,
    feature_frequency_table,
    pairwise_rank_correlations,
    run_seed_stability,
)
from src.features.feature_registry import TAIWAN_CATEGORICAL_COLUMNS, TAIWAN_ENGINEERED_FEATURES
from src.models.scre_credit import SCRECreditHybridClassifier, SCRE_COMPONENT_WEIGHTS


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


DATASET = "taiwan"
PRIMARY_COST_SCENARIO = "B_FN5_FP1"
ABLATION_MODEL_DIR = MODELS_DIR / "ablation_study"


@dataclass
class ProbabilityModelWrapper:
    """Expose raw-feature predict_proba for one SCRE ablation."""

    base_models: dict[str, object]
    classifier: SCRECreditHybridClassifier | None = None
    single_model_name: str | None = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities for the configured ablation."""

        if self.single_model_name is not None:
            return self.base_models[self.single_model_name].predict_proba(X)
        if self.classifier is None:
            raise RuntimeError("SCRE classifier is required when single_model_name is not set.")
        probability_frame = pd.DataFrame(
            {
                model_name: _positive_class_probability(model, X)
                for model_name, model in self.base_models.items()
                if model_name in self.classifier.model_names_
            },
            index=X.index,
        )
        return self.classifier.predict_proba_from_base(probability_frame)


def parse_args() -> argparse.Namespace:
    """Parse ablation runtime controls."""

    config = load_experiment_config()
    default_stability_seeds = int(config.get("shap_stability_seeds", 50))
    parser = argparse.ArgumentParser(description="Run SCRE-Credit ablation study.")
    parser.add_argument("--stability-seeds", type=int, default=default_stability_seeds)
    parser.add_argument(
        "--stability-explain-rows",
        type=int,
        default=None,
        help="Rows used for SHAP stability. Default uses the full validation split.",
    )
    parser.add_argument("--faithfulness-max-k", type=int, default=10)
    return parser.parse_args()


def seed_list(n_seeds: int, base_seed: int = RANDOM_SEED) -> list[int]:
    """Return deterministic seeds for SHAP stability."""

    rng = np.random.default_rng(base_seed)
    return [int(seed) for seed in rng.choice(np.arange(1, 1_000_000), size=n_seeds, replace=False)]


def component_weights_without(excluded_components: list[str]) -> dict[str, float]:
    """Return SCRE component weights with selected components zeroed and renormalized."""

    weights = SCRE_COMPONENT_WEIGHTS.copy()
    for component in excluded_components:
        if component not in weights:
            raise KeyError(f"Unknown SCRE component: {component}")
        weights[component] = 0.0
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one SCRE component weight must remain positive.")
    return {key: value / total for key, value in weights.items()}


def performance_only_weights() -> dict[str, float]:
    """Return weights using only PR-AUC, ROC-AUC, and recall."""

    return component_weights_without(["calibration_error", "expected_cost", "stability_score", "faithfulness_score"])


def load_taiwan_metric_table() -> pd.DataFrame:
    """Load selected Taiwan SCRE input metrics from Prompt 13 outputs."""

    path = TABLES_DIR / "scre_model_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing SCRE weight table: {path}")
    table = pd.read_csv(path)
    table = table.loc[table["dataset"] == DATASET].copy()
    if table.empty:
        raise ValueError("No Taiwan rows found in scre_model_weights.csv")
    required = [
        "model",
        "model_path",
        "selected_calibration_method",
        "pr_auc",
        "roc_auc",
        "recall",
        "calibration_error",
        "expected_cost",
        "stability_score",
        "faithfulness_score",
        "base_selected_threshold",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required SCRE metric columns: {missing}")
    table["stability_score"] = 1.0
    table["faithfulness_score"] = 1.0
    return table.reset_index(drop=True)


def load_base_models(metric_table: pd.DataFrame) -> dict[str, object]:
    """Load calibrated base-model artifacts referenced by the metric table."""

    models = {}
    for row in metric_table.itertuples(index=False):
        path = Path(row.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing calibrated base model artifact: {path}")
        models[row.model] = joblib.load(path)
    return models


def predict_probability_frame(
    models: dict[str, object],
    X: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """Predict positive-class probability frame for selected models."""

    probabilities = {
        model_name: _positive_class_probability(models[model_name], X)
        for model_name in model_names
    }
    return pd.DataFrame(probabilities, index=X.index)


def expected_cost_from_metrics(metrics: dict[str, float | int]) -> float:
    """Compute expected cost from binary metrics."""

    return float(FN_COST * int(metrics["fn"]) + FP_COST * int(metrics["fp"]))


def threshold_grid_table(y_true: pd.Series, y_proba: np.ndarray) -> pd.DataFrame:
    """Evaluate validation expected cost over thresholds 0.01..0.99."""

    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "expected_cost": expected_cost_from_metrics(metrics),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def best_threshold_from_validation(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Return validation cost-minimizing threshold."""

    table = threshold_grid_table(y_true, y_proba)
    return float(table.sort_values(["expected_cost", "threshold"]).iloc[0]["threshold"])


def evaluation_row(
    ablation_id: str,
    ablation_name: str,
    description: str,
    model_role: str,
    n_base_models: int,
    selected_models: list[str],
    threshold: float,
    y_validation: pd.Series,
    validation_proba: np.ndarray,
    y_test: pd.Series,
    test_proba: np.ndarray,
) -> dict[str, object]:
    """Create one ablation result row from validation/test probabilities."""

    validation_metrics = binary_classification_metrics(y_validation, validation_proba, threshold=threshold)
    test_metrics = binary_classification_metrics(y_test, test_proba, threshold=threshold)
    validation_cost = expected_cost_from_metrics(validation_metrics)
    test_cost = expected_cost_from_metrics(test_metrics)
    return {
        "dataset": DATASET,
        "ablation_id": ablation_id,
        "ablation_name": ablation_name,
        "description": description,
        "model_role": model_role,
        "n_base_models": int(n_base_models),
        "selected_models": "; ".join(selected_models),
        "threshold": float(threshold),
        "threshold_source": "validation_cost_min_FN5_FP1",
        "test_set_used_for_training_or_selection": False,
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_pr_auc": validation_metrics["pr_auc"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "validation_brier": validation_metrics["brier_score"],
        "validation_ece": validation_metrics["ece"],
        "validation_expected_cost": validation_cost,
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "brier": test_metrics["brier_score"],
        "ece": test_metrics["ece"],
        "expected_cost": test_cost,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "specificity": test_metrics["specificity"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
    }


def fit_scre_ablation(
    metric_table: pd.DataFrame,
    validation_probabilities: pd.DataFrame,
    test_probabilities: pd.DataFrame,
    y_validation: pd.Series,
    component_weights: dict[str, float],
) -> tuple[SCRECreditHybridClassifier, np.ndarray, np.ndarray]:
    """Fit one SCRE-Credit ablation on validation probabilities."""

    model_names = metric_table["model"].tolist()
    classifier = SCRECreditHybridClassifier(
        component_weights=component_weights,
        final_calibration_method="isotonic",
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    classifier.fit_from_probabilities(validation_probabilities[model_names], y_validation, metric_table)
    validation_proba = classifier.validation_probability_
    test_proba = classifier.predict_proba_from_base(test_probabilities[model_names])[:, 1]
    return classifier, validation_proba, test_proba


def retrain_without_engineered_features(
    selected_metric_table: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Retrain the selected base-model pool after removing engineered features."""

    ABLATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    categorical_columns = [column for column in TAIWAN_CATEGORICAL_COLUMNS if column in X_train.columns]
    specs = build_calibration_model_specs(DATASET, X_train, categorical_columns)
    spec_map = {spec.name: spec for spec in specs}

    rows = []
    models = {}
    validation_probabilities = {}
    test_probabilities = {}
    for row in selected_metric_table.itertuples(index=False):
        if row.model not in spec_map:
            raise KeyError(f"No model spec available for A6 no-engineered retrain: {row.model}")
        spec = spec_map[row.model]
        calibration_method = str(row.selected_calibration_method)
        print(f"A6 retrain without engineered features: {row.model} [{calibration_method}]")
        fitted = _fit_probability_model(spec.estimator, calibration_method, X_train, y_train)
        model_path = ABLATION_MODEL_DIR / f"a6_without_engineered_{row.model}_{calibration_method}.joblib"
        joblib.dump(fitted, model_path)
        models[row.model] = fitted

        validation_proba = _positive_class_probability(fitted, X_validation)
        test_proba = _positive_class_probability(fitted, X_test)
        validation_probabilities[row.model] = validation_proba
        test_probabilities[row.model] = test_proba

        threshold = best_threshold_from_validation(y_validation, validation_proba)
        threshold_metrics = binary_classification_metrics(y_validation, validation_proba, threshold=threshold)
        metrics_050 = binary_classification_metrics(y_validation, validation_proba, threshold=0.5)
        rows.append(
            {
                "dataset": DATASET,
                "model": row.model,
                "source_phase": row.source_phase,
                "model_family": row.model_family,
                "selected_calibration_method": calibration_method,
                "model_path": str(model_path),
                "pr_auc": metrics_050["pr_auc"],
                "roc_auc": metrics_050["roc_auc"],
                "recall": metrics_050["recall"],
                "calibration_error": metrics_050["brier_score"],
                "brier_score": metrics_050["brier_score"],
                "ece": metrics_050["ece"],
                "expected_cost": expected_cost_from_metrics(threshold_metrics),
                "base_selected_threshold": threshold,
                "cost_scenario": PRIMARY_COST_SCENARIO,
                "stability_score": 1.0,
                "faithfulness_score": 1.0,
                "stability_metric_source": "neutral_equal_for_scre_weighting_ablation",
                "faithfulness_metric_source": "neutral_equal_for_scre_weighting_ablation",
            }
        )

    return (
        pd.DataFrame(rows),
        models,
        pd.DataFrame(validation_probabilities, index=X_validation.index),
        pd.DataFrame(test_probabilities, index=X_test.index),
    )


def shap_ranked_features(candidate_columns: list[str]) -> list[str]:
    """Return SHAP-ranked raw features followed by any remaining model columns."""

    path = TABLES_DIR / "shap_top_features.csv"
    features: list[str] = []
    if path.exists():
        shap_table = pd.read_csv(path).sort_values("rank")
        for feature in shap_table["raw_feature"].tolist():
            if feature in candidate_columns and feature not in features:
                features.append(feature)
    for feature in candidate_columns:
        if feature not in features:
            features.append(feature)
    return features


def compute_ablation_faithfulness(
    wrapper: ProbabilityModelWrapper,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    ranked_features: list[str],
    max_k: int,
    ablation_id: str,
) -> tuple[float, pd.DataFrame]:
    """Compute top-k deletion faithfulness for one ablation wrapper."""

    baseline_proba = wrapper.predict_proba(X_test)[:, 1]
    baseline_metrics = faithfulness_metrics(y_test, baseline_proba, threshold, fn_cost=FN_COST, fp_cost=FP_COST)
    replacements = baseline_feature_values(X_train, categorical_columns=TAIWAN_CATEGORICAL_COLUMNS)
    table = topk_deletion_table(
        wrapper,
        X_test,
        y_test,
        ranked_features,
        replacements,
        threshold,
        baseline_proba,
        baseline_metrics,
        max_k=max_k,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    table.insert(0, "ablation_id", ablation_id)
    if table.empty:
        return 0.0, table
    final_row = table.sort_values("step").iloc[-1]
    raw_score = max(0.0, float(final_row["pr_auc_drop"]))
    return raw_score, table


def run_stability_scope(
    df: pd.DataFrame,
    scope: str,
    feature_subset: list[str] | None,
    n_seeds: int,
    explain_rows: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full-validation SHAP stability for one ablation feature scope."""

    config = StabilityDatasetConfig(
        dataset=DATASET,
        scope=scope,
        target=TARGET_COLUMN,
        categorical_columns=[column for column in TAIWAN_CATEGORICAL_COLUMNS if feature_subset is None or column in feature_subset],
        feature_subset=feature_subset,
    )
    seed_results = run_seed_stability(
        df=df,
        config=config,
        seeds=seed_list(n_seeds),
        explain_rows=explain_rows,
        base_random_seed=RANDOM_SEED,
    )
    correlations = pairwise_rank_correlations(seed_results)
    frequency = feature_frequency_table(seed_results)
    score = float(correlations["spearman_rho"].mean()) if not correlations.empty else np.nan
    return score, seed_results, correlations, frequency


def normalize_positive(values: pd.Series) -> pd.Series:
    """Normalize nonnegative scores to [0, 1], returning ones when tied."""

    series = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    value_range = float(series.max() - series.min())
    if not np.isfinite(value_range) or value_range == 0.0:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / value_range


def plot_ablation_barplot(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save ablation metric barplots."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values("ablation_id")
    metrics = ["pr_auc", "roc_auc", "recall", "f1", "faithfulness_score"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15), sharex=True)
    for ax, metric in zip(axes, metrics):
        ax.bar(ordered["ablation_id"], ordered[metric], color="#3f6f95")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
        if metric != "expected_cost":
            ax.set_ylim(0.0, max(1.0, float(ordered[metric].max()) * 1.08))
    axes[-1].set_xlabel("Ablation")
    fig.suptitle("SCRE-Credit Ablation Results", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cost_vs_auc(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save expected cost versus ROC-AUC scatter plot."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(results["roc_auc"], results["expected_cost"], s=80, color="#8f4f6f")
    for row in results.itertuples(index=False):
        ax.annotate(row.ablation_id, (row.roc_auc, row.expected_cost), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Expected cost")
    ax.set_title("Ablation Cost vs ROC-AUC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the ablation study."""

    args = parse_args()
    run_id = start_run(
        "ablation_study",
        dataset=DATASET,
        params={
            "stability_seeds": args.stability_seeds,
            "stability_explain_rows": args.stability_explain_rows or "full_validation_split",
            "faithfulness_max_k": args.faithfulness_max_k,
            "test_set_used_for_selection": False,
        },
        tags=["phase-19", "ablation"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(TAIWAN_MODEL_READY)
        split = stratified_train_validation_test_split(df, target_column=TARGET_COLUMN)
        X_train = split.train.drop(columns=[TARGET_COLUMN])
        y_train = split.train[TARGET_COLUMN]
        X_validation = split.validation.drop(columns=[TARGET_COLUMN])
        y_validation = split.validation[TARGET_COLUMN]
        X_test = split.test.drop(columns=[TARGET_COLUMN])
        y_test = split.test[TARGET_COLUMN]

        no_engineered_columns = [column for column in X_train.columns if column not in TAIWAN_ENGINEERED_FEATURES]
        X_train_no_engineered = X_train[no_engineered_columns].copy()
        X_validation_no_engineered = X_validation[no_engineered_columns].copy()
        X_test_no_engineered = X_test[no_engineered_columns].copy()

        explain_rows = args.stability_explain_rows or len(X_validation)
        full_stability, full_seed, full_corr, full_freq = run_stability_scope(
            df,
            scope="ablation_full_features",
            feature_subset=None,
            n_seeds=args.stability_seeds,
            explain_rows=explain_rows,
        )
        no_engineered_stability, no_eng_seed, no_eng_corr, no_eng_freq = run_stability_scope(
            df,
            scope="ablation_without_engineered_features",
            feature_subset=no_engineered_columns,
            n_seeds=args.stability_seeds,
            explain_rows=explain_rows,
        )
        pd.concat([full_seed, no_eng_seed], ignore_index=True).to_csv(
            TABLES_DIR / "ablation_shap_stability_seed_results.csv",
            index=False,
        )
        pd.concat([full_corr, no_eng_corr], ignore_index=True).to_csv(
            TABLES_DIR / "ablation_shap_rank_correlations.csv",
            index=False,
        )
        pd.concat([full_freq, no_eng_freq], ignore_index=True).to_csv(
            TABLES_DIR / "ablation_shap_topk_frequency.csv",
            index=False,
        )

        metric_table = load_taiwan_metric_table()
        base_models = load_base_models(metric_table)
        model_names = metric_table["model"].tolist()
        validation_probabilities = predict_probability_frame(base_models, X_validation, model_names)
        test_probabilities = predict_probability_frame(base_models, X_test, model_names)

        no_eng_metric_table, no_eng_models, no_eng_validation_probabilities, no_eng_test_probabilities = (
            retrain_without_engineered_features(
                metric_table,
                X_train_no_engineered,
                y_train,
                X_validation_no_engineered,
                y_validation,
                X_test_no_engineered,
            )
        )

        ablation_rows = []
        wrappers: dict[str, tuple[ProbabilityModelWrapper, pd.DataFrame, pd.DataFrame, list[str], float]] = {}

        best_single = metric_table.sort_values(["expected_cost", "pr_auc"], ascending=[True, False]).iloc[0]
        best_model_name = str(best_single["model"])
        best_threshold = float(best_single["base_selected_threshold"])
        best_validation_proba = validation_probabilities[best_model_name].to_numpy()
        best_test_proba = test_probabilities[best_model_name].to_numpy()
        ablation_rows.append(
            evaluation_row(
                "A0",
                "Best single model",
                "Validation-cost-selected single calibrated base model.",
                "base_model",
                1,
                [best_model_name],
                best_threshold,
                y_validation,
                best_validation_proba,
                y_test,
                best_test_proba,
            )
        )
        wrappers["A0"] = (
            ProbabilityModelWrapper({best_model_name: base_models[best_model_name]}, single_model_name=best_model_name),
            X_train,
            X_test,
            shap_ranked_features(list(X_test.columns)),
            best_threshold,
        )

        ablations = [
            (
                "A1",
                "Full SCRE-Credit",
                "All selected base models with full SCRE component weights.",
                metric_table,
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without([]),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A2",
                "SCRE without calibration score",
                "Calibration component removed and remaining SCRE weights renormalized.",
                metric_table,
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without(["calibration_error"]),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A3",
                "SCRE without cost score",
                "Expected-cost component removed and remaining SCRE weights renormalized.",
                metric_table,
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without(["expected_cost"]),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A4",
                "SCRE without stability score",
                "Stability component removed and remaining SCRE weights renormalized.",
                metric_table,
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without(["stability_score"]),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A5",
                "SCRE without faithfulness score",
                "Faithfulness component removed and remaining SCRE weights renormalized.",
                metric_table,
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without(["faithfulness_score"]),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A6",
                "SCRE without engineered features",
                "Selected base model pool retrained after removing all Taiwan engineered features.",
                no_eng_metric_table,
                no_eng_validation_probabilities,
                no_eng_test_probabilities,
                no_eng_models,
                component_weights_without([]),
                X_train_no_engineered,
                X_test_no_engineered,
                no_engineered_stability,
            ),
            (
                "A7",
                "SCRE without monotonic models",
                "Monotonic XGBoost and monotonic LightGBM removed from the base model pool.",
                metric_table.loc[~metric_table["model"].str.contains("monotonic", case=False, na=False)].copy(),
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without([]),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A8",
                "SCRE with performance-only weights",
                "Only PR-AUC, ROC-AUC, and recall are used for SCRE reliability weights.",
                metric_table,
                validation_probabilities,
                test_probabilities,
                base_models,
                performance_only_weights(),
                X_train,
                X_test,
                full_stability,
            ),
            (
                "A9",
                "SCRE without scorecard baseline",
                "WOE scorecard logistic baseline removed from the base model pool.",
                metric_table.loc[~metric_table["model"].str.contains("scorecard", case=False, na=False)].copy(),
                validation_probabilities,
                test_probabilities,
                base_models,
                component_weights_without([]),
                X_train,
                X_test,
                full_stability,
            ),
        ]

        for (
            ablation_id,
            ablation_name,
            description,
            table,
            val_probs,
            tst_probs,
            models,
            weights,
            train_frame,
            test_frame,
            shap_stability,
        ) in ablations:
            print(f"Evaluating {ablation_id}: {ablation_name}")
            selected_models = table["model"].tolist()
            classifier, validation_proba, test_proba = fit_scre_ablation(
                table,
                val_probs,
                tst_probs,
                y_validation,
                weights,
            )
            row = evaluation_row(
                ablation_id,
                ablation_name,
                description,
                "scre_credit",
                len(selected_models),
                selected_models,
                classifier.threshold_,
                y_validation,
                validation_proba,
                y_test,
                test_proba,
            )
            row["component_weights"] = str(weights)
            row["shap_stability"] = shap_stability
            row["shap_stability_source"] = "mean_pairwise_spearman_from_50_seed_lightgbm_rank_stability"
            ablation_rows.append(row)
            wrappers[ablation_id] = (
                ProbabilityModelWrapper({name: models[name] for name in selected_models}, classifier=classifier),
                train_frame,
                test_frame,
                shap_ranked_features(list(test_frame.columns)),
                classifier.threshold_,
            )

        results = pd.DataFrame(ablation_rows)
        if "shap_stability" not in results.columns:
            results["shap_stability"] = np.nan
        results.loc[results["ablation_id"] == "A0", "shap_stability"] = full_stability
        results.loc[results["ablation_id"] == "A0", "shap_stability_source"] = (
            "best_single_model_reported_against_full_feature_lightgbm_stability_context"
        )

        faithfulness_rows = []
        faithfulness_raw_scores = {}
        for ablation_id, (wrapper, train_frame, test_frame, ranked_features, threshold) in wrappers.items():
            print(f"Computing faithfulness top-k deletion for {ablation_id}")
            raw_score, details = compute_ablation_faithfulness(
                wrapper,
                train_frame,
                test_frame,
                y_test,
                threshold,
                ranked_features,
                max_k=args.faithfulness_max_k,
                ablation_id=ablation_id,
            )
            faithfulness_raw_scores[ablation_id] = raw_score
            faithfulness_rows.append(details)

        faithfulness_details = pd.concat(faithfulness_rows, ignore_index=True)
        faithfulness_details.to_csv(TABLES_DIR / "ablation_faithfulness_details.csv", index=False)
        results["faithfulness_raw_topk_pr_auc_drop"] = results["ablation_id"].map(faithfulness_raw_scores)
        results["faithfulness_score"] = normalize_positive(results["faithfulness_raw_topk_pr_auc_drop"])
        results["faithfulness_score_source"] = (
            f"normalized_top_{args.faithfulness_max_k}_deletion_pr_auc_drop_on_test_predictions"
        )
        results.loc[results["ablation_id"] == "A6", "feature_scope"] = "without_engineered_features"
        results["feature_scope"] = results["feature_scope"].fillna("full_feature_set")

        ordered_ids = [f"A{i}" for i in range(10)]
        results["ablation_order"] = results["ablation_id"].map({key: idx for idx, key in enumerate(ordered_ids)})
        results = results.sort_values("ablation_order").drop(columns=["ablation_order"]).reset_index(drop=True)

        results_path = TABLES_DIR / "ablation_results.csv"
        results.to_csv(results_path, index=False)
        plot_ablation_barplot(results, FIGURES_DIR / "ablation_results_barplot.png")
        plot_cost_vs_auc(results, FIGURES_DIR / "ablation_cost_vs_auc.png")

        print(
            results[
                [
                    "ablation_id",
                    "ablation_name",
                    "roc_auc",
                    "pr_auc",
                    "recall",
                    "f1",
                    "brier",
                    "ece",
                    "expected_cost",
                    "shap_stability",
                    "faithfulness_score",
                ]
            ].to_string(index=False)
        )
        finish_run(
            run_id,
            metrics={
                "ablation_rows": int(len(results)),
                "best_pr_auc": float(results["pr_auc"].max()),
                "best_pr_auc_ablation": str(results.sort_values("pr_auc", ascending=False).iloc[0]["ablation_id"]),
                "lowest_expected_cost": float(results["expected_cost"].min()),
                "lowest_expected_cost_ablation": str(
                    results.sort_values("expected_cost", ascending=True).iloc[0]["ablation_id"]
                ),
                "full_feature_shap_stability": float(full_stability),
                "no_engineered_shap_stability": float(no_engineered_stability),
            },
            artifacts={
                "ablation_results": results_path,
                "ablation_results_barplot": FIGURES_DIR / "ablation_results_barplot.png",
                "ablation_cost_vs_auc": FIGURES_DIR / "ablation_cost_vs_auc.png",
                "ablation_faithfulness_details": TABLES_DIR / "ablation_faithfulness_details.csv",
                "ablation_shap_rank_correlations": TABLES_DIR / "ablation_shap_rank_correlations.csv",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
