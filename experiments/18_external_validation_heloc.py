"""Run Prompt 20 HELOC external validation for the SCRE-Credit framework."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import FN_COST, FP_COST, RANDOM_SEED, load_experiment_config
from data.split_leakage import stratified_train_validation_test_split
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from monotonic_models import HELOC_MONOTONIC_PRIORS, monotonic_constraint_vector
from scorecard import make_scorecard_pipeline
from scre_credit import make_model_pipeline, make_numeric_model_pipeline
from src.config.paths import FIGURES_DIR, HELOC_MODEL_READY, MODELS_DIR, SHAP_DIR, TABLES_DIR
from src.explainability.faithfulness import (
    baseline_feature_values,
    faithfulness_metrics,
    group_deletion_table,
    random_feature_deletion_table,
    single_feature_perturbation_table,
    topk_deletion_table,
    topk_insertion_table,
)
from src.explainability.shap_analysis import (
    compute_shap_analysis,
    save_global_shap_plots,
    shap_top_features_table,
)
from src.explainability.shap_stability import (
    StabilityDatasetConfig,
    feature_frequency_table,
    pairwise_rank_correlations,
    run_seed_stability,
)
from src.features.feature_registry import HELOC_CATEGORICAL_COLUMNS, HELOC_ENGINEERED_FEATURES, HELOC_TARGET
from src.models.scre_credit import SCRECreditHybridClassifier


try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


DATASET = "heloc"
CALIBRATION_METHOD = "isotonic"
CV_FOLDS = 3
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
MODEL_OUTPUT_DIR = MODELS_DIR / "external_validation_heloc"
SHAP_OUTPUT_DIR = SHAP_DIR / "external_validation_heloc"

HELOC_FAITHFULNESS_GROUPS = {
    "credit_history": [
        "ExternalRiskEstimate",
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "AverageMInFile",
        "credit_history_length_proxy",
    ],
    "delinquency": [
        "NumTrades60Ever2DerogPubRec",
        "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MaxDelqEver",
        "delinquency_intensity",
        "negative_trade_signal",
    ],
    "inquiry_pressure": [
        "MSinceMostRecentInqexcl7days",
        "NumInqLast6M",
        "NumInqLast6Mexcl7days",
        "recent_inquiry_pressure",
    ],
    "credit_burden": [
        "NetFractionRevolvingBurden",
        "NetFractionInstallBurden",
        "NumBank2NatlTradesWHighUtilization",
        "revolving_burden_proxy",
        "installment_burden_proxy",
        "high_utilization_signal",
    ],
    "trade_activity": [
        "NumSatisfactoryTrades",
        "NumTotalTrades",
        "NumTradesOpeninLast12M",
        "PercentInstallTrades",
        "NumRevolvingTradesWBalance",
        "NumInstallTradesWBalance",
        "PercentTradesWBalance",
        "trade_activity_ratio",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse external-validation runtime controls."""

    config = load_experiment_config()
    default_stability_seeds = int(config.get("shap_stability_seeds", 50))
    parser = argparse.ArgumentParser(description="Run HELOC external validation for SCRE-Credit.")
    parser.add_argument("--stability-seeds", type=int, default=default_stability_seeds)
    parser.add_argument(
        "--stability-explain-rows",
        type=int,
        default=None,
        help="Rows used for reduced HELOC SHAP stability. Default uses the full validation split.",
    )
    parser.add_argument("--faithfulness-max-k", type=int, default=10)
    return parser.parse_args()


def seed_list(n_seeds: int, base_seed: int = RANDOM_SEED) -> list[int]:
    """Return deterministic model seeds for external-validation stability."""

    rng = np.random.default_rng(base_seed)
    return [int(seed) for seed in rng.choice(np.arange(1, 1_000_000), size=n_seeds, replace=False)]


def build_reduced_model_set(X_train: pd.DataFrame) -> dict[str, object]:
    """Build the required reduced HELOC model set."""

    if XGBClassifier is None:
        raise ImportError("xgboost is required for HELOC external validation.")
    if LGBMClassifier is None:
        raise ImportError("lightgbm is required for HELOC external validation.")

    categorical = [column for column in HELOC_CATEGORICAL_COLUMNS if column in X_train.columns]
    return {
        "logistic_regression": make_model_pipeline(
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_SEED),
            X_train,
            categorical,
        ),
        "scorecard_baseline": make_scorecard_pipeline(categorical_columns=categorical),
        "xgboost": make_model_pipeline(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=4,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            X_train,
            categorical,
        ),
        "lightgbm": make_model_pipeline(
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            ),
            X_train,
            categorical,
        ),
        "monotonic_boosting": make_numeric_model_pipeline(
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=40,
                monotone_constraints=monotonic_constraint_vector(list(X_train.columns), HELOC_MONOTONIC_PRIORS),
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
            )
        ),
    }


def fit_calibrated_model(estimator, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit an isotonic calibrated model using train-only CV."""

    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=CALIBRATION_METHOD,
        cv=CV_FOLDS,
        n_jobs=1,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def expected_cost_from_metrics(metrics: dict[str, float | int]) -> float:
    """Compute FN/FP expected cost from a metric dictionary."""

    return float(FN_COST * int(metrics["fn"]) + FP_COST * int(metrics["fp"]))


def threshold_analysis_table(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Evaluate all thresholds for one model and split."""

    rows = []
    for threshold in THRESHOLDS:
        metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
        expected_cost = expected_cost_from_metrics(metrics)
        rows.append(
            {
                "dataset": DATASET,
                "model": model_name,
                "split": split_name,
                "threshold": float(threshold),
                "fn_cost": float(FN_COST),
                "fp_cost": float(FP_COST),
                "expected_cost": expected_cost,
                "expected_cost_per_1000": expected_cost / len(y_true) * 1000.0,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def summarize_split_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
    model_role: str,
) -> dict[str, object]:
    """Return the result-row metrics for one model/split."""

    metrics = binary_classification_metrics(y_true, y_proba, threshold=threshold)
    expected_cost = expected_cost_from_metrics(metrics)
    return {
        "dataset": DATASET,
        "model": model_name,
        "model_role": model_role,
        "split": split_name,
        "calibration_method": CALIBRATION_METHOD if model_role == "base_model" else "weighted_sum_plus_isotonic",
        "threshold": float(threshold),
        "threshold_source": "validation_cost_min_FN5_FP1",
        "fn_cost": float(FN_COST),
        "fp_cost": float(FP_COST),
        "expected_cost": expected_cost,
        "expected_cost_per_1000": expected_cost / len(y_true) * 1000.0,
        "test_set_used_for_training_or_selection": False,
        **metrics,
    }


def best_validation_threshold(thresholds: pd.DataFrame) -> float:
    """Select the validation threshold minimizing expected cost."""

    best = thresholds.sort_values(["expected_cost", "threshold"]).iloc[0]
    return float(best["threshold"])


def build_scre_metric_table(validation_rows: pd.DataFrame) -> pd.DataFrame:
    """Create SCRE-Credit metric table from validation performance."""

    metric_rows = []
    for row in validation_rows.itertuples(index=False):
        metric_rows.append(
            {
                "model": row.model,
                "pr_auc": float(row.pr_auc),
                "roc_auc": float(row.roc_auc),
                "recall": float(row.recall),
                "calibration_error": float(row.ece),
                "expected_cost": float(row.expected_cost),
                "stability_score": 1.0,
                "faithfulness_score": 1.0,
                "stability_score_source": "neutral_equal_weight_external_validation",
                "faithfulness_score_source": "neutral_equal_weight_external_validation",
            }
        )
    return pd.DataFrame(metric_rows)


def fit_scre_credit(
    validation_probabilities: pd.DataFrame,
    y_validation: pd.Series,
    test_probabilities: pd.DataFrame,
    metric_table: pd.DataFrame,
) -> tuple[SCRECreditHybridClassifier, np.ndarray, np.ndarray]:
    """Fit SCRE-Credit on validation probabilities and predict validation/test."""

    scre = SCRECreditHybridClassifier(fn_cost=FN_COST, fp_cost=FP_COST, final_calibration_method="isotonic")
    scre.fit_from_probabilities(validation_probabilities, y_validation, metric_table)
    validation_proba = scre.predict_proba_from_base(validation_probabilities)[:, 1]
    test_proba = scre.predict_proba_from_base(test_probabilities)[:, 1]
    return scre, validation_proba, test_proba


def tree_explanation_model_name(results: pd.DataFrame) -> str:
    """Choose the best validation tree model for SHAP and reduced faithfulness."""

    candidates = ["xgboost", "lightgbm", "monotonic_boosting"]
    subset = results.loc[
        (results["split"] == "validation")
        & (results["model"].isin(candidates))
        & (results["model_role"] == "base_model")
    ]
    if subset.empty:
        raise ValueError("No tree model available for HELOC SHAP explanation.")
    return str(subset.sort_values(["pr_auc", "roc_auc"], ascending=False).iloc[0]["model"])


def ranked_raw_features_from_shap(shap_table: pd.DataFrame, candidate_columns: list[str]) -> list[str]:
    """Return SHAP-ranked raw features followed by any remaining columns."""

    ranked = []
    for feature in shap_table.sort_values("rank")["raw_feature"].tolist():
        if feature in candidate_columns and feature not in ranked:
            ranked.append(feature)
    for feature in candidate_columns:
        if feature not in ranked:
            ranked.append(feature)
    return ranked


def run_global_shap(
    model,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    explanation_model_name: str,
) -> pd.DataFrame:
    """Run global SHAP explanation for the selected HELOC tree model."""

    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    explain_rows = len(X_validation)
    computation = compute_shap_analysis(
        model,
        X_background=X_train,
        X_explain=X_validation,
        model_name=explanation_model_name,
        max_background_rows=min(500, len(X_train)),
        random_seed=RANDOM_SEED,
    )
    save_global_shap_plots(computation, SHAP_OUTPUT_DIR)
    top_features = shap_top_features_table(computation, raw_features=list(X_train.columns), top_n=len(X_train.columns))
    top_features.insert(0, "explain_rows", explain_rows)
    top_features.insert(0, "explainer_type", computation.explainer_type)
    top_features.insert(0, "model", explanation_model_name)
    top_features.insert(0, "dataset", DATASET)
    top_features.to_csv(TABLES_DIR / "external_validation_heloc_shap_top_features.csv", index=False)
    return top_features


def run_reduced_shap_stability(
    df: pd.DataFrame,
    seeds: list[int],
    explain_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run reduced HELOC SHAP stability on engineered concept features."""

    config = StabilityDatasetConfig(
        dataset=DATASET,
        scope="external_reduced",
        target=HELOC_TARGET,
        categorical_columns=[column for column in HELOC_CATEGORICAL_COLUMNS if column in df.columns],
        feature_subset=[column for column in HELOC_ENGINEERED_FEATURES if column in df.columns],
    )
    seed_results = run_seed_stability(
        df=df,
        config=config,
        seeds=seeds,
        explain_rows=explain_rows,
        base_random_seed=RANDOM_SEED,
    )
    correlations = pairwise_rank_correlations(seed_results)
    frequency = feature_frequency_table(seed_results)
    seed_results.to_csv(TABLES_DIR / "external_validation_heloc_shap_stability_seed_results.csv", index=False)
    correlations.to_csv(TABLES_DIR / "external_validation_heloc_shap_rank_correlations.csv", index=False)
    frequency.to_csv(TABLES_DIR / "external_validation_heloc_shap_topk_frequency.csv", index=False)
    return seed_results, correlations, frequency


def run_reduced_faithfulness(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    ranked_features: list[str],
    max_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run reduced HELOC faithfulness checks for the selected tree model."""

    replacements = baseline_feature_values(X_train, categorical_columns=HELOC_CATEGORICAL_COLUMNS)
    baseline_metrics = faithfulness_metrics(y_test, y_proba, threshold, fn_cost=FN_COST, fp_cost=FP_COST)
    single = single_feature_perturbation_table(
        model,
        X_test,
        y_test,
        ranked_features,
        replacements,
        threshold,
        y_proba,
        baseline_metrics,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    deletion = topk_deletion_table(
        model,
        X_test,
        y_test,
        ranked_features,
        replacements,
        threshold,
        y_proba,
        baseline_metrics,
        max_k=max_k,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    insertion = topk_insertion_table(
        model,
        X_test,
        y_test,
        ranked_features,
        replacements,
        threshold,
        y_proba,
        baseline_metrics,
        max_k=max_k,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    random_deletion = random_feature_deletion_table(
        model,
        X_test,
        y_test,
        list(X_test.columns),
        replacements,
        threshold,
        y_proba,
        baseline_metrics,
        random_seed=RANDOM_SEED,
        max_k=max_k,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    group_results = group_deletion_table(
        model,
        X_test,
        y_test,
        HELOC_FAITHFULNESS_GROUPS,
        replacements,
        threshold,
        y_proba,
        baseline_metrics,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )
    results = pd.concat([single, deletion, insertion, random_deletion], ignore_index=True)
    for table in [results, group_results]:
        table.insert(0, "baseline_expected_cost", baseline_metrics["expected_cost"])
        table.insert(0, "baseline_recall", baseline_metrics["recall"])
        table.insert(0, "baseline_pr_auc", baseline_metrics["pr_auc"])
        table.insert(0, "baseline_roc_auc", baseline_metrics["roc_auc"])
        table.insert(0, "threshold", threshold)
        table.insert(0, "dataset", DATASET)
    results.to_csv(TABLES_DIR / "external_validation_heloc_faithfulness_results.csv", index=False)
    group_results.to_csv(TABLES_DIR / "external_validation_heloc_faithfulness_group_results.csv", index=False)
    return results, group_results


def build_taiwan_vs_heloc_comparison(heloc_results: pd.DataFrame) -> pd.DataFrame:
    """Create a comparable Taiwan-vs-HELOC framework result table."""

    mapping = {
        "logistic_regression": "Logistic Regression",
        "woe_scorecard_logistic_regression": "Scorecard baseline",
        "scorecard_baseline": "Scorecard baseline",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "monotonic_lightgbm": "Monotonic boosting",
        "monotonic_boosting": "Monotonic boosting",
        "SCRE-Credit": "SCRE-Credit",
    }
    rows = []
    taiwan_path = TABLES_DIR / "scre_credit_results_taiwan.csv"
    if taiwan_path.exists():
        taiwan = pd.read_csv(taiwan_path)
        taiwan = taiwan.loc[taiwan["split"] == "test"].copy()
        for source_model, label in mapping.items():
            match = taiwan.loc[taiwan["model"] == source_model]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row["framework_model"] = label
            rows.append(row)

    heloc = heloc_results.loc[heloc_results["split"] == "test"].copy()
    for source_model, label in mapping.items():
        match = heloc.loc[heloc["model"] == source_model]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        row["framework_model"] = label
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if "expected_cost_per_1000" not in comparison.columns:
        comparison["expected_cost_per_1000"] = np.nan
    for index, row in comparison.iterrows():
        if pd.notna(row.get("expected_cost_per_1000", np.nan)):
            continue
        count_columns = ["tn", "fp", "fn", "tp"]
        if all(column in comparison.columns for column in count_columns):
            n_rows = sum(float(row.get(column, 0.0)) for column in count_columns)
            if n_rows > 0 and pd.notna(row.get("expected_cost", np.nan)):
                comparison.loc[index, "expected_cost_per_1000"] = float(row["expected_cost"]) / n_rows * 1000.0
    keep = [
        "dataset",
        "framework_model",
        "model",
        "model_role",
        "calibration_method",
        "threshold",
        "expected_cost",
        "expected_cost_per_1000",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "f1",
        "brier_score",
        "ece",
    ]
    comparison = comparison[[column for column in keep if column in comparison.columns]]
    comparison.to_csv(TABLES_DIR / "taiwan_vs_heloc_framework_comparison.csv", index=False)
    return comparison


def plot_taiwan_vs_heloc_metrics(comparison: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Taiwan-vs-HELOC discrimination and threshold metrics."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["roc_auc", "pr_auc", "recall", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        pivot = comparison.pivot_table(index="framework_model", columns="dataset", values=metric, aggfunc="first")
        pivot = pivot.sort_index()
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylim(0.0, max(1.0, float(np.nanmax(pivot.to_numpy())) * 1.1))
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_taiwan_vs_heloc_calibration(comparison: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Taiwan-vs-HELOC calibration metrics."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["brier_score", "ece", "expected_cost_per_1000"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        pivot = comparison.pivot_table(index="framework_model", columns="dataset", values=metric, aggfunc="first")
        pivot = pivot.sort_index()
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the full HELOC external-validation script."""

    args = parse_args()
    run_id = start_run(
        "external_validation_heloc",
        dataset=DATASET,
        params={
            "model_set": "logistic, scorecard, xgboost, lightgbm, monotonic_boosting, scre_credit",
            "calibration_method": CALIBRATION_METHOD,
            "calibration_cv_folds": CV_FOLDS,
            "threshold_grid": "0.01_to_0.99_step_0.01",
            "shap_stability_seeds": args.stability_seeds,
            "faithfulness_max_k": args.faithfulness_max_k,
        },
        tags=["phase-18", "external-validation", "heloc"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(HELOC_MODEL_READY)
        split = stratified_train_validation_test_split(df, target_column=HELOC_TARGET)
        X_train = split.train.drop(columns=[HELOC_TARGET])
        y_train = split.train[HELOC_TARGET]
        X_validation = split.validation.drop(columns=[HELOC_TARGET])
        y_validation = split.validation[HELOC_TARGET]
        X_test = split.test.drop(columns=[HELOC_TARGET])
        y_test = split.test[HELOC_TARGET]

        estimators = build_reduced_model_set(X_train)
        fitted_models: dict[str, object] = {}
        validation_probabilities: dict[str, np.ndarray] = {}
        test_probabilities: dict[str, np.ndarray] = {}
        result_rows = []
        threshold_tables = []

        for model_name, estimator in estimators.items():
            print(f"Fitting calibrated HELOC model: {model_name}")
            model = fit_calibrated_model(estimator, X_train, y_train)
            fitted_models[model_name] = model
            joblib.dump(model, MODEL_OUTPUT_DIR / f"{model_name}_{CALIBRATION_METHOD}.joblib")

            validation_proba = model.predict_proba(X_validation)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
            validation_probabilities[model_name] = validation_proba
            test_probabilities[model_name] = test_proba

            threshold_table = threshold_analysis_table(y_validation, validation_proba, model_name, "validation")
            threshold_tables.append(threshold_table)
            threshold = best_validation_threshold(threshold_table)
            result_rows.append(
                summarize_split_metrics(
                    y_validation,
                    validation_proba,
                    threshold,
                    model_name,
                    "validation",
                    "base_model",
                )
            )
            result_rows.append(
                summarize_split_metrics(
                    y_test,
                    test_proba,
                    threshold,
                    model_name,
                    "test",
                    "base_model",
                )
            )

        validation_results = pd.DataFrame(result_rows)
        metric_table = build_scre_metric_table(validation_results.loc[validation_results["split"] == "validation"])
        metric_table.to_csv(TABLES_DIR / "external_validation_heloc_scre_metric_inputs.csv", index=False)
        scre, scre_validation_proba, scre_test_proba = fit_scre_credit(
            pd.DataFrame(validation_probabilities),
            y_validation.reset_index(drop=True),
            pd.DataFrame(test_probabilities),
            metric_table,
        )
        joblib.dump(scre, MODEL_OUTPUT_DIR / "scre_credit.joblib")
        scre.weight_table_.to_csv(TABLES_DIR / "external_validation_heloc_scre_weights.csv", index=False)

        result_rows.append(
            summarize_split_metrics(
                y_validation,
                scre_validation_proba,
                scre.threshold_,
                "SCRE-Credit",
                "validation",
                "scre_credit",
            )
        )
        result_rows.append(
            summarize_split_metrics(
                y_test,
                scre_test_proba,
                scre.threshold_,
                "SCRE-Credit",
                "test",
                "scre_credit",
            )
        )

        results = pd.DataFrame(result_rows)
        results.insert(1, "dataset_role", "external_validation")
        results.insert(2, "framework_validation_mode", "same_framework_retrained_on_heloc_feature_space")
        results.insert(3, "target", HELOC_TARGET)
        results.to_csv(TABLES_DIR / "external_validation_heloc_results.csv", index=False)

        threshold_analysis = pd.concat(threshold_tables, ignore_index=True)
        threshold_analysis.to_csv(TABLES_DIR / "external_validation_heloc_threshold_analysis.csv", index=False)

        explanation_model_name = tree_explanation_model_name(results)
        shap_table = run_global_shap(
            fitted_models[explanation_model_name],
            X_train,
            X_validation,
            explanation_model_name,
        )
        ranked_features = ranked_raw_features_from_shap(shap_table, list(X_test.columns))

        stability_explain_rows = args.stability_explain_rows or len(X_validation)
        stability_seed_results, stability_correlations, stability_frequency = run_reduced_shap_stability(
            df,
            seeds=seed_list(args.stability_seeds),
            explain_rows=stability_explain_rows,
        )

        explanation_threshold = float(
            results.loc[
                (results["model"] == explanation_model_name) & (results["split"] == "validation"),
                "threshold",
            ].iloc[0]
        )
        explanation_test_proba = test_probabilities[explanation_model_name]
        faithfulness_results, faithfulness_group_results = run_reduced_faithfulness(
            fitted_models[explanation_model_name],
            X_train,
            X_test,
            y_test,
            explanation_test_proba,
            explanation_threshold,
            ranked_features,
            max_k=args.faithfulness_max_k,
        )

        comparison = build_taiwan_vs_heloc_comparison(results)
        plot_taiwan_vs_heloc_metrics(comparison, FIGURES_DIR / "taiwan_vs_heloc_metrics.png")
        plot_taiwan_vs_heloc_calibration(comparison, FIGURES_DIR / "taiwan_vs_heloc_calibration.png")

        print(results.to_string(index=False))
        print(comparison.to_string(index=False))
        finish_run(
            run_id,
            metrics={
                "heloc_result_rows": int(len(results)),
                "comparison_rows": int(len(comparison)),
                "explanation_model": explanation_model_name,
                "shap_top_features_rows": int(len(shap_table)),
                "stability_seed_result_rows": int(len(stability_seed_results)),
                "stability_pairwise_rows": int(len(stability_correlations)),
                "stability_frequency_rows": int(len(stability_frequency)),
                "faithfulness_rows": int(len(faithfulness_results)),
                "faithfulness_group_rows": int(len(faithfulness_group_results)),
                "scre_test_pr_auc": float(
                    results.loc[
                        (results["model"] == "SCRE-Credit") & (results["split"] == "test"),
                        "pr_auc",
                    ].iloc[0]
                ),
            },
            artifacts={
                "external_validation_heloc_results": TABLES_DIR / "external_validation_heloc_results.csv",
                "taiwan_vs_heloc_framework_comparison": TABLES_DIR
                / "taiwan_vs_heloc_framework_comparison.csv",
                "taiwan_vs_heloc_metrics": FIGURES_DIR / "taiwan_vs_heloc_metrics.png",
                "taiwan_vs_heloc_calibration": FIGURES_DIR / "taiwan_vs_heloc_calibration.png",
                "shap_global_bar": SHAP_OUTPUT_DIR / "global_bar.png",
                "shap_global_beeswarm": SHAP_OUTPUT_DIR / "global_beeswarm.png",
                "shap_stability_seed_results": TABLES_DIR
                / "external_validation_heloc_shap_stability_seed_results.csv",
                "faithfulness_results": TABLES_DIR / "external_validation_heloc_faithfulness_results.csv",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
