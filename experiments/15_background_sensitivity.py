"""Run SHAP background-set sensitivity analysis."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from src.config.paths import CALIBRATED_MODELS_DIR, FIGURES_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.explainability.background_sensitivity import (
    build_background_sets,
    combined_background_sensitivity_results,
    compute_all_background_importance,
    pairwise_background_comparisons,
    plot_background_rank_shift,
    plot_background_shap_drift,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


DATASET = "taiwan"
MODEL_NAME = "lightgbm"


def parse_args() -> argparse.Namespace:
    """Parse runtime options."""

    parser = argparse.ArgumentParser(description="Run SHAP background sensitivity.")
    parser.add_argument("--background-size", type=int, default=200)
    parser.add_argument("--kmeans-size", type=int, default=50)
    parser.add_argument("--explain-rows", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run all background sensitivity outputs."""

    args = parse_args()
    run_id = start_run(
        "background_sensitivity",
        dataset=DATASET,
        params={
            "model": MODEL_NAME,
            "background_size": args.background_size,
            "kmeans_size": args.kmeans_size,
            "explain_rows": args.explain_rows,
            "top_k": args.top_k,
            "random_seed": RANDOM_SEED,
        },
        tags=["phase-15", "background-sensitivity"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        model_path = CALIBRATED_MODELS_DIR / DATASET / f"{MODEL_NAME}_uncalibrated.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")
        pipeline = joblib.load(model_path)

        df = pd.read_csv(TAIWAN_MODEL_READY)
        split = stratified_train_validation_test_split(df, target_column=TARGET_COLUMN)
        X_train = split.train.drop(columns=[TARGET_COLUMN])
        y_train = split.train[TARGET_COLUMN]
        X_explain = split.test.drop(columns=[TARGET_COLUMN]).sample(
            min(args.explain_rows, len(split.test)),
            random_state=RANDOM_SEED,
        )

        backgrounds = build_background_sets(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            sample_size=args.background_size,
            kmeans_size=args.kmeans_size,
            random_seed=RANDOM_SEED,
        )
        importance = compute_all_background_importance(
            pipeline=pipeline,
            backgrounds=backgrounds,
            X_explain=X_explain,
            raw_features=list(X_train.columns),
        )
        comparisons = pairwise_background_comparisons(importance, top_k=args.top_k)
        results = combined_background_sensitivity_results(
            importance=importance,
            comparisons=comparisons,
            dataset=DATASET,
            model=MODEL_NAME,
        )

        results_path = TABLES_DIR / "background_sensitivity_results.csv"
        rank_shift_path = FIGURES_DIR / "background_rank_shift.png"
        drift_path = FIGURES_DIR / "background_shap_drift.png"
        results.to_csv(results_path, index=False)
        plot_background_rank_shift(importance, rank_shift_path, top_n=15)
        plot_background_shap_drift(importance, drift_path, top_n=15)

        print(
            importance.loc[importance["rank"] <= 10, [
                "background",
                "feature",
                "rank",
                "mean_abs_shap",
                "mean_shap",
            ]]
            .sort_values(["background", "rank"])
            .to_string(index=False)
        )
        print(
            comparisons[
                [
                    "background",
                    "top_k_overlap_rate",
                    "spearman_rho",
                    "kendall_tau",
                    "mean_abs_shap_drift",
                    "mean_relative_shap_drift",
                ]
            ].to_string(index=False)
        )

        finish_run(
            run_id,
            metrics={
                "feature_importance_rows": int(len(importance)),
                "background_comparison_rows": int(len(comparisons)),
                "min_topk_overlap_rate": float(comparisons["top_k_overlap_rate"].min()),
                "mean_spearman_rho": float(comparisons["spearman_rho"].mean()),
                "max_mean_abs_shap_drift": float(comparisons["mean_abs_shap_drift"].max()),
            },
            artifacts={
                "background_sensitivity_results": results_path,
                "background_rank_shift": rank_shift_path,
                "background_shap_drift": drift_path,
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
