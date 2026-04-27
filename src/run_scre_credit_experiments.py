"""Run SCRE-Credit experiments on Taiwan and/or HELOC."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import FN_COST, FP_COST, RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE
from data_preprocessing import PROJECT_ROOT, TARGET_COLUMN
from evaluation import binary_classification_metrics, threshold_optimization_table
from experiment_registry import compact_utc_timestamp, finish_run, short_run_id, start_run
from heloc_preprocessing import (
    HELOC_CATEGORICAL_COLUMNS,
    HELOC_PROCESSED_PATH,
    HELOC_TARGET,
    clean_heloc_dataset,
    load_heloc_dataset,
)
from model_training import CATEGORICAL_COLUMNS, split_features_target
from reliability_analysis import cost_curve_table, manual_review_band_table, metric_ci_table
from scre_credit import ScreCreditEnsemble, build_candidate_models, three_way_policy_metrics


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
TAIWAN_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "taiwan_model_ready.csv"


def load_dataset(dataset: str):
    """Return X/y/split/categorical metadata for a supported dataset."""

    if dataset == "taiwan":
        df = pd.read_csv(TAIWAN_PROCESSED_PATH)
        split = split_features_target(df, target_column=TARGET_COLUMN, random_state=RANDOM_SEED)
        return split.X_train, split.X_test, split.y_train, split.y_test, CATEGORICAL_COLUMNS

    if dataset == "heloc":
        if HELOC_PROCESSED_PATH.exists():
            df = pd.read_csv(HELOC_PROCESSED_PATH)
        else:
            df = clean_heloc_dataset(load_heloc_dataset())
            HELOC_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(HELOC_PROCESSED_PATH, index=False)

        X = df.drop(columns=[HELOC_TARGET])
        y = df[HELOC_TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_SEED,
        )
        return X_train, X_test, y_train, y_test, HELOC_CATEGORICAL_COLUMNS

    raise ValueError(f"Unsupported dataset: {dataset}")


def run_one_dataset(dataset: str, n_bootstraps: int, quick: bool) -> dict:
    """Fit SCRE-Credit and save core outputs for one dataset."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, categorical_columns = load_dataset(dataset)
    candidates = build_candidate_models(
        X_train,
        categorical_columns=categorical_columns,
        random_state=RANDOM_SEED,
        include_catboost=not quick,
        include_scorecard=True,
        model_profile="heloc_reduced" if dataset == "heloc" else "taiwan_full",
    )

    run_id = start_run(
        name="scre_credit",
        dataset=dataset,
        params={
            "n_candidates": len(candidates),
            "candidate_models": [name for name, _ in candidates],
            "n_bootstraps": n_bootstraps,
            "quick": quick,
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
            "top_level_test_size": TEST_SIZE,
            "internal_validation_size_of_train_validation": VALIDATION_SIZE / (1.0 - TEST_SIZE),
        },
        tags=["scre-credit", "external-validation" if dataset == "heloc" else "main"],
    )

    try:
        ensemble = ScreCreditEnsemble(
            candidates,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
            threshold=0.5,
            random_state=RANDOM_SEED,
            validation_size=VALIDATION_SIZE / (1.0 - TEST_SIZE),
        )
        ensemble.fit(X_train, y_train)
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        metrics_050 = binary_classification_metrics(y_test, y_proba, threshold=0.5)
        thresholds = threshold_optimization_table(
            ensemble.validation_y_true_,
            ensemble.validation_proba_,
            false_negative_cost=FN_COST,
            false_positive_cost=FP_COST,
        )
        thresholds.insert(0, "selection_split", "validation")
        thresholds.insert(0, "dataset", dataset)
        best_threshold = float(thresholds.iloc[0]["threshold"])
        metrics_best = binary_classification_metrics(y_test, y_proba, threshold=best_threshold)
        cost_050 = FN_COST * metrics_050["fn"] + FP_COST * metrics_050["fp"]
        cost_best = FN_COST * metrics_best["fn"] + FP_COST * metrics_best["fp"]
        cost_improvement_pct = (cost_050 - cost_best) / cost_050 if cost_050 else 0.0
        metrics_050["expected_cost_fn5_fp1"] = cost_050
        metrics_best["expected_cost_fn5_fp1"] = cost_best
        metrics_best["cost_improvement_pct_vs_050"] = cost_improvement_pct

        prefix = f"{dataset}_scre_credit"
        run_prefix = f"{prefix}_{compact_utc_timestamp()}_{short_run_id(run_id)}"
        candidate_pool = pd.DataFrame({"model": [name for name, _ in candidates]})
        candidate_pool.to_csv(
            TABLES_DIR / f"{prefix}_candidate_pool.csv",
            index=False,
        )
        candidate_pool.to_csv(
            TABLES_DIR / f"{run_prefix}_candidate_pool.csv",
            index=False,
        )
        ensemble.weight_table_.to_csv(TABLES_DIR / f"{prefix}_weights.csv", index=False)
        ensemble.weight_table_.to_csv(TABLES_DIR / f"{run_prefix}_weights.csv", index=False)
        ensemble.validation_results_.to_csv(TABLES_DIR / f"{prefix}_validation_results.csv", index=False)
        ensemble.validation_results_.to_csv(TABLES_DIR / f"{run_prefix}_validation_results.csv", index=False)
        thresholds.to_csv(TABLES_DIR / f"{prefix}_thresholds.csv", index=False)
        thresholds.to_csv(TABLES_DIR / f"{run_prefix}_thresholds.csv", index=False)
        ensemble.three_way_threshold_search_.to_csv(
            TABLES_DIR / f"{prefix}_three_way_threshold_search.csv",
            index=False,
        )
        ensemble.three_way_threshold_search_.to_csv(
            TABLES_DIR / f"{run_prefix}_three_way_threshold_search.csv",
            index=False,
        )
        ci_table = metric_ci_table(y_test, y_proba, n_bootstraps=n_bootstraps)
        ci_table.to_csv(
            TABLES_DIR / f"{prefix}_bootstrap_ci.csv",
            index=False,
        )
        ci_table.to_csv(
            TABLES_DIR / f"{run_prefix}_bootstrap_ci.csv",
            index=False,
        )
        cost_curve = cost_curve_table(y_test, y_proba)
        cost_curve.to_csv(TABLES_DIR / f"{prefix}_cost_curve.csv", index=False)
        cost_curve.to_csv(TABLES_DIR / f"{run_prefix}_cost_curve.csv", index=False)
        review_band = manual_review_band_table(y_test, y_proba, best_threshold)
        review_band.to_csv(
            TABLES_DIR / f"{prefix}_manual_review_band.csv",
            index=False,
        )
        review_band.to_csv(
            TABLES_DIR / f"{run_prefix}_manual_review_band.csv",
            index=False,
        )
        three_way_test = three_way_policy_metrics(
            y_test,
            y_proba,
            ensemble.t_low_,
            ensemble.t_high_,
            fn_cost=FN_COST,
            fp_cost=FP_COST,
            review_cost=ensemble.review_cost,
        )
        pd.DataFrame([three_way_test]).to_csv(
            TABLES_DIR / f"{prefix}_three_way_test_policy.csv",
            index=False,
        )
        pd.DataFrame([three_way_test]).to_csv(
            TABLES_DIR / f"{run_prefix}_three_way_test_policy.csv",
            index=False,
        )
        joblib.dump(ensemble, MODELS_DIR / f"{prefix}.joblib")
        joblib.dump(ensemble, MODELS_DIR / f"{run_prefix}.joblib")

        finish_run(
            run_id,
            metrics={
                "threshold_050": metrics_050,
                "best_cost_threshold": best_threshold,
                "threshold_selection_split": "validation",
                "best_threshold_metrics": metrics_best,
                "cost_improvement_pct_vs_050": cost_improvement_pct,
                "three_way_policy": three_way_test,
            },
            artifacts={
                "candidate_pool": TABLES_DIR / f"{prefix}_candidate_pool.csv",
                "weights": TABLES_DIR / f"{prefix}_weights.csv",
                "thresholds": TABLES_DIR / f"{prefix}_thresholds.csv",
                "three_way_threshold_search": TABLES_DIR / f"{prefix}_three_way_threshold_search.csv",
                "three_way_test_policy": TABLES_DIR / f"{prefix}_three_way_test_policy.csv",
                "model": MODELS_DIR / f"{prefix}.joblib",
                "timestamped_weights": TABLES_DIR / f"{run_prefix}_weights.csv",
                "timestamped_model": MODELS_DIR / f"{run_prefix}.joblib",
            },
        )
        return {
            "dataset": dataset,
            "best_threshold": best_threshold,
            "metrics_050": metrics_050,
            "metrics_best": metrics_best,
            "three_way_test": three_way_test,
        }
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="taiwan")
    parser.add_argument("--n-bootstraps", type=int, default=300)
    parser.add_argument("--quick", action="store_true", help="Skip CatBoost for faster smoke tests.")
    args = parser.parse_args()

    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    rows = []
    for dataset in datasets:
        result = run_one_dataset(dataset, n_bootstraps=args.n_bootstraps, quick=args.quick)
        rows.append(
            {
                "dataset": dataset,
                "best_threshold": result["best_threshold"],
                "cost_improvement_pct_vs_050": result["metrics_best"].get("cost_improvement_pct_vs_050"),
                "t_low": result["three_way_test"]["t_low"],
                "t_high": result["three_way_test"]["t_high"],
                "three_way_expected_policy_cost": result["three_way_test"]["expected_policy_cost"],
                "manual_review_rate": result["three_way_test"]["manual_review_rate"],
                "review_capture_recall": result["three_way_test"]["review_capture_recall"],
                **{f"t050_{k}": v for k, v in result["metrics_050"].items()},
                **{f"best_{k}": v for k, v in result["metrics_best"].items()},
            }
        )

    pd.DataFrame(rows).to_csv(TABLES_DIR / "scre_credit_external_validation_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
