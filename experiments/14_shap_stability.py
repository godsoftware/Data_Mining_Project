"""Run Prompt 16 SHAP stability experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import RANDOM_SEED, load_experiment_config
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET
from src.config.paths import FIGURES_DIR, HELOC_MODEL_READY, TABLES_DIR, TAIWAN_MODEL_READY
from src.explainability.shap_stability import (
    StabilityDatasetConfig,
    feature_frequency_table,
    kendalls_w_summary_table,
    pairwise_rank_correlations,
    plot_rank_stability_heatmap,
    plot_top_feature_frequency,
    run_seed_stability,
)


HELOC_REDUCED_FEATURES = [
    "delinquency_intensity",
    "trade_activity_ratio",
    "recent_inquiry_pressure",
    "revolving_burden_proxy",
    "installment_burden_proxy",
    "credit_history_length_proxy",
    "negative_trade_signal",
    "high_utilization_signal",
]


def parse_args() -> argparse.Namespace:
    """Parse stability-runtime controls."""

    config = load_experiment_config()
    default_seeds = int(config.get("shap_stability_seeds", 50))
    parser = argparse.ArgumentParser(description="Run SHAP rank-stability experiments.")
    parser.add_argument("--n-seeds", type=int, default=default_seeds)
    parser.add_argument("--explain-rows", type=int, default=400)
    return parser.parse_args()


def seed_list(n_seeds: int, base_seed: int = RANDOM_SEED) -> list[int]:
    """Return deterministic model seeds for stability runs."""

    rng = np.random.default_rng(base_seed)
    return [int(seed) for seed in rng.choice(np.arange(1, 1_000_000), size=n_seeds, replace=False)]


def main() -> None:
    """Run Taiwan full and HELOC reduced SHAP stability."""

    args = parse_args()
    run_id = start_run(
        "shap_stability",
        dataset="both",
        params={
            "n_seeds": args.n_seeds,
            "explain_rows": args.explain_rows,
            "random_seed": RANDOM_SEED,
            "model_family": "lightgbm",
        },
        tags=["phase-14", "shap-stability"],
    )
    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        seeds = seed_list(args.n_seeds)

        taiwan = pd.read_csv(TAIWAN_MODEL_READY)
        heloc = pd.read_csv(HELOC_MODEL_READY)
        dataset_runs = [
            (
                taiwan,
                StabilityDatasetConfig(
                    dataset="taiwan",
                    scope="full",
                    target=TARGET_COLUMN,
                    categorical_columns=["SEX", "EDUCATION", "MARRIAGE"],
                ),
            ),
            (
                heloc,
                StabilityDatasetConfig(
                    dataset="heloc",
                    scope="reduced",
                    target=HELOC_TARGET,
                    categorical_columns=HELOC_CATEGORICAL_COLUMNS,
                    feature_subset=HELOC_REDUCED_FEATURES,
                ),
            ),
        ]

        seed_results = []
        for df, config in dataset_runs:
            seed_results.append(
                run_seed_stability(
                    df=df,
                    config=config,
                    seeds=seeds,
                    explain_rows=args.explain_rows,
                    base_random_seed=RANDOM_SEED,
                )
            )

        seed_results_table = pd.concat(seed_results, ignore_index=True)
        correlations = pairwise_rank_correlations(seed_results_table)
        kendalls_w_summary = kendalls_w_summary_table(seed_results_table)
        frequency = feature_frequency_table(seed_results_table)

        seed_results_path = TABLES_DIR / "shap_stability_seed_results.csv"
        correlations_path = TABLES_DIR / "shap_rank_correlations.csv"
        kendalls_w_path = TABLES_DIR / "shap_kendalls_w.csv"
        frequency_path = TABLES_DIR / "shap_topk_frequency.csv"
        heatmap_path = FIGURES_DIR / "shap_rank_stability_heatmap.png"
        frequency_figure_path = FIGURES_DIR / "shap_top_feature_frequency.png"

        seed_results_table.to_csv(seed_results_path, index=False)
        correlations.to_csv(correlations_path, index=False)
        kendalls_w_summary.to_csv(kendalls_w_path, index=False)
        frequency.to_csv(frequency_path, index=False)
        plot_rank_stability_heatmap(correlations, heatmap_path)
        plot_top_feature_frequency(frequency, frequency_figure_path)

        print(
            correlations.groupby(["dataset", "stability_scope"])[
                ["spearman_rho", "kendall_tau", "kendalls_w", "top5_overlap_rate", "top10_overlap_rate"]
            ]
            .mean()
            .reset_index()
            .to_string(index=False)
        )
        print(
            frequency.groupby(["dataset", "stability_scope"])
            .head(10)[
                [
                    "dataset",
                    "stability_scope",
                    "feature",
                    "mean_rank",
                    "rank_std",
                    "top5_frequency",
                    "top10_frequency",
                ]
            ]
            .to_string(index=False)
        )

        finish_run(
            run_id,
            metrics={
                "n_seeds": int(args.n_seeds),
                "seed_result_rows": int(len(seed_results_table)),
                "pairwise_correlation_rows": int(len(correlations)),
                "taiwan_mean_spearman": float(
                    correlations.loc[correlations["dataset"] == "taiwan", "spearman_rho"].mean()
                ),
                "heloc_mean_spearman": float(
                    correlations.loc[correlations["dataset"] == "heloc", "spearman_rho"].mean()
                ),
                "taiwan_kendalls_w": float(
                    kendalls_w_summary.loc[kendalls_w_summary["dataset"] == "taiwan", "kendalls_w"].iloc[0]
                ),
                "heloc_kendalls_w": float(
                    kendalls_w_summary.loc[kendalls_w_summary["dataset"] == "heloc", "kendalls_w"].iloc[0]
                ),
            },
            artifacts={
                "shap_stability_seed_results": seed_results_path,
                "shap_rank_correlations": correlations_path,
                "shap_kendalls_w": kendalls_w_path,
                "shap_topk_frequency": frequency_path,
                "shap_rank_stability_heatmap": heatmap_path,
                "shap_top_feature_frequency": frequency_figure_path,
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
