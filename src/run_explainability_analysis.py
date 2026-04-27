"""Generate SHAP, LIME, stability, and faithfulness outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline

from config.settings import RANDOM_SEED
from data_preprocessing import PROJECT_ROOT, TARGET_COLUMN
from explainability import (
    build_lime_explainer,
    compute_shap_values,
    faithfulness_permutation_test,
    ranking_stability_table,
    save_shap_summary_plots,
    top_k_frequency,
    top_shap_features,
)
from model_training import (
    CATEGORICAL_COLUMNS,
    build_preprocessor,
    candidate_models,
    make_pipeline,
    split_features_target,
)


PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "taiwan_model_ready.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "final_model.joblib"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap"
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "experiment_config.yaml"


def make_smotenc_pipeline(model, X: pd.DataFrame, random_state: int):
    """Create the same sampler-first pipeline used in enhanced modelling."""

    categorical_indices = [
        X.columns.get_loc(column)
        for column in CATEGORICAL_COLUMNS
        if column in X.columns
    ]
    sampler = SMOTENC(
        categorical_features=categorical_indices,
        random_state=random_state,
    )
    return ImbalancedPipeline(
        steps=[
            ("sampler", sampler),
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def rebuild_selected_pipeline(model_name: str, strategy: str, X: pd.DataFrame, seed: int):
    """Recreate the selected model for repeated-seed stability analysis."""

    if model_name == "xgboost_tuned_randomized_search":
        metadata_path = TABLES_DIR / "final_model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = candidate_models(random_state=seed)["xgboost"]
        tuned_params = metadata.get("best_params", {})
        model_params = {
            key.replace("model__", ""): value
            for key, value in tuned_params.items()
            if key.startswith("model__")
        }
        model.set_params(**model_params, random_state=seed)
    else:
        model = candidate_models(random_state=seed)[model_name]

    if strategy == "smotenc":
        return make_smotenc_pipeline(model, X, random_state=seed)
    return make_pipeline(model, X)


def save_lime_outputs(final_model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    """Save LIME explanation for the most borderline test customer."""

    lime_explainer = build_lime_explainer(
        X_train,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    probabilities = pd.Series(final_model.predict_proba(X_test)[:, 1], index=X_test.index)
    borderline_idx = probabilities.sub(0.5).abs().idxmin()

    lime_exp = lime_explainer.explain_instance(
        X_test.loc[borderline_idx].to_numpy(),
        lambda values: final_model.predict_proba(
            pd.DataFrame(values, columns=X_train.columns)
        ),
        num_features=10,
    )
    lime_exp.save_to_file(str(SHAP_DIR / "lime_borderline_customer.html"))
    fig = lime_exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "lime_borderline_customer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        lime_exp.as_list(),
        columns=["feature_rule", "lime_weight"],
    ).to_csv(TABLES_DIR / "lime_borderline_customer.csv", index=False)


def plot_faithfulness(faithfulness: pd.DataFrame) -> None:
    """Save bar plot for perturbation probability shifts."""

    if faithfulness.empty:
        return

    ordered = faithfulness.sort_values("mean_probability_shift", ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(ordered["perturbed_feature"], ordered["mean_probability_shift"], color="#6a994e")
    plt.xlabel("Mean absolute probability shift")
    plt.title("Probability Shift After Feature Perturbation")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "probability_shift_after_perturbation.png", dpi=300)
    plt.close()


def plot_top5_frequency(frequency: pd.DataFrame) -> None:
    """Save SHAP top-5 stability frequency plot."""

    if frequency.empty:
        return

    ordered = frequency.sort_values("top_k_rate", ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(ordered["feature"], ordered["top_k_rate"], color="#457b9d")
    plt.xlabel("Share of repeated-seed runs")
    plt.title("SHAP Top-5 Feature Stability")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_stability_top5_frequency.png", dpi=300)
    plt.close()


def load_stability_seed_count(default: int = 50) -> int:
    """Return the configured number of repeated seeds for SHAP stability."""

    if not EXPERIMENT_CONFIG_PATH.exists():
        return default
    with EXPERIMENT_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return int(config.get("shap_stability_seeds", default))


def parse_args() -> argparse.Namespace:
    """Parse explainability and stability experiment options."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stability-seeds",
        type=int,
        default=load_stability_seed_count(),
        help="Number of repeated seeds for SHAP rank-stability analysis.",
    )
    parser.add_argument(
        "--shap-max-rows",
        type=int,
        default=1000,
        help="Maximum test rows used for the main SHAP analysis.",
    )
    parser.add_argument(
        "--stability-max-rows",
        type=int,
        default=500,
        help="Maximum test rows used per repeated-seed stability fit.",
    )
    return parser.parse_args()


def main() -> None:
    """Run explainability and reliability analyses."""

    args = parse_args()
    if args.stability_seeds < 1:
        raise ValueError("--stability-seeds must be at least 1")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split = split_features_target(df, target_column=TARGET_COLUMN, random_state=RANDOM_SEED)
    final_model = joblib.load(MODEL_PATH)

    print("Computing SHAP values for final model...")
    X_shap, shap_values = compute_shap_values(
        final_model,
        split.X_test,
        max_rows=args.shap_max_rows,
    )
    save_shap_summary_plots(X_shap, shap_values, SHAP_DIR)
    shap_top = top_shap_features(shap_values, top_n=20)
    shap_top.to_csv(TABLES_DIR / "shap_top_features.csv", index=False)

    shap_raw_sample = split.X_test.loc[X_shap.index]
    high_risk_position = int(final_model.predict_proba(shap_raw_sample)[:, 1].argmax())
    try:
        import shap

        shap.plots.waterfall(shap_values[high_risk_position], show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_waterfall_high_risk_customer.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Waterfall plot skipped: {exc}")

    print("Saving LIME explanation...")
    save_lime_outputs(final_model, split.X_train, split.X_test)

    raw_top_features = [
        feature
        for feature in shap_top["feature"].tolist()
        if feature in split.X_test.columns
    ]
    fallback_features = [
        "PAY_0",
        "PAY_2",
        "delay_count",
        "severe_delay_count",
        "LIMIT_BAL",
        "payment_to_bill_ratio",
        "utilization_proxy",
    ]
    faithfulness_features = list(dict.fromkeys(raw_top_features + fallback_features))[:10]
    faithfulness = faithfulness_permutation_test(
        final_model,
        split.X_test,
        split.y_test,
        faithfulness_features,
    )
    faithfulness.to_csv(TABLES_DIR / "faithfulness_test_results.csv", index=False)
    plot_faithfulness(faithfulness)

    metadata_path = TABLES_DIR / "final_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = metadata["selected_model"]
    strategy = metadata["imbalance_strategy"]

    print("Running repeated-seed SHAP stability analysis...")
    seeds = list(range(1, args.stability_seeds + 1))
    rankings: dict[int, list[str]] = {}
    ranking_rows: list[dict] = []

    for seed in seeds:
        pipeline = rebuild_selected_pipeline(model_name, strategy, split.X_train, seed)
        pipeline.fit(split.X_train, split.y_train)
        _, seed_shap_values = compute_shap_values(
            pipeline,
            split.X_test,
            max_rows=args.stability_max_rows,
        )
        seed_top = top_shap_features(seed_shap_values, top_n=10)
        rankings[seed] = seed_top["feature"].tolist()
        for _, row in seed_top.iterrows():
            ranking_rows.append(
                {
                    "seed": seed,
                    "rank": int(row["rank"]),
                    "feature": row["feature"],
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                }
            )

    stability_rankings = pd.DataFrame(ranking_rows)
    stability_rankings.to_csv(TABLES_DIR / "shap_stability_rankings.csv", index=False)

    spearman = ranking_stability_table(rankings)
    spearman.to_csv(TABLES_DIR / "shap_stability_spearman.csv", index=False)

    frequency = top_k_frequency(rankings, k=5)
    frequency.to_csv(TABLES_DIR / "shap_top5_frequency.csv", index=False)
    plot_top5_frequency(frequency)

    print(shap_top.head(10).to_string(index=False))
    print("Mean pairwise Spearman rho:", round(float(spearman["spearman_rho"].mean()), 4))


if __name__ == "__main__":
    main()
