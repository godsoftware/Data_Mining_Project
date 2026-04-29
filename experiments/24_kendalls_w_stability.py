"""Create detailed Kendall's W SHAP stability comparison tables."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.config.paths import EXPERIMENT_LOGS_DIR, FIGURES_DIR, TABLES_DIR
from src.explainability.shap_stability import kendalls_w_by_feature_subset
from src.utils.logging_utils import create_experiment_logger


LIN_WANG_ARXIV_URL = "https://arxiv.org/abs/2508.01851"
LIN_WANG_DOI_URL = "https://doi.org/10.3390/risks13120238"

FEATURE_SUBSETS = [
    "all_features",
    "top_5_features",
    "top_10_features",
    "mid_importance_features",
]


def read_required_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_existing_seed_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the current Taiwan and HELOC seed-level SHAP stability artifacts."""

    main = read_required_csv(TABLES_DIR / "shap_stability_seed_results.csv")
    taiwan = main.loc[main["dataset"] == "taiwan"].copy()
    if taiwan.empty:
        raise ValueError("No Taiwan rows found in shap_stability_seed_results.csv.")

    external_heloc_path = TABLES_DIR / "external_validation_heloc_shap_stability_seed_results.csv"
    if external_heloc_path.exists():
        heloc = pd.read_csv(external_heloc_path)
        heloc = heloc.loc[heloc["dataset"] == "heloc"].copy()
        heloc["stability_scope"] = heloc["stability_scope"].replace({"external_reduced": "reduced"})
    else:
        heloc = main.loc[main["dataset"] == "heloc"].copy()
    if heloc.empty:
        raise ValueError("No HELOC rows found in SHAP stability seed artifacts.")
    return taiwan, heloc


def final_operational_model(dataset: str) -> str:
    """Return the final operational winner from the SCRE revision comparison."""

    table = read_required_csv(TABLES_DIR / f"scre_revision_comparison_{dataset}.csv")
    winner = table.sort_values(["operational_rank", "expected_cost", "pr_auc"], ascending=[True, True, False]).iloc[0]
    return str(winner["model"])


def unavailable_rows(
    dataset: str,
    analysis_target: str,
    model: str,
    reason: str,
    proxy_note: str,
) -> pd.DataFrame:
    """Create transparent placeholder rows when direct seed-level SHAP ranks do not exist."""

    rows = []
    for subset in FEATURE_SUBSETS:
        rows.append(
            {
                "dataset": dataset,
                "analysis_target": analysis_target,
                "model": model,
                "model_family": model,
                "stability_scope": "not_available",
                "feature_subset": subset,
                "kendalls_w": np.nan,
                "chi_square": np.nan,
                "degrees_of_freedom": np.nan,
                "p_value": np.nan,
                "n_seeds": np.nan,
                "n_features": np.nan,
                "features_used": "",
                "status": "not_computed_from_current_seed_results",
                "metric_direction": "higher_is_better",
                "source_artifact": "",
                "direct_model_specific": False,
                "limitation_note": reason,
                "proxy_note": proxy_note,
            }
        )
    return pd.DataFrame(rows)


def computed_rows(dataset: str, seed_results: pd.DataFrame, source_artifact: str) -> pd.DataFrame:
    """Compute detailed Kendall's W rows from an existing seed-result table."""

    summary = kendalls_w_by_feature_subset(seed_results)
    summary.insert(1, "analysis_target", "Available LightGBM SHAP stability protocol")
    summary.insert(2, "model", "LightGBM")
    summary["metric_direction"] = "higher_is_better"
    summary["source_artifact"] = source_artifact
    summary["direct_model_specific"] = True
    summary["limitation_note"] = (
        "This is a LightGBM seed-stability protocol. It is directly computed for LightGBM, "
        "not for CatBoost, Scorecard, or SCRE ensemble variants."
    )
    summary["proxy_note"] = ""
    return summary


def build_dataset_table(dataset: str, seed_results: pd.DataFrame, source_artifact: str) -> pd.DataFrame:
    """Build a dataset-specific Kendall's W table with computed and unavailable targets."""

    rows = [computed_rows(dataset, seed_results, source_artifact)]
    winner = final_operational_model(dataset)
    if dataset == "taiwan":
        rows.append(
            unavailable_rows(
                dataset=dataset,
                analysis_target="Final operational model",
                model=winner,
                reason=(
                    f"The final operational model is {winner}; current seed-level SHAP stability artifacts "
                    "were produced for LightGBM only, so a model-specific Kendall's W is not available."
                ),
                proxy_note="Use the LightGBM protocol row only as an explanation-stability context, not as CatBoost evidence.",
            )
        )
        rows.append(
            unavailable_rows(
                dataset=dataset,
                analysis_target="Requested XGBoost/CatBoost comparator",
                model="Best CatBoost",
                reason="No CatBoost or XGBoost seed-level SHAP ranking table exists in the current outputs.",
                proxy_note="A new CatBoost/XGBoost SHAP stability run would be required for a direct comparison.",
            )
        )
    else:
        rows.append(
            unavailable_rows(
                dataset=dataset,
                analysis_target="Final HELOC model",
                model=winner,
                reason=(
                    f"The final HELOC operational model is {winner}; current HELOC seed-level SHAP stability "
                    "artifacts were produced for LightGBM only."
                ),
                proxy_note="This is expected because Scorecard explanations are not SHAP seed-rank artifacts here.",
            )
        )

    for variant in ["SCRE-Pareto", "SCRE-Optimized"]:
        rows.append(
            unavailable_rows(
                dataset=dataset,
                analysis_target=variant,
                model=variant,
                reason=(
                    f"{variant} is a probability ensemble over calibrated base models. The current project "
                    "does not contain seed-level model-agnostic SHAP rankings for this ensemble."
                ),
                proxy_note="Do not report the LightGBM W as direct SCRE ensemble stability.",
            )
        )

    table = pd.concat(rows, ignore_index=True)
    table["kendalls_w"] = pd.to_numeric(table["kendalls_w"], errors="coerce")
    table = table[
        [
            "dataset",
            "analysis_target",
            "model",
            "model_family",
            "stability_scope",
            "feature_subset",
            "kendalls_w",
            "chi_square",
            "degrees_of_freedom",
            "p_value",
            "n_seeds",
            "n_features",
            "features_used",
            "status",
            "metric_direction",
            "source_artifact",
            "direct_model_specific",
            "limitation_note",
            "proxy_note",
        ]
    ]
    return table


def metric_lookup(table: pd.DataFrame, feature_subset: str) -> float:
    """Return a computed LightGBM Kendall's W for a feature subset."""

    match = table.loc[
        (table["analysis_target"] == "Available LightGBM SHAP stability protocol")
        & (table["feature_subset"] == feature_subset)
    ]
    if match.empty:
        return np.nan
    return float(match["kendalls_w"].iloc[0])


def build_literature_comparison(taiwan: pd.DataFrame, heloc: pd.DataFrame) -> pd.DataFrame:
    """Create the required literature comparison rows."""

    taiwan_top5 = metric_lookup(taiwan, "top_5_features")
    taiwan_all = metric_lookup(taiwan, "all_features")
    taiwan_top10 = metric_lookup(taiwan, "top_10_features")
    taiwan_mid = metric_lookup(taiwan, "mid_importance_features")
    heloc_top5 = metric_lookup(heloc, "top_5_features")
    heloc_all = metric_lookup(heloc, "all_features")
    heloc_top10 = metric_lookup(heloc, "top_10_features")
    heloc_mid = metric_lookup(heloc, "mid_importance_features")

    rows = [
        {
            "comparison_row": "Lin & Wang 2025",
            "dataset": "UCI Taiwan credit card default",
            "model": "XGBoost",
            "n_seeds": 100,
            "direct_model_specific": True,
            "all_features_kendalls_w": 0.9775,
            "top5_kendalls_w": 0.93,
            "top10_kendalls_w": np.nan,
            "mid_importance_kendalls_w": 0.34,
            "source": "Lin and Wang 2025, SHAP Stability in Credit Risk Management",
            "source_url": LIN_WANG_ARXIV_URL,
            "doi_url": LIN_WANG_DOI_URL,
            "comparison_note": (
                "Reference uses 100 XGBoost models on UCI Taiwan with mostly raw/one-hot features. "
                "The top-5 W=0.93 and mid-importance W=0.34 values are used as the literature benchmark."
            ),
            "status": "literature_reference",
        },
        {
            "comparison_row": "Our Taiwan final model",
            "dataset": "taiwan",
            "model": final_operational_model("taiwan"),
            "n_seeds": np.nan,
            "direct_model_specific": False,
            "all_features_kendalls_w": np.nan,
            "top5_kendalls_w": np.nan,
            "top10_kendalls_w": np.nan,
            "mid_importance_kendalls_w": np.nan,
            "available_lightgbm_all_features_w": taiwan_all,
            "available_lightgbm_top5_w": taiwan_top5,
            "available_lightgbm_top10_w": taiwan_top10,
            "available_lightgbm_mid_w": taiwan_mid,
            "source": "Current project outputs",
            "source_url": "",
            "doi_url": "",
            "comparison_note": (
                "Final operational model is not the LightGBM stability model. "
                "Direct CatBoost/operational-model SHAP stability is not available in current artifacts."
            ),
            "status": "direct_model_specific_w_not_available",
        },
        {
            "comparison_row": "Our HELOC final model",
            "dataset": "heloc",
            "model": final_operational_model("heloc"),
            "n_seeds": np.nan,
            "direct_model_specific": False,
            "all_features_kendalls_w": np.nan,
            "top5_kendalls_w": np.nan,
            "top10_kendalls_w": np.nan,
            "mid_importance_kendalls_w": np.nan,
            "available_lightgbm_all_features_w": heloc_all,
            "available_lightgbm_top5_w": heloc_top5,
            "available_lightgbm_top10_w": heloc_top10,
            "available_lightgbm_mid_w": heloc_mid,
            "source": "Current project outputs",
            "source_url": "",
            "doi_url": "",
            "comparison_note": (
                "Final HELOC model is not the LightGBM stability model. "
                "Direct Scorecard SHAP seed-rank stability is not available."
            ),
            "status": "direct_model_specific_w_not_available",
        },
        {
            "comparison_row": "Our SCRE-Pareto",
            "dataset": "taiwan_and_heloc",
            "model": "SCRE-Pareto",
            "n_seeds": np.nan,
            "direct_model_specific": False,
            "all_features_kendalls_w": np.nan,
            "top5_kendalls_w": np.nan,
            "top10_kendalls_w": np.nan,
            "mid_importance_kendalls_w": np.nan,
            "available_taiwan_lightgbm_top5_w": taiwan_top5,
            "available_heloc_lightgbm_top5_w": heloc_top5,
            "source": "Current project outputs",
            "source_url": "",
            "doi_url": "",
            "comparison_note": (
                "SCRE-Pareto ensemble-specific SHAP stability was not computed. "
                "Current LightGBM W can only be cited as supporting context."
            ),
            "status": "ensemble_specific_w_not_available",
        },
        {
            "comparison_row": "Our SCRE-Optimized",
            "dataset": "taiwan_and_heloc",
            "model": "SCRE-Optimized",
            "n_seeds": np.nan,
            "direct_model_specific": False,
            "all_features_kendalls_w": np.nan,
            "top5_kendalls_w": np.nan,
            "top10_kendalls_w": np.nan,
            "mid_importance_kendalls_w": np.nan,
            "available_taiwan_lightgbm_top5_w": taiwan_top5,
            "available_heloc_lightgbm_top5_w": heloc_top5,
            "source": "Current project outputs",
            "source_url": "",
            "doi_url": "",
            "comparison_note": (
                "SCRE-Optimized ensemble-specific SHAP stability was not computed. "
                "Current LightGBM W can only be cited as supporting context."
            ),
            "status": "ensemble_specific_w_not_available",
        },
    ]
    return pd.DataFrame(rows)


def plot_comparison(taiwan: pd.DataFrame, heloc: pd.DataFrame, output_path: Path) -> None:
    """Save a Kendall's W comparison plot for computed rows and literature benchmark."""

    plot_rows = [
        {
            "label": "Lin & Wang 2025\nXGBoost top-5",
            "kendalls_w": 0.93,
            "feature_subset": "top_5_features",
        },
        {
            "label": "Our Taiwan\nLightGBM top-5",
            "kendalls_w": metric_lookup(taiwan, "top_5_features"),
            "feature_subset": "top_5_features",
        },
        {
            "label": "Our HELOC\nLightGBM top-5",
            "kendalls_w": metric_lookup(heloc, "top_5_features"),
            "feature_subset": "top_5_features",
        },
        {
            "label": "Lin & Wang 2025\nXGBoost all",
            "kendalls_w": 0.9775,
            "feature_subset": "all_features",
        },
        {
            "label": "Our Taiwan\nLightGBM all",
            "kendalls_w": metric_lookup(taiwan, "all_features"),
            "feature_subset": "all_features",
        },
        {
            "label": "Our HELOC\nLightGBM all",
            "kendalls_w": metric_lookup(heloc, "all_features"),
            "feature_subset": "all_features",
        },
    ]
    plot_table = pd.DataFrame(plot_rows).dropna(subset=["kendalls_w"])
    colors = np.where(plot_table["feature_subset"].eq("top_5_features"), "#2f6f9f", "#7a8f3a")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_table["label"], plot_table["kendalls_w"], color=colors)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Kendall's W")
    ax.set_title("SHAP Rank Stability: Literature Benchmark vs Current Computed Protocols")
    ax.axhline(0.93, color="#444444", linestyle="--", linewidth=1, label="Lin & Wang top-5 W=0.93")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", labelrotation=20)
    for index, value in enumerate(plot_table["kendalls_w"]):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Create all Kendall's W stability outputs."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = create_experiment_logger("kendalls_w_stability", EXPERIMENT_LOGS_DIR / "24_kendalls_w_stability.log")

    taiwan_seed_results, heloc_seed_results = load_existing_seed_results()
    taiwan = build_dataset_table("taiwan", taiwan_seed_results, "outputs/tables/shap_stability_seed_results.csv")
    heloc = build_dataset_table(
        "heloc",
        heloc_seed_results,
        "outputs/tables/external_validation_heloc_shap_stability_seed_results.csv",
    )
    literature = build_literature_comparison(taiwan, heloc)

    taiwan_path = TABLES_DIR / "kendalls_w_stability_taiwan.csv"
    heloc_path = TABLES_DIR / "kendalls_w_stability_heloc.csv"
    literature_path = TABLES_DIR / "shap_stability_literature_comparison.csv"
    figure_path = FIGURES_DIR / "kendalls_w_comparison.png"

    taiwan.to_csv(taiwan_path, index=False)
    heloc.to_csv(heloc_path, index=False)
    literature.to_csv(literature_path, index=False)
    plot_comparison(taiwan, heloc, figure_path)

    logger.info("Saved %s", taiwan_path)
    logger.info("Saved %s", heloc_path)
    logger.info("Saved %s", literature_path)
    logger.info("Saved %s", figure_path)
    print(
        taiwan.loc[taiwan["status"].eq("computed_from_existing_seed_results")][
            ["dataset", "analysis_target", "feature_subset", "kendalls_w", "chi_square", "p_value", "n_seeds", "n_features"]
        ].to_string(index=False)
    )
    print(
        heloc.loc[heloc["status"].eq("computed_from_existing_seed_results")][
            ["dataset", "analysis_target", "feature_subset", "kendalls_w", "chi_square", "p_value", "n_seeds", "n_features"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
