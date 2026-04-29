"""Run Prompt 15 SHAP and LIME explainability outputs."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.settings import RANDOM_SEED
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from experiment_registry import finish_run, start_run
from src.config.paths import CALIBRATED_MODELS_DIR, LIME_DIR, SHAP_DIR, TABLES_DIR, TAIWAN_MODEL_READY
from src.explainability.lime_analysis import explain_lime_cases, shap_lime_agreement_table
from src.explainability.shap_analysis import (
    ShapComputation,
    compute_shap_analysis,
    local_shap_feature_table,
    save_dependence_plots,
    save_global_shap_plots,
    save_local_waterfall_plots,
    shap_top_features_table,
)


DATASET = "taiwan"
CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]
DEFAULT_MODEL_NAME = "lightgbm"

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    """Parse explainability runtime options."""

    parser = argparse.ArgumentParser(description="Generate SHAP/LIME explainability outputs.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Calibrated-model basename to explain.")
    parser.add_argument("--sample-size", type=int, default=500, help="Rows used for global SHAP plots.")
    parser.add_argument("--background-size", type=int, default=200, help="Rows used as SHAP background.")
    parser.add_argument("--lime-samples", type=int, default=1000, help="Perturbed samples per LIME explanation.")
    return parser.parse_args()


def model_path(model_name: str) -> Path:
    """Return the uncalibrated calibrated-model artifact path used for explainability."""

    return CALIBRATED_MODELS_DIR / DATASET / f"{model_name}_uncalibrated.joblib"


def selected_threshold(model_name: str, default: float = 0.5) -> float:
    """Load the validation-selected threshold for the explained model."""

    result_path = TABLES_DIR / "scre_credit_results_taiwan.csv"
    if not result_path.exists():
        return default
    results = pd.read_csv(result_path)
    match = results.loc[
        (results["model"] == model_name)
        & (results["split"] == "validation")
        & (results["model_role"] == "base_model")
    ]
    if match.empty:
        return default
    return float(match.iloc[0]["threshold"])


def select_local_cases(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Select TP/TN/FP/FN and borderline cases from test predictions."""

    y_true_array = y_true.to_numpy(dtype=int)
    y_pred = (y_proba >= threshold).astype(int)
    candidates = pd.DataFrame(
        {
            "row_position": np.arange(len(y_true_array)),
            "row_id": y_true.index.to_numpy(),
            "y_true": y_true_array,
            "y_pred": y_pred,
            "predicted_probability": y_proba,
            "threshold": threshold,
            "distance_to_threshold": np.abs(y_proba - threshold),
        }
    )
    definitions = {
        "true_positive": (1, 1),
        "true_negative": (0, 0),
        "false_positive": (0, 1),
        "false_negative": (1, 0),
    }

    rows = []
    used_positions: set[int] = set()
    for case_type, (target, prediction) in definitions.items():
        subset = candidates.loc[
            (candidates["y_true"] == target) & (candidates["y_pred"] == prediction)
        ].copy()
        if subset.empty:
            continue
        if prediction == 1:
            selected = subset.sort_values("predicted_probability", ascending=False).iloc[0]
        else:
            selected = subset.sort_values("predicted_probability", ascending=True).iloc[0]
        rows.append({**selected.to_dict(), "case_type": case_type})
        used_positions.add(int(selected["row_position"]))

    borderline = candidates.loc[~candidates["row_position"].isin(used_positions)].sort_values(
        "distance_to_threshold"
    ).iloc[0]
    rows.append({**borderline.to_dict(), "case_type": "borderline_manual_review"})
    return pd.DataFrame(rows)


def main() -> None:
    """Generate Prompt 15 explainability artifacts."""

    args = parse_args()
    run_id = start_run(
        "explainability_shap_lime",
        dataset=DATASET,
        params={
            "model": args.model,
            "sample_size": args.sample_size,
            "background_size": args.background_size,
            "lime_samples": args.lime_samples,
        },
        tags=["phase-13", "explainability"],
    )
    try:
        path = model_path(args.model)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        SHAP_DIR.mkdir(parents=True, exist_ok=True)
        LIME_DIR.mkdir(parents=True, exist_ok=True)
        (SHAP_DIR / "dependence_top_features").mkdir(parents=True, exist_ok=True)
        (SHAP_DIR / "local_waterfall").mkdir(parents=True, exist_ok=True)
        (LIME_DIR / "local_explanations").mkdir(parents=True, exist_ok=True)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(TAIWAN_MODEL_READY)
        split = stratified_train_validation_test_split(df, target_column=TARGET_COLUMN)
        X_train = split.train.drop(columns=[TARGET_COLUMN])
        X_test = split.test.drop(columns=[TARGET_COLUMN])
        y_test = split.test[TARGET_COLUMN]
        pipeline = joblib.load(path)
        threshold = selected_threshold(args.model)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        case_metadata = select_local_cases(y_test, y_proba, threshold)
        X_cases = X_test.iloc[case_metadata["row_position"].astype(int).to_numpy()].copy()

        X_global = X_test.sample(min(args.sample_size, len(X_test)), random_state=RANDOM_SEED)
        X_explain = pd.concat([X_global, X_cases], axis=0).drop_duplicates()
        global_positions = X_explain.index.isin(X_global.index)
        local_positions = [X_explain.index.get_loc(index) for index in X_cases.index]

        shap_computation = compute_shap_analysis(
            pipeline=pipeline,
            X_background=X_train,
            X_explain=X_explain,
            model_name=args.model,
            max_background_rows=args.background_size,
            random_seed=RANDOM_SEED,
        )

        # Global plots should not be dominated by the deliberately selected local cases.
        global_computation = ShapComputation(
            X_transformed=shap_computation.X_transformed.loc[global_positions],
            shap_values=shap_computation.shap_values[global_positions],
            explainer_type=shap_computation.explainer_type,
            model_name=shap_computation.model_name,
        )
        save_global_shap_plots(global_computation, SHAP_DIR)

        top_features = shap_top_features_table(
            global_computation,
            raw_features=list(X_train.columns),
            top_n=20,
        )
        top_features.insert(0, "dataset", DATASET)
        top_features.insert(1, "model", args.model)
        top_features.to_csv(TABLES_DIR / "shap_top_features.csv", index=False)

        save_dependence_plots(
            global_computation,
            top_features["feature"].head(5).tolist(),
            SHAP_DIR / "dependence_top_features",
        )

        local_computation = ShapComputation(
            X_transformed=shap_computation.X_transformed.iloc[local_positions],
            shap_values=shap_computation.shap_values[local_positions],
            explainer_type=shap_computation.explainer_type,
            model_name=shap_computation.model_name,
        )
        save_local_waterfall_plots(
            local_computation,
            case_metadata,
            SHAP_DIR / "local_waterfall",
        )
        shap_local = local_shap_feature_table(
            local_computation,
            case_metadata,
            raw_features=list(X_train.columns),
            top_k=10,
        )

        lime_local = explain_lime_cases(
            pipeline=pipeline,
            X_cases=X_cases,
            case_metadata=case_metadata,
            X_train=X_train,
            output_dir=LIME_DIR / "local_explanations",
            categorical_columns=CATEGORICAL_COLUMNS,
            num_features=10,
            num_samples=args.lime_samples,
            random_seed=RANDOM_SEED,
        )
        lime_local.insert(0, "dataset", DATASET)
        lime_local.insert(1, "model", args.model)
        lime_local.to_csv(TABLES_DIR / "lime_local_features.csv", index=False)

        agreement = shap_lime_agreement_table(shap_local, lime_local, top_k=10)
        agreement.insert(0, "dataset", DATASET)
        agreement.insert(1, "model", args.model)
        agreement.to_csv(TABLES_DIR / "shap_lime_agreement.csv", index=False)

        metadata = pd.DataFrame(
            [
                {
                    "dataset": DATASET,
                    "model": args.model,
                    "model_path": str(path),
                    "explainer_type": shap_computation.explainer_type,
                    "tree_explainer_used": shap_computation.explainer_type == "tree",
                    "sample_size": int(len(X_global)),
                    "background_size": int(min(args.background_size, len(X_train))),
                    "local_case_count": int(len(case_metadata)),
                    "lime_samples": int(args.lime_samples),
                    "fallback_note": (
                        "TreeExplainer succeeded."
                        if shap_computation.explainer_type == "tree"
                        else "Permutation SHAP fallback used; interpret runtime and variance accordingly."
                    ),
                }
            ]
        )
        metadata.to_csv(TABLES_DIR / "shap_explainer_metadata.csv", index=False)

        print(top_features.head(10).to_string(index=False))
        print(agreement.to_string(index=False))
        finish_run(
            run_id,
            metrics={
                "top_feature_count": int(len(top_features)),
                "local_case_count": int(len(case_metadata)),
                "shap_lime_agreement_rows": int(len(agreement)),
                "tree_explainer_used": bool(shap_computation.explainer_type == "tree"),
            },
            artifacts={
                "shap_global_bar": SHAP_DIR / "global_bar.png",
                "shap_global_beeswarm": SHAP_DIR / "global_beeswarm.png",
                "shap_dependence_top_features": SHAP_DIR / "dependence_top_features",
                "shap_local_waterfall": SHAP_DIR / "local_waterfall",
                "lime_local_explanations": LIME_DIR / "local_explanations",
                "shap_top_features": TABLES_DIR / "shap_top_features.csv",
                "lime_local_features": TABLES_DIR / "lime_local_features.csv",
                "shap_lime_agreement": TABLES_DIR / "shap_lime_agreement.csv",
                "shap_explainer_metadata": TABLES_DIR / "shap_explainer_metadata.csv",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
