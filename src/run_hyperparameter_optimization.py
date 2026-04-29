"""Phase 8 Optuna hyperparameter optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from config.settings import FN_COST, FP_COST, RANDOM_SEED, load_experiment_config
from data.split_leakage import stratified_train_validation_test_split
from data_preprocessing import TARGET_COLUMN
from evaluation import binary_classification_metrics
from experiment_registry import finish_run, start_run
from heloc_preprocessing import HELOC_CATEGORICAL_COLUMNS, HELOC_TARGET
from scre_credit import make_model_pipeline
from src.config.paths import BOOSTING_MODELS_DIR, FIGURES_DIR, HELOC_MODEL_READY, TABLES_DIR, TAIWAN_MODEL_READY

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None


EXPERIMENT_CONFIG = load_experiment_config()
OPTUNA_TRIALS_TAIWAN = int(EXPERIMENT_CONFIG.get("optuna_trials_taiwan", 100))
OPTUNA_TRIALS_HELOC = int(EXPERIMENT_CONFIG.get("optuna_trials_heloc", 50))
CV_N_SPLITS = int(EXPERIMENT_CONFIG.get("n_splits", 5))
CV_N_REPEATS = int(EXPERIMENT_CONFIG.get("n_repeats", 3))


DATASET_CONFIGS = {
    "taiwan": {
        "path": TAIWAN_MODEL_READY,
        "target": TARGET_COLUMN,
        "categorical_columns": ["SEX", "EDUCATION", "MARRIAGE"],
        "default_trials": OPTUNA_TRIALS_TAIWAN,
    },
    "heloc": {
        "path": HELOC_MODEL_READY,
        "target": HELOC_TARGET,
        "categorical_columns": HELOC_CATEGORICAL_COLUMNS,
        "default_trials": OPTUNA_TRIALS_HELOC,
    },
}


def _available_model_types(requested: list[str]) -> list[str]:
    """Return requested model types that have installed dependencies."""

    available = []
    for model_type in requested:
        if model_type == "xgboost" and XGBClassifier is not None:
            available.append(model_type)
        elif model_type == "lightgbm" and LGBMClassifier is not None:
            available.append(model_type)
        elif model_type == "catboost" and CatBoostClassifier is not None:
            available.append(model_type)
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ImportError(f"Missing dependencies for requested model types: {missing}")
    return available


def suggest_model(trial: optuna.Trial, model_types: list[str], random_state: int = RANDOM_SEED):
    """Sample a model family and hyperparameters."""

    model_type = trial.suggest_categorical("model_type", model_types)

    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 150, 550, step=50),
            max_depth=trial.suggest_int("xgb_max_depth", 2, 6),
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.65, 1.0),
            min_child_weight=trial.suggest_int("xgb_min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-3, 20.0, log=True),
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("lgbm_n_estimators", 150, 650, step=50),
            learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("lgbm_num_leaves", 15, 95),
            max_depth=trial.suggest_int("lgbm_max_depth", 3, 10),
            min_child_samples=trial.suggest_int("lgbm_min_child_samples", 10, 140),
            subsample=trial.suggest_float("lgbm_subsample", 0.65, 1.0),
            colsample_bytree=trial.suggest_float("lgbm_colsample_bytree", 0.65, 1.0),
            reg_alpha=trial.suggest_float("lgbm_reg_alpha", 1e-8, 5.0, log=True),
            reg_lambda=trial.suggest_float("lgbm_reg_lambda", 1e-3, 20.0, log=True),
            class_weight=trial.suggest_categorical("lgbm_class_weight", [None, "balanced"]),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if model_type == "catboost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("cat_iterations", 150, 450, step=50),
            depth=trial.suggest_int("cat_depth", 3, 7),
            learning_rate=trial.suggest_float("cat_learning_rate", 0.01, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("cat_l2_leaf_reg", 0.5, 20.0, log=True),
            random_strength=trial.suggest_float("cat_random_strength", 0.1, 5.0, log=True),
            bagging_temperature=trial.suggest_float("cat_bagging_temperature", 0.0, 2.0),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _fold_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    """Compute the Phase 8 objective and secondary metrics for one fold."""

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "fn_weighted_cost": float(FN_COST * fn + FP_COST * fp),
        "fn": float(fn),
        "fp": float(fp),
        "tp": float(tp),
        "tn": float(tn),
    }


def evaluate_trial_cv(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    model_types: list[str],
    n_splits: int,
    n_repeats: int,
    random_state: int = RANDOM_SEED,
) -> dict[str, float]:
    """Evaluate one sampled model with repeated stratified CV."""

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    fold_rows = []
    for train_idx, validation_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_validation = X.iloc[validation_idx]
        y_validation = y.iloc[validation_idx]

        estimator = suggest_model(trial, model_types=model_types, random_state=random_state)
        pipeline = make_model_pipeline(estimator, X_train, categorical_columns)
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_validation)[:, 1]
        fold_rows.append(_fold_metrics(y_validation, y_proba))

    summary = pd.DataFrame(fold_rows).mean().to_dict()
    summary["pr_auc_std"] = float(pd.DataFrame(fold_rows)["pr_auc"].std())
    return {key: float(value) for key, value in summary.items()}


def objective_factory(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    model_types: list[str],
    n_splits: int,
    n_repeats: int,
):
    """Build an Optuna objective that maximizes mean CV PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        metrics = evaluate_trial_cv(
            trial,
            X,
            y,
            categorical_columns=categorical_columns,
            model_types=model_types,
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        return metrics["pr_auc"]

    return objective


def _trial_rows(study: optuna.Study, dataset: str, n_splits: int, n_repeats: int) -> list[dict]:
    """Flatten Optuna trial data for CSV output."""

    rows = []
    for trial in study.trials:
        row = {
            "dataset": dataset,
            "study_name": study.study_name,
            "trial_number": trial.number,
            "state": str(trial.state.name),
            "objective": "mean_cv_pr_auc",
            "primary_metric_pr_auc": trial.value,
            "cv_n_splits": n_splits,
            "cv_n_repeats": n_repeats,
            "cv_total_folds": n_splits * n_repeats,
        }
        row.update({f"metric_{key}": value for key, value in trial.user_attrs.items()})
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    return rows


def _params_from_trial(trial: optuna.trial.FrozenTrial) -> dict:
    """Convert a best trial's parameters to a JSON-safe dict."""

    return {key: value for key, value in trial.params.items()}


def build_model_from_best_params(params: dict, random_state: int = RANDOM_SEED):
    """Instantiate the best model from saved Optuna params."""

    model_type = params["model_type"]
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=int(params["xgb_n_estimators"]),
            max_depth=int(params["xgb_max_depth"]),
            learning_rate=float(params["xgb_learning_rate"]),
            subsample=float(params["xgb_subsample"]),
            colsample_bytree=float(params["xgb_colsample_bytree"]),
            min_child_weight=int(params["xgb_min_child_weight"]),
            reg_alpha=float(params["xgb_reg_alpha"]),
            reg_lambda=float(params["xgb_reg_lambda"]),
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=int(params["lgbm_n_estimators"]),
            learning_rate=float(params["lgbm_learning_rate"]),
            num_leaves=int(params["lgbm_num_leaves"]),
            max_depth=int(params["lgbm_max_depth"]),
            min_child_samples=int(params["lgbm_min_child_samples"]),
            subsample=float(params["lgbm_subsample"]),
            colsample_bytree=float(params["lgbm_colsample_bytree"]),
            reg_alpha=float(params["lgbm_reg_alpha"]),
            reg_lambda=float(params["lgbm_reg_lambda"]),
            class_weight=params.get("lgbm_class_weight"),
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if model_type == "catboost":
        return CatBoostClassifier(
            iterations=int(params["cat_iterations"]),
            depth=int(params["cat_depth"]),
            learning_rate=float(params["cat_learning_rate"]),
            l2_leaf_reg=float(params["cat_l2_leaf_reg"]),
            random_strength=float(params["cat_random_strength"]),
            bagging_temperature=float(params["cat_bagging_temperature"]),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported best model_type: {model_type}")


def fit_best_on_train_evaluate_validation(
    study: optuna.Study,
    dataset: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_columns: list[str],
) -> dict:
    """Fit the best HPO model on train split and evaluate validation split."""

    best_params = _params_from_trial(study.best_trial)
    estimator = build_model_from_best_params(best_params)
    pipeline = make_model_pipeline(estimator, X_train, categorical_columns)
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_validation)[:, 1]
    metrics = binary_classification_metrics(y_validation, y_proba, threshold=0.5)

    model_dir = BOOSTING_MODELS_DIR / "tuned"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{dataset}_optuna_best_{best_params['model_type']}.joblib"
    joblib.dump(pipeline, model_path)

    return {
        "dataset": dataset,
        "best_model_type": best_params["model_type"],
        "best_trial_number": int(study.best_trial.number),
        "best_cv_pr_auc": float(study.best_value),
        "best_cv_roc_auc": float(study.best_trial.user_attrs.get("roc_auc", np.nan)),
        "best_cv_recall": float(study.best_trial.user_attrs.get("recall", np.nan)),
        "best_cv_brier_score": float(study.best_trial.user_attrs.get("brier_score", np.nan)),
        "best_cv_fn_weighted_cost": float(study.best_trial.user_attrs.get("fn_weighted_cost", np.nan)),
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "validation_recall": float(metrics["recall"]),
        "validation_brier_score": float(metrics["brier_score"]),
        "validation_fn_weighted_cost": float(FN_COST * metrics["fn"] + FP_COST * metrics["fp"]),
        "best_params_json": json.dumps(best_params, sort_keys=True),
        "model_path": str(model_path),
    }


def plot_optimization_history(studies: dict[str, optuna.Study], output_path: Path) -> None:
    """Save a composite optimization-history plot."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for dataset, study in studies.items():
        completed = [trial for trial in study.trials if trial.value is not None]
        numbers = [trial.number for trial in completed]
        values = [trial.value for trial in completed]
        best_values = np.maximum.accumulate(values) if values else []
        ax.plot(numbers, values, marker="o", linestyle="", alpha=0.35, label=f"{dataset} trials")
        ax.plot(numbers, best_values, linewidth=2, label=f"{dataset} best")
    ax.set_title("Optuna Optimization History")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean CV PR-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_param_importance(studies: dict[str, optuna.Study], output_path: Path) -> None:
    """Save a composite parameter-importance plot."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_studies = len(studies)
    fig, axes = plt.subplots(n_studies, 1, figsize=(9, 4.2 * n_studies))
    if n_studies == 1:
        axes = [axes]

    for ax, (dataset, study) in zip(axes, studies.items()):
        try:
            importances = get_param_importances(study)
        except Exception:
            importances = {}
        top_items = list(importances.items())[:12]
        if not top_items:
            ax.text(0.5, 0.5, "No importance data", ha="center", va="center")
            ax.set_axis_off()
            continue
        labels = [item[0] for item in top_items][::-1]
        values = [item[1] for item in top_items][::-1]
        ax.barh(labels, values, color="#4f7cac")
        ax.set_title(f"{dataset} parameter importance")
        ax.set_xlabel("Importance")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_optuna_run_status(search_results: pd.DataFrame, best_params: pd.DataFrame) -> Path:
    """Write a compact completion/audit table for the latest Optuna artifacts."""

    rows = []
    minimum_trials = {"taiwan": OPTUNA_TRIALS_TAIWAN, "heloc": OPTUNA_TRIALS_HELOC}
    for dataset, best_row in best_params.sort_values("dataset").groupby("dataset").first().iterrows():
        dataset_trials = search_results.loc[search_results["dataset"] == dataset].copy()
        completed = dataset_trials.loc[dataset_trials["state"] == "COMPLETE"]
        requested_trials = int(best_row.get("n_trials", len(completed)))
        completed_trials = int(len(completed))
        model_types = sorted(
            str(value)
            for value in completed.get("param_model_type", pd.Series(dtype=object)).dropna().unique()
        )
        test_set_value = best_row["test_set_used"]
        test_set_used = (
            str(test_set_value).strip().lower() == "true"
            if isinstance(test_set_value, str)
            else bool(test_set_value)
        )
        rows.append(
            {
                "dataset": dataset,
                "status": "completed" if completed_trials >= requested_trials else "partial",
                "requested_trials": requested_trials,
                "minimum_prompt_trials": int(minimum_trials.get(dataset, requested_trials)),
                "completed_trials": completed_trials,
                "completed_trials_in_current_artifacts": completed_trials,
                "cv_n_splits": int(best_row["cv_n_splits"]),
                "cv_n_repeats": int(best_row["cv_n_repeats"]),
                "cv_total_folds": int(best_row["cv_total_folds"]),
                "objective_primary": best_row["objective_primary"],
                "model_search_space_completed": ", ".join(model_types),
                "best_model_type": best_row["best_model_type"],
                "best_trial_number": int(best_row["best_trial_number"]),
                "best_cv_pr_auc": float(best_row["best_cv_pr_auc"]),
                "validation_pr_auc": float(best_row["validation_pr_auc"]),
                "test_set_used": test_set_used,
                "study_path": best_row["study_path"],
                "model_path": best_row["model_path"],
            }
        )

    output_path = TABLES_DIR / "optuna_run_status.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def run_one_dataset(
    dataset: str,
    n_trials: int,
    model_types: list[str],
    n_splits: int,
    n_repeats: int,
) -> tuple[optuna.Study, list[dict], dict]:
    """Run one dataset's Optuna study."""

    config = DATASET_CONFIGS[dataset]
    df = pd.read_csv(config["path"])
    split = stratified_train_validation_test_split(df, target_column=config["target"])
    X_train = split.train.drop(columns=[config["target"]])
    y_train = split.train[config["target"]].reset_index(drop=True)
    X_validation = split.validation.drop(columns=[config["target"]])
    y_validation = split.validation[config["target"]].reset_index(drop=True)

    model_types = _available_model_types(model_types)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"{dataset}_pr_auc_optuna",
    )
    study.optimize(
        objective_factory(
            X_train.reset_index(drop=True),
            y_train,
            categorical_columns=config["categorical_columns"],
            model_types=model_types,
            n_splits=n_splits,
            n_repeats=n_repeats,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    study_dir = BOOSTING_MODELS_DIR / "tuned"
    study_dir.mkdir(parents=True, exist_ok=True)
    study_path = study_dir / f"{dataset}_optuna_study.joblib"
    joblib.dump(study, study_path)

    trial_rows = _trial_rows(study, dataset, n_splits=n_splits, n_repeats=n_repeats)
    best_row = fit_best_on_train_evaluate_validation(
        study,
        dataset,
        X_train,
        y_train,
        X_validation,
        y_validation,
        categorical_columns=config["categorical_columns"],
    )
    best_row.update(
        {
            "n_trials": n_trials,
            "cv_n_splits": n_splits,
            "cv_n_repeats": n_repeats,
            "cv_total_folds": n_splits * n_repeats,
            "objective_primary": "PR-AUC",
            "objective_secondary": "ROC-AUC, Recall, Brier score, FN-weighted cost",
            "test_set_used": False,
            "study_path": str(study_path),
        }
    )
    return study, trial_rows, best_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["taiwan", "heloc", "both"], default="both")
    parser.add_argument("--taiwan-trials", type=int, default=OPTUNA_TRIALS_TAIWAN)
    parser.add_argument("--heloc-trials", type=int, default=OPTUNA_TRIALS_HELOC)
    parser.add_argument("--n-splits", type=int, default=CV_N_SPLITS)
    parser.add_argument("--n-repeats", type=int, default=CV_N_REPEATS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm", "catboost"],
        choices=["xgboost", "lightgbm", "catboost"],
    )
    args = parser.parse_args()

    datasets = ["taiwan", "heloc"] if args.dataset == "both" else [args.dataset]
    run_id = start_run(
        "hyperparameter_optimization",
        dataset=args.dataset,
        params={
            "taiwan_trials": args.taiwan_trials,
            "heloc_trials": args.heloc_trials,
            "n_splits": args.n_splits,
            "n_repeats": args.n_repeats,
            "models": args.models,
            "objective": "PR-AUC",
        },
        tags=["phase-8", "optuna"],
    )

    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        studies = {}
        all_trial_rows = []
        best_rows = []

        if args.dataset != "both":
            existing_trials_path = TABLES_DIR / "optuna_trials.csv"
            existing_best_path = TABLES_DIR / "best_params.csv"
            if existing_trials_path.exists():
                existing_trials = pd.read_csv(existing_trials_path)
                existing_trials = existing_trials.loc[~existing_trials["dataset"].isin(datasets)]
                all_trial_rows.extend(existing_trials.to_dict(orient="records"))
            if existing_best_path.exists():
                existing_best = pd.read_csv(existing_best_path)
                existing_best = existing_best.loc[~existing_best["dataset"].isin(datasets)]
                best_rows.extend(existing_best.to_dict(orient="records"))

        for dataset in datasets:
            n_trials = args.taiwan_trials if dataset == "taiwan" else args.heloc_trials
            print(
                f"Running Optuna for {dataset}: {n_trials} trials, "
                f"{args.n_splits}x{args.n_repeats} repeated stratified CV"
            )
            study, trial_rows, best_row = run_one_dataset(
                dataset,
                n_trials=n_trials,
                model_types=args.models,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
            )
            studies[dataset] = study
            all_trial_rows.extend(trial_rows)
            best_rows.append(best_row)
            pd.DataFrame(all_trial_rows).to_csv(TABLES_DIR / "hyperparameter_search_results.csv", index=False)
            pd.DataFrame(all_trial_rows).to_csv(TABLES_DIR / "optuna_trials.csv", index=False)
            pd.DataFrame(best_rows).to_csv(TABLES_DIR / "best_params.csv", index=False)

        search_results = pd.DataFrame(all_trial_rows)
        best_params = pd.DataFrame(best_rows)
        search_results.to_csv(TABLES_DIR / "hyperparameter_search_results.csv", index=False)
        search_results.to_csv(TABLES_DIR / "optuna_trials.csv", index=False)
        best_params.to_csv(TABLES_DIR / "best_params.csv", index=False)
        optuna_status_path = write_optuna_run_status(search_results, best_params)
        plot_optimization_history(studies, FIGURES_DIR / "optuna_optimization_history.png")
        plot_param_importance(studies, FIGURES_DIR / "optuna_param_importance.png")

        print(best_params.to_string(index=False))
        finish_run(
            run_id,
            metrics={
                f"{row['dataset']}_best_cv_pr_auc": row["best_cv_pr_auc"]
                for row in best_rows
            },
            artifacts={
                "hyperparameter_search_results": TABLES_DIR / "hyperparameter_search_results.csv",
                "optuna_trials": TABLES_DIR / "optuna_trials.csv",
                "best_params": TABLES_DIR / "best_params.csv",
                "optuna_run_status": optuna_status_path,
                "optimization_history": FIGURES_DIR / "optuna_optimization_history.png",
                "param_importance": FIGURES_DIR / "optuna_param_importance.png",
            },
        )
    except Exception as exc:
        finish_run(run_id, status="failed", notes=str(exc))
        raise


if __name__ == "__main__":
    main()
