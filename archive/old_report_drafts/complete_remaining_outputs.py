"""Complete remaining EDA, tuning, local XAI, and report tables."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from data_preprocessing import PROJECT_ROOT, TARGET_COLUMN
from evaluation import binary_classification_metrics
from explainability import build_lime_explainer, transformed_feature_frame
from model_training import (
    CATEGORICAL_COLUMNS,
    candidate_models,
    make_pipeline,
    split_features_target,
)


RAW_CLEAN_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_ready_data.csv"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "final_model.joblib"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
SHAP_DIR = PROJECT_ROOT / "outputs" / "shap_results"
REPORT_DIR = PROJECT_ROOT / "report"


def ensure_dirs() -> None:
    """Create output folders."""

    for path in [TABLES_DIR, FIGURES_DIR, SHAP_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_dataset_tables() -> None:
    """Save dataset, literature, feature-engineering, and search-space tables."""

    variable_rows = [
        ("ID", "Identifier", "Unique customer ID; removed before modelling."),
        ("LIMIT_BAL", "Credit/account", "Credit limit in NT dollars."),
        ("SEX", "Demographic", "Gender code: 1 = male, 2 = female."),
        ("EDUCATION", "Demographic", "Education level; anomalous values merged into Others."),
        ("MARRIAGE", "Demographic", "Marital status; anomalous value 0 merged into Others."),
        ("AGE", "Demographic", "Customer age in years."),
        ("PAY_0, PAY_2-PAY_6", "Payment status", "Six-month repayment status / delay history."),
        ("BILL_AMT1-BILL_AMT6", "Bill amount", "Six-month bill statement amounts."),
        ("PAY_AMT1-PAY_AMT6", "Payment amount", "Six-month previous payment amounts."),
        ("default_next_month", "Target", "1 = default next month, 0 = non-default."),
    ]
    pd.DataFrame(variable_rows, columns=["variable", "group", "description"]).to_csv(
        TABLES_DIR / "dataset_variable_description.csv",
        index=False,
    )

    literature_rows = [
        (
            "Yeh and Lien (2009)",
            "UCI Taiwan default dataset; compared six data-mining methods.",
            "Probability quality matters more than only hard classification.",
            "Supports calibration/probability focus.",
        ),
        (
            "Jin, Wu, and Zhao (2022)",
            "SMOTE-XGBoost for UCI credit-card defaulters.",
            "SMOTE-XGBoost can improve recall/AUC in imbalanced default prediction.",
            "Motivates testing resampling, though our SMOTENC did not win.",
        ),
        (
            "Bhandary and Ghosh (2025)",
            "Compared statistical and machine-learning models on the UCI dataset.",
            "Modern ML models achieved about 0.77 AUC and around 81.8% top accuracy.",
            "Our XGBoost AUC 0.780 is in the expected literature band.",
        ),
        (
            "Lin and Wang (2025)",
            "SHAP stability case study using 100 XGBoost models.",
            "Top SHAP drivers tend to be more stable than middle-ranked features.",
            "Directly motivates our 10-seed SHAP stability analysis.",
        ),
    ]
    pd.DataFrame(
        literature_rows,
        columns=["paper", "scope", "main_finding", "relevance_to_project"],
    ).to_csv(TABLES_DIR / "reviewed_papers_summary.csv", index=False)

    feature_rows = [
        ("delay_count", "Delay", "Number of months with positive repayment delay."),
        ("max_delay", "Delay", "Maximum delay level across six months."),
        ("avg_delay", "Delay", "Average clipped repayment delay."),
        ("recent_delay", "Delay", "Indicator for recent delay in PAY_0 or PAY_2."),
        ("severe_delay_count", "Delay", "Count of months with delay >= 2."),
        ("total_bill_amt", "Bill", "Total six-month bill amount."),
        ("avg_bill_amt", "Bill", "Average six-month bill amount."),
        ("max_bill_amt", "Bill", "Maximum bill amount."),
        ("bill_trend", "Bill", "BILL_AMT1 - BILL_AMT6."),
        ("total_pay_amt", "Payment", "Total six-month payment amount."),
        ("avg_pay_amt", "Payment", "Average six-month payment amount."),
        ("max_pay_amt", "Payment", "Maximum payment amount."),
        ("payment_to_bill_ratio", "Ratio", "total_pay_amt / (abs(total_bill_amt) + 1)."),
        ("utilization_proxy", "Ratio", "avg_bill_amt / (LIMIT_BAL + 1)."),
    ]
    pd.DataFrame(feature_rows, columns=["feature", "group", "definition"]).to_csv(
        TABLES_DIR / "feature_engineering_list.csv",
        index=False,
    )

    search_rows = [
        ("xgboost", "model__n_estimators", "[200, 300, 400, 600]"),
        ("xgboost", "model__max_depth", "[3, 4, 5, 6]"),
        ("xgboost", "model__learning_rate", "Uniform 0.02-0.10"),
        ("xgboost", "model__subsample", "Uniform 0.75-1.00"),
        ("xgboost", "model__colsample_bytree", "Uniform 0.75-1.00"),
        ("xgboost", "model__min_child_weight", "[1, 2, 3, 4, 5]"),
    ]
    pd.DataFrame(search_rows, columns=["model", "parameter", "search_values"]).to_csv(
        TABLES_DIR / "hyperparameter_search_space.csv",
        index=False,
    )


def signed_log1p(values: pd.Series) -> pd.Series:
    """Signed log transform for bill variables that can be negative."""

    return np.sign(values) * np.log1p(np.abs(values))


def save_extra_eda_figures() -> None:
    """Generate the remaining EDA figures requested in the roadmap."""

    df = pd.read_csv(RAW_CLEAN_PATH)
    sns.set_theme(style="whitegrid")

    category_specs = [
        ("EDUCATION", "Default Rate by Education", "default_rate_by_education.png"),
        ("MARRIAGE", "Default Rate by Marriage", "default_rate_by_marriage.png"),
        ("SEX", "Default Rate by Sex", "default_rate_by_sex.png"),
    ]

    rate_tables = []
    for column, title, filename in category_specs:
        table = (
            df.groupby(column)[TARGET_COLUMN]
            .agg(default_rate="mean", count="size")
            .reset_index()
        )
        table["variable"] = column
        rate_tables.append(table.rename(columns={column: "category"}))

        plt.figure(figsize=(7, 4))
        sns.barplot(data=table, x=column, y="default_rate", color="#457b9d")
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Default rate")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close()

    delay_count = (df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].clip(lower=0) > 0).sum(axis=1)
    delay_df = pd.DataFrame({"delay_count": delay_count, TARGET_COLUMN: df[TARGET_COLUMN]})
    delay_table = (
        delay_df.groupby("delay_count")[TARGET_COLUMN]
        .agg(default_rate="mean", count="size")
        .reset_index()
    )
    delay_table.to_csv(TABLES_DIR / "default_rate_by_delay_count.csv", index=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=delay_table, x="delay_count", y="default_rate", color="#6a994e")
    plt.title("Default Rate by Delayed Payment Count")
    plt.xlabel("Delayed payment count")
    plt.ylabel("Default rate")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "default_rate_by_delay_count.png", dpi=300)
    plt.close()

    pd.concat(rate_tables, ignore_index=True).to_csv(
        TABLES_DIR / "default_rate_by_demographics.csv",
        index=False,
    )

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), bill_cols):
        sns.histplot(signed_log1p(df[column]), bins=40, ax=ax, color="#2a9d8f")
        ax.set_title(column)
        ax.set_xlabel("signed log1p amount")
    fig.suptitle("BILL_AMT1-BILL_AMT6 Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bill_amount_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, column in zip(axes.ravel(), pay_cols):
        sns.histplot(np.log1p(df[column]), bins=40, ax=ax, color="#e76f51")
        ax.set_title(column)
        ax.set_xlabel("log1p amount")
    fig.suptitle("PAY_AMT1-PAY_AMT6 Distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "payment_amount_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ks_statistic(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Return ROC-based KS statistic."""

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def save_final_metrics_with_gmean_and_ks() -> None:
    """Save final metrics at 0.50 and recommended cost threshold."""

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split = split_features_target(df, target_column=TARGET_COLUMN, random_state=42)
    model = joblib.load(MODEL_PATH)
    y_proba = model.predict_proba(split.X_test)[:, 1]

    rows = []
    for threshold, label in [(0.50, "default_050"), (0.15, "cost_sensitive_015")]:
        metrics = binary_classification_metrics(split.y_test, y_proba, threshold=threshold)
        metrics["operating_point"] = label
        metrics["expected_cost_fn5_fp1"] = 5 * metrics["fn"] + metrics["fp"]
        metrics["g_mean"] = float(np.sqrt(metrics["recall"] * metrics["specificity"]))
        metrics["ks_statistic"] = ks_statistic(split.y_test, y_proba)
        rows.append(metrics)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "final_metrics_with_gmean_ks.csv", index=False)

    # Decile-level KS table for scorecard-style reporting.
    decile_df = pd.DataFrame({"y_true": split.y_test.to_numpy(), "score": y_proba})
    decile_df["decile"] = pd.qcut(
        decile_df["score"].rank(method="first"),
        10,
        labels=False,
    ) + 1
    decile_df["decile"] = 11 - decile_df["decile"]
    total_bad = decile_df["y_true"].sum()
    total_good = len(decile_df) - total_bad
    ks_rows = []
    cum_bad = 0
    cum_good = 0
    for decile, group in decile_df.groupby("decile"):
        bad = int(group["y_true"].sum())
        good = int(len(group) - bad)
        cum_bad += bad
        cum_good += good
        ks_rows.append(
            {
                "decile": int(decile),
                "count": int(len(group)),
                "bad": bad,
                "good": good,
                "bad_rate": bad / len(group),
                "cum_bad_rate": cum_bad / total_bad,
                "cum_good_rate": cum_good / total_good,
                "ks": abs(cum_bad / total_bad - cum_good / total_good),
            }
        )
    pd.DataFrame(ks_rows).to_csv(TABLES_DIR / "ks_decile_table.csv", index=False)


def run_small_hyperparameter_tuning() -> None:
    """Run a compact RandomizedSearchCV for the final XGBoost family."""

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split = split_features_target(df, target_column=TARGET_COLUMN, random_state=42)
    model = candidate_models(random_state=42)["xgboost"]
    pipeline = make_pipeline(model, split.X_train)

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": randint(3, 7),
        "model__learning_rate": uniform(0.02, 0.08),
        "model__subsample": uniform(0.75, 0.25),
        "model__colsample_bytree": uniform(0.75, 0.25),
        "model__min_child_weight": randint(1, 6),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        random_state=42,
        refit=True,
        verbose=1,
    )
    search.fit(split.X_train, split.y_train)

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    keep_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_model__n_estimators",
        "param_model__max_depth",
        "param_model__learning_rate",
        "param_model__subsample",
        "param_model__colsample_bytree",
        "param_model__min_child_weight",
    ]
    results[keep_cols].to_csv(TABLES_DIR / "hyperparameter_tuning_results.csv", index=False)

    y_proba = search.best_estimator_.predict_proba(split.X_test)[:, 1]
    tuned_metrics = binary_classification_metrics(split.y_test, y_proba, threshold=0.5)
    tuned_metrics["model"] = "xgboost_tuned_randomized_search"
    tuned_metrics["best_cv_average_precision"] = float(search.best_score_)
    tuned_metrics["best_params"] = json.dumps(search.best_params_, sort_keys=True)
    pd.DataFrame([tuned_metrics]).to_csv(TABLES_DIR / "tuned_xgboost_test_metrics.csv", index=False)
    joblib.dump(search.best_estimator_, PROJECT_ROOT / "outputs" / "models" / "xgboost_tuned_randomized_search.joblib")


def local_case_indices(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict[str, int]:
    """Select representative local explanation cases."""

    pred = (y_proba >= threshold).astype(int)
    y_array = y_true.to_numpy()
    probabilities = pd.Series(y_proba, index=y_true.index)

    cases: dict[str, int] = {}
    correct_default = y_true.index[(y_array == 1) & (pred == 1)]
    correct_non_default = y_true.index[(y_array == 0) & (pred == 0)]
    false_negative = y_true.index[(y_array == 1) & (pred == 0)]
    false_positive = y_true.index[(y_array == 0) & (pred == 1)]

    if len(correct_default):
        cases["correct_default_high_risk"] = probabilities.loc[correct_default].idxmax()
    if len(correct_non_default):
        cases["correct_non_default_low_risk"] = probabilities.loc[correct_non_default].idxmin()
    if len(false_negative):
        cases["false_negative_borderline"] = probabilities.loc[false_negative].sub(threshold).abs().idxmin()
    elif len(false_positive):
        cases["false_positive_borderline"] = probabilities.loc[false_positive].sub(threshold).abs().idxmin()

    borderline = probabilities.sub(threshold).abs().idxmin()
    cases["borderline_near_threshold"] = int(borderline)
    return cases


def save_local_shap_and_lime_cases() -> None:
    """Generate three-plus local SHAP and LIME explanations."""

    import shap

    df = pd.read_csv(PROCESSED_DATA_PATH)
    split = split_features_target(df, target_column=TARGET_COLUMN, random_state=42)
    model = joblib.load(MODEL_PATH)
    y_proba = model.predict_proba(split.X_test)[:, 1]
    cases = local_case_indices(split.y_test, y_proba, threshold=0.15)

    # SHAP fallback uses model-agnostic permutation on transformed feature space.
    X_background_raw = split.X_train.sample(min(100, len(split.X_train)), random_state=42)
    X_background = transformed_feature_frame(model, X_background_raw)
    selected_raw = split.X_test.loc[list(cases.values())]
    selected_transformed = transformed_feature_frame(model, selected_raw)
    fitted_model = model.named_steps["model"]
    masker = shap.maskers.Independent(X_background, max_samples=min(100, len(X_background)))
    explainer = shap.Explainer(
        lambda values: fitted_model.predict_proba(values)[:, 1],
        masker,
        algorithm="permutation",
        feature_names=list(selected_transformed.columns),
    )
    shap_values = explainer(
        selected_transformed,
        max_evals=2 * selected_transformed.shape[1] + 1,
        silent=True,
    )

    case_rows = []
    for position, (case_name, row_index) in enumerate(cases.items()):
        probability = float(model.predict_proba(split.X_test.loc[[row_index]])[:, 1][0])
        actual = int(split.y_test.loc[row_index])
        predicted = int(probability >= 0.15)
        case_rows.append(
            {
                "case_name": case_name,
                "row_index": int(row_index),
                "actual": actual,
                "predicted_at_threshold_015": predicted,
                "predicted_probability": probability,
            }
        )
        shap.plots.waterfall(shap_values[position], show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(
            SHAP_DIR / f"shap_waterfall_{case_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    pd.DataFrame(case_rows).to_csv(TABLES_DIR / "local_explanation_cases.csv", index=False)

    lime_explainer = build_lime_explainer(
        split.X_train,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    lime_rows = []
    for case_name, row_index in cases.items():
        exp = lime_explainer.explain_instance(
            split.X_test.loc[row_index].to_numpy(),
            lambda values: model.predict_proba(pd.DataFrame(values, columns=split.X_train.columns)),
            num_features=10,
        )
        exp.save_to_file(str(SHAP_DIR / f"lime_{case_name}.html"))
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"lime_{case_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        for feature_rule, weight in exp.as_list():
            lime_rows.append(
                {
                    "case_name": case_name,
                    "row_index": int(row_index),
                    "feature_rule": feature_rule,
                    "lime_weight": float(weight),
                }
            )
    pd.DataFrame(lime_rows).to_csv(TABLES_DIR / "lime_local_cases.csv", index=False)


def main() -> None:
    """Run all completion tasks."""

    ensure_dirs()
    save_dataset_tables()
    save_extra_eda_figures()
    save_final_metrics_with_gmean_and_ks()
    run_small_hyperparameter_tuning()
    save_local_shap_and_lime_cases()
    print("Remaining outputs completed.")


if __name__ == "__main__":
    main()

