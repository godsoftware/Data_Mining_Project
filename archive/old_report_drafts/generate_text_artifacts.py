"""Generate final Markdown report, slide notes, and updated limitation notes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_preprocessing import PROJECT_ROOT


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "report"


def md_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    """Small dependency-free Markdown table renderer."""

    shown = df.head(max_rows).copy()
    for column in shown.columns:
        shown[column] = shown[column].astype(str).str.slice(0, 70)
    headers = list(shown.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def csv(name: str) -> pd.DataFrame:
    """Load an output CSV table."""

    return pd.read_csv(TABLES_DIR / name)


def main() -> None:
    """Write report/final_report.md and report/presentation_slides.md."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((TABLES_DIR / "final_model_metadata.json").read_text(encoding="utf-8"))
    metrics = csv("final_metrics_with_gmean_ks.csv")
    default_metrics = metrics.loc[metrics["operating_point"] == "default_050"].iloc[0]
    cost_metrics = metrics.loc[metrics["operating_point"] == "cost_sensitive_015"].iloc[0]
    shap = csv("shap_top_features.csv")
    stability = csv("shap_stability_spearman.csv")
    mean_rho = stability["spearman_rho"].mean()

    report = f"""# Explainable Credit Card Default Prediction Using Machine Learning, Imbalance Handling, Calibration, and SHAP Stability Analysis

## Abstract

This project builds an explainable credit-card default prediction framework using the UCI Default of Credit Card Clients / Taiwan dataset. The workflow combines data cleaning, payment-behavior feature engineering, imbalance-aware model comparison, compact hyperparameter tuning, calibration, cost-sensitive threshold optimization, SHAP, LIME, SHAP stability, and faithfulness testing. The final selected model is a tuned XGBoost classifier.

## 1. Problem Definition

The aim is to predict whether a credit-card client will default next month. The project does not optimize accuracy alone. It also asks whether model probabilities are calibrated, whether the classification threshold is business-aware, and whether the explanations are stable and faithful.

## 2. Dataset

The dataset contains 30,000 Taiwanese credit-card accounts, 23 predictors, and a binary target. The target is renamed to `default_next_month`. The class distribution is 23,364 non-default and 6,636 default customers.

Cleaning decisions:

- EDUCATION values 0, 5, and 6 were merged into 4 = Others.
- MARRIAGE value 0 was merged into 3 = Others.
- Missing values and duplicate rows were checked.
- Negative bill amounts were reported and preserved because they may represent overpayment or credit balance.

### Dataset Variable Description

{md_table(csv("dataset_variable_description.csv"), 12)}

## 3. Literature and Research Gap

Prior work often focuses on prediction, imbalance handling, or explanation separately. This project combines predictive performance, calibration, threshold optimization, SHAP/LIME explanations, SHAP stability, and faithfulness testing in one framework.

{md_table(csv("reviewed_papers_summary.csv"), 8)}

## 4. Methodology

The pipeline follows a leakage-safe design: train-test split first, transformations fit only on the training side, and resampling tested only inside training pipelines. Candidate models included Logistic Regression, balanced Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, XGBoost, LightGBM, and SMOTENC variants. A compact RandomizedSearchCV was then used for XGBoost.

### Feature Engineering

{md_table(csv("feature_engineering_list.csv"), 20)}

### Hyperparameter Search Space

{md_table(csv("hyperparameter_search_space.csv"), 10)}

## 5. Model Selection

The final model is `{metadata["selected_model"]}` with `{metadata["imbalance_strategy"]}`. It was selected because compact tuning improved ROC-AUC, Brier score, recall, and FN-weighted threshold cost while keeping PR-AUC in the same literature band.

### Hyperparameter Tuning Results

{md_table(csv("hyperparameter_tuning_results.csv"), 8)}

## 6. Final Evaluation

At threshold 0.50:

- Accuracy: {default_metrics["accuracy"]:.4f}
- Precision: {default_metrics["precision"]:.4f}
- Recall: {default_metrics["recall"]:.4f}
- F1-score: {default_metrics["f1"]:.4f}
- ROC-AUC: {default_metrics["roc_auc"]:.4f}
- PR-AUC: {default_metrics["pr_auc"]:.4f}
- Brier score: {default_metrics["brier_score"]:.4f}
- G-mean: {default_metrics["g_mean"]:.4f}
- KS statistic: {default_metrics["ks_statistic"]:.4f}

At cost-sensitive threshold 0.15:

- Precision: {cost_metrics["precision"]:.4f}
- Recall: {cost_metrics["recall"]:.4f}
- F1-score: {cost_metrics["f1"]:.4f}
- G-mean: {cost_metrics["g_mean"]:.4f}
- Expected cost: {cost_metrics["expected_cost_fn5_fp1"]:.0f}

### Threshold Comparison

{md_table(csv("threshold_comparison.csv"), 10)}

### Calibration Results

{md_table(csv("calibration_results.csv"), 6)}

## 7. Explainable AI Results

The most important SHAP features are:

{md_table(shap, 10)}

The top features are financially meaningful. `PAY_0`, `max_delay`, `recent_delay`, and `delay_count` capture repayment behavior; `LIMIT_BAL` and `utilization_proxy` capture exposure and credit burden.

### Local Explanation Cases

{md_table(csv("local_explanation_cases.csv"), 10)}

## 8. SHAP Stability and Faithfulness

The final XGBoost configuration was retrained across 10 random seeds. The mean pairwise Spearman correlation between SHAP rankings was {mean_rho:.3f}, suggesting strong ranking stability.

### SHAP Top-5 Frequency

{md_table(csv("shap_top5_frequency.csv"), 10)}

### Faithfulness Results

{md_table(csv("faithfulness_test_results.csv"), 10)}

Perturbing the top SHAP-ranked variables caused the largest AUC drops and probability shifts, especially for `PAY_0`, supporting the faithfulness of the explanations.

## 9. Remaining Limitations

- The project uses one public dataset from Taiwan; external validation is still needed.
- Hyperparameter tuning was compact rather than exhaustive.
- SHAP stability used 10 repeated seeds, while recent literature uses up to 100 models.
- Fairness analysis by demographic subgroup was not deeply investigated.
- The project is academic and should not be used as an automated credit decision system without governance, bias review, and external validation.

## 10. Conclusion

The project meets the proposed goals. It predicts default, handles imbalance-aware evaluation, optimizes threshold, evaluates calibration, explains decisions with SHAP and LIME, and tests explanation reliability using stability and faithfulness analysis. The tuned XGBoost model is selected as the final model because it provides strong discrimination, improved FN-weighted cost, and stable explanations centered on recent repayment behavior.
"""

    (REPORT_DIR / "final_report.md").write_text(report, encoding="utf-8")

    slides = f"""# Presentation Slides

## 1. Title
- Explainable Credit Card Default Prediction
- UCI Taiwan dataset
- Tuned XGBoost + calibration + threshold + SHAP/LIME

## 2. Problem
- Predict next-month default.
- Do not optimize accuracy alone.
- Evaluate reliability of both probabilities and explanations.

## 3. Dataset
- 30,000 rows, 23 predictors.
- 6,636 default and 23,364 non-default customers.
- EDUCATION and MARRIAGE category anomalies cleaned.

## 4. Feature Engineering
- Delay count, max delay, recent delay.
- Bill/payment totals, averages, trend.
- Payment-to-bill ratio and utilization proxy.

## 5. Modelling
- Logistic Regression, RF, HistGB, XGBoost, LightGBM.
- SMOTENC variants tested.
- Compact RandomizedSearchCV applied to XGBoost.

## 6. Final Results
- ROC-AUC: {default_metrics["roc_auc"]:.3f}
- PR-AUC: {default_metrics["pr_auc"]:.3f}
- Threshold 0.50 recall: {default_metrics["recall"]:.3f}
- Threshold 0.15 recall: {cost_metrics["recall"]:.3f}

## 7. Threshold and Calibration
- FN cost = 5, FP cost = 1.
- Best cost threshold: 0.15.
- Calibration checked with Brier score and calibration curve.

## 8. Explainability
- Top SHAP features: {", ".join(shap["feature"].head(5))}
- PAY_0 and delay behavior dominate default risk.
- LIME produced local explanations for representative customers.

## 9. Stability and Faithfulness
- Mean SHAP Spearman rho: {mean_rho:.3f}
- Top features are stable across repeated seeds.
- Perturbing top SHAP features causes largest prediction shifts.

## 10. Limitations and Conclusion
- Single dataset; external validation needed.
- Compact tuning and 10-seed stability.
- Project satisfies prediction, explanation, calibration, threshold, stability, and faithfulness goals.
"""
    (REPORT_DIR / "presentation_slides.md").write_text(slides, encoding="utf-8")

    limitations = f"""# Remaining Limitations and How They Were Addressed

## Addressed During Completion

- Extra EDA graphs were added for EDUCATION, MARRIAGE, SEX, delay_count, BILL_AMT, and PAY_AMT distributions.
- Dataset variable, literature review, feature engineering, hyperparameter search, and tuning result tables were added.
- G-mean and KS were added to final evaluation.
- A compact RandomizedSearchCV tuning step was added for XGBoost.
- Local SHAP and LIME explanations were expanded to multiple customer cases.
- Final report and presentation drafts were generated.

## Still Remaining

- External validation is not possible with the current single public dataset.
- Tuning is compact, not exhaustive.
- SHAP stability uses 10 seeds, not 100.
- Fairness analysis is not deeply developed.
- The model is an academic prototype and not production-ready for real credit decisions.
"""
    (REPORT_DIR / "remaining_limitations.md").write_text(limitations, encoding="utf-8")

    print("Generated final_report.md, presentation_slides.md, and remaining_limitations.md")


if __name__ == "__main__":
    main()

