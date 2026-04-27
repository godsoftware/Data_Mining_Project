# Active Research Plan - SCRE-Credit

This is the active project direction. It replaces the earlier final-report-oriented workflow, but it does not discard the validated Taiwan pipeline.

## Scope

Primary dataset:

- UCI Default of Credit Card Clients / Taiwan.

External robustness dataset:

- FICO HELOC / Explainable Machine Learning Challenge.

Important constraint:

- Taiwan and HELOC are not merged.
- Because their feature schemas differ, HELOC is used to validate the methodology, not to directly score HELOC rows with a Taiwan-trained model.

## New Method Name

SCRE-Credit:

A Stability- and Cost-Regularized Explainable Ensemble Framework for Credit Risk Prediction.

Turkish:

SCRE-Credit: Kredi Riski Tahmini için Kararlılık ve Maliyet Düzenlemeli Açıklanabilir Hibrit Ensemble Framework'ü.

## Core Claim

The project should not claim a new breakthrough base algorithm. The defensible claim is:

> This project proposes a hybrid credit-risk classification framework that jointly evaluates model performance, probability calibration, cost-sensitive decision-making, SHAP/LIME explainability, explanation stability, faithfulness, and external robustness, and uses these reliability signals in ensemble weighting.

The novelty is not XGBoost, SHAP, or SMOTE by themselves. The novelty is weighting and selecting models by a combined reliability objective: performance + calibration + cost + stability + faithfulness.

## Reliability Score

For each model `m`, SCRE-Credit computes normalized validation-set components:

```text
R_m =
    0.20 * PR_AUC_norm
  + 0.15 * ROC_AUC_norm
  + 0.15 * Recall_norm
  + 0.15 * Calibration_norm
  + 0.15 * Cost_norm
  + 0.10 * Stability_norm
  + 0.10 * Faithfulness_norm
```

Current implementation details:

- `Calibration_norm` uses min-max normalized Brier score as a lower-is-better component.
- `Cost_norm` uses min-max normalized FN/FP expected cost as a lower-is-better component.
- `Stability_norm` and `Faithfulness_norm` accept external repeated-seed/faithfulness scores; when these experiments are not yet supplied, the component is neutral and equal across models.
- Ensemble weights are direct reliability weights:

```text
w_m = R_m / sum(R_all_models)
P_final = sum(w_m * P_m)
```

## Three-Level Decision

Final decisions are three-level:

- Low risk: `P_final < t_low`
- Manual review: `t_low <= P_final <= t_high`
- High risk: `P_final > t_high`

The thresholds are not fixed. They are selected on the internal validation split by policy cost and recall trade-off, using:

- false negative cost,
- false positive cost,
- manual review cost,
- maximum review-rate constraint,
- minimum review-capture recall constraint.

## Preserved Work

- Taiwan data loading and cleaning.
- EDUCATION/MARRIAGE anomaly handling.
- Payment-behavior feature engineering.
- Stratified train/test split.
- Logistic Regression, Random Forest, XGBoost, LightGBM.
- SMOTENC/class-weight comparisons.
- Threshold and calibration analysis.
- SHAP and LIME explanations.
- SHAP stability and perturbation faithfulness.

## Archived From Active Scope

- Final report generation scripts.
- Presentation generation scripts.
- Old roadmap checklist.
- German Credit / Option A as an active modeling path.

## Added Modules

- `src/heloc_preprocessing.py`: HELOC external dataset cleaning.
- `src/scorecard.py`: WOE Logistic scorecard baseline.
- `src/scre_credit.py`: reliability-weighted ensemble.
- `src/reliability_analysis.py`: bootstrap CIs, cost curves, manual-review band, subgroup metrics.
- `src/monotonic_models.py`: monotonic-prior helper definitions.
- `src/counterfactual_recourse.py`: first recourse/counterfactual utilities.
- `src/experiment_registry.py`: JSONL experiment tracking.
- `src/run_scre_credit_experiments.py`: reproducible SCRE-Credit runner.
- `src/xai_reliability.py`: SHAP-LIME agreement utilities.
- `src/faithfulness_extensions.py`: deletion/insertion faithfulness curves.

## Active Model Pools

Taiwan full pool:

- M1 Logistic Regression
- M2 WOE / Scorecard Logistic Regression
- M3 Random Forest
- M4 XGBoost
- M5 LightGBM
- M6 CatBoost
- M7 Monotonic XGBoost
- M8 Monotonic LightGBM
- M9 SMOTENC + XGBoost
- M10 Class-weighted LightGBM

HELOC reduced pool:

- M1 Logistic Regression
- M2 WOE / Scorecard Logistic Regression
- M3 XGBoost
- M4 LightGBM
- M5 Monotonic XGBoost / LightGBM
- M6 SCRE-Credit ensemble

## Next Experiments

1. DONE - Run HELOC preprocessing quality report.
2. DONE - Run SCRE-Credit smoke test on Taiwan.
3. DONE - Run SCRE-Credit smoke test on HELOC.
4. DONE - Install Optuna and CatBoost into `tf210win`.
5. DONE - Add M1-M10 Taiwan model pool and reduced HELOC pool.
6. DONE - Add extended metrics: ECE, calibration slope/intercept, G-mean, cost improvement.
7. DONE - Add code support for SHAP-LIME agreement, Kendall's W, top-k overlap, deletion/insertion curves.
8. TODO - Add Optuna search for XGBoost/LightGBM/CatBoost.
9. TODO - Add 30-seed SHAP stability first, then scale to 50/100.
10. TODO - Run deletion/insertion/ablation faithfulness curves for all final candidate models.
11. TODO - Add subgroup reliability tables for Taiwan demographics and HELOC risk bands.
12. TODO - Add paired bootstrap model comparisons.
13. TODO - Add manual-review band analysis to final operating policy.

## Smoke-Test Results

Latest SCRE-Credit smoke tests were run with reliability-score weighting and 10 bootstrap samples for fast verification.

Taiwan:

- Full M1-M10 candidate pool ran successfully.
- Ensemble weights are now `w_m = R_m / sum(R)`, not softmax weights.
- Threshold 0.50 ROC-AUC: 0.7823.
- Threshold 0.50 PR-AUC: 0.5629.
- Threshold 0.50 recall: 0.4499.
- Threshold 0.50 ECE: 0.0680.
- Threshold 0.50 calibration slope: 1.0214.
- Cost-sensitive threshold selected by the current run: 0.20.
- Cost improvement vs threshold 0.50 under FN=5, FP=1: 17.81%.
- Three-level validation-selected policy: `t_low=0.20`, `t_high=0.60`.
- Test manual-review rate: 43.30%.
- Test review-capture recall: 83.42%.

HELOC:

- Reduced external robustness pool ran successfully.
- Ensemble weights are now `w_m = R_m / sum(R)`, not softmax weights.
- Threshold 0.50 ROC-AUC: 0.8075.
- Threshold 0.50 PR-AUC: 0.8144.
- Threshold 0.50 recall: 0.7607.
- Threshold 0.50 ECE: 0.0118.
- Threshold 0.50 calibration slope: 1.0179.
- Cost-sensitive threshold selected by the current run: 0.20.
- Cost improvement vs threshold 0.50 under FN=5, FP=1: 41.56%.
- Three-level validation-selected policy: `t_low=0.05`, `t_high=0.45`.
- Test manual-review rate: 41.82%.
- Test review-capture recall: 100.00%.

These are smoke-test outputs, not final tuned results.
