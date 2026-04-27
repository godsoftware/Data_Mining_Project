# SCRE-Credit Project Scope

## Fixed Scope

Project name:

SCRE-Credit: A Stability- and Cost-Regularized Explainable Ensemble Framework for Credit Risk Prediction

Primary dataset:

- UCI Default of Credit Card Clients / Taiwan

External validation dataset:

- FICO HELOC / Explainable Machine Learning Challenge dataset

Taiwan and HELOC are not merged. Taiwan is the main development and evaluation dataset. HELOC is used only to test whether the SCRE-Credit methodology is portable to a second financial credit-risk dataset with a different feature schema.

## Main Claim

This project does not claim to invent a new base machine-learning algorithm. The defensible contribution is an integrated credit-risk framework that combines:

- predictive performance,
- probability calibration,
- cost-sensitive thresholding,
- manual-review / reject-option decision bands,
- SHAP and LIME explainability,
- explanation stability,
- faithfulness testing,
- external robustness validation.

The SCRE-Credit ensemble weights candidate models by a reliability score rather than by ROC-AUC or accuracy alone.

## Active In Scope

- Taiwan data loading, cleaning, validation, and feature engineering.
- HELOC data loading, cleaning, validation, and reduced external robustness pipeline.
- Logistic Regression baseline.
- WOE / scorecard-style Logistic Regression baseline.
- Random Forest.
- XGBoost.
- LightGBM.
- CatBoost.
- Monotonic XGBoost and monotonic LightGBM.
- SMOTENC + XGBoost.
- Class-weighted LightGBM.
- Reliability-weighted SCRE-Credit ensemble.
- ROC-AUC, PR-AUC, recall, precision, F1, specificity, G-mean.
- Brier score, ECE, calibration slope/intercept, calibration curve.
- Cost curves and threshold optimization with FN cost > FP cost.
- Three-way decision policy: low risk, manual review, high risk.
- SHAP global/local analysis.
- LIME local analysis.
- SHAP stability and faithfulness experiments.
- Bootstrap confidence intervals.
- Statistical comparison and ablation study as planned extension steps.

## Explicitly Out of Scope

- German Credit as an active modelling dataset.
- Conformal prediction.
- Final report drafting in the current phase.
- Presentation generation in the current phase.
- Claims of a new theoretical ML algorithm.
- Claims of a publishable breakthrough before full stability, ablation, statistical, and external robustness evidence is complete.

## Archive Policy

Legacy work is not deleted. It is archived so the active research scope remains clean:

- archive/old_german_credit_option/
- archive/old_report_drafts/
- archive/old_notebooks/

## Current Limitation To Track

Current SCRE-Credit implementation supports reliability-weighted ensembling and external HELOC robustness. The stability and faithfulness components can be injected into the reliability score, but the full 30/50/100-seed stability and expanded deletion/insertion faithfulness experiments must still be run before making strong claims about explanation reliability.
