# SCRE-Credit Project Scope

## Active Scope

This repository is an experimental, reproducible research pipeline for credit-risk modelling. The active scope is SCRE-Credit: a Stability- and Cost-Regularized Ensemble for Credit Risk.

The active modelling work is limited to data preparation, feature engineering, model training, calibration, threshold and cost analysis, explainability, stability, faithfulness, and external validation experiments.

## Out of Scope

- German Credit / Option A as an active modelling track.
- Final report drafting as the current project objective.
- Presentation generation as the current project objective.
- Conformal prediction.
- Production loan-decision deployment.
- Regulatory approval or compliance claims.

## Datasets

Primary dataset:

- UCI Default of Credit Card Clients / Taiwan.

External validation dataset:

- FICO HELOC / Explainable Machine Learning Challenge.

Taiwan and HELOC are not merged. Taiwan is the primary development dataset. HELOC is used to evaluate whether the methodology is portable to a second credit-risk dataset with a different feature schema.

## Proposed Method

SCRE-Credit is a Stability- and Cost-Regularized Ensemble for Credit Risk. It combines candidate probability models using a reliability score based on predictive performance, probability calibration, cost-sensitive behavior, explanation stability, and faithfulness evidence.

The method is a framework-level contribution. It does not claim to invent a new base classifier.

## Main Research Questions

- Can a reliability-weighted ensemble improve credit-risk performance beyond single baseline and boosting models?
- Does cost-sensitive model selection and thresholding reduce expected false-negative-heavy decision cost?
- Do calibrated probabilities improve decision reliability without being presented as an ROC-AUC improvement trick?
- Are explanations stable across seeds, model families, and background samples?
- Does the same SCRE-Credit pipeline remain usable on FICO HELOC as an external validation dataset?

## What Will Not Be Claimed

- That SCRE-Credit is a new theoretical machine-learning algorithm.
- That Taiwan and HELOC results are directly comparable after merging.
- That HELOC validates population-level generalization for Taiwan.
- That SHAP or LIME explanations are causal.
- That the pipeline is production-ready for real lending decisions.
- That publishable novelty is established before full stability, faithfulness, ablation, statistical comparison, and external-validation evidence is complete.
