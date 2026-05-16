# Strict Literature Reproduction Summary

## Protocol status
- Split policy: frozen 60/20/20 stratified train/validation/test split.
- CV: RepeatedStratifiedKFold with n_splits=5, n_repeats=3, random_state=42.
- Resampling: inside imbalanced-learn pipelines only; no split-before-SMOTE violation.
- Test usage: held out until the validation-selected locked configuration was fixed.
- New features: none.

## 1. Did literature-inspired resampling improve AUC/PR-AUC?
### HELOC top validation configs
- CatBoost Balanced: PR-AUC=0.7993, ROC-AUC=0.8033, cost-policy precision=0.5740, recall=0.9747, cost=873.0
- CatBoost SqrtBalanced: PR-AUC=0.7968, ROC-AUC=0.8025, cost-policy precision=0.5617, recall=0.9834, cost=873.0
- CatBoost class_weights: PR-AUC=0.7966, ROC-AUC=0.8027, cost-policy precision=0.6092, recall=0.9533, cost=868.0
- CatBoost baseline: PR-AUC=0.7954, ROC-AUC=0.8022, cost-policy precision=0.5469, recall=0.9932, cost=880.0
- XGBoost + SMOTE-Tomek: PR-AUC=0.7953, ROC-AUC=0.8002, cost-policy precision=0.6010, recall=0.9562, cost=877.0
### TAIWAN top validation configs
- XGBoost + no resampling: PR-AUC=0.5642, ROC-AUC=0.7889, cost-policy precision=0.3749, recall=0.7747, cost=3209.0
- LightGBM + no resampling: PR-AUC=0.5621, ROC-AUC=0.7861, cost-policy precision=0.3830, recall=0.7611, cost=3212.0
- GBDT + RandomOverSampler: PR-AUC=0.5604, ROC-AUC=0.7840, cost-policy precision=0.3627, recall=0.7943, cost=3217.0
- CatBoost SqrtBalanced: PR-AUC=0.5596, ROC-AUC=0.7857, cost-policy precision=0.3459, recall=0.8357, cost=3187.0
- CatBoost baseline: PR-AUC=0.5593, ROC-AUC=0.7867, cost-policy precision=0.3795, recall=0.7762, cost=3169.0

Overall, resampling was evaluated as a protocol stress test, not as a score-inflation exercise. Improvements must be interpreted on validation first and then checked once on the natural held-out test distribution for the locked configuration.

## 2. Did KMeansSMOTE produce a literature-style jump?
- heloc: best KMeansSMOTE PR-AUC=0.7949, no-resampling/baseline PR-AUC=0.7954; did not materially help.
- taiwan: best KMeansSMOTE PR-AUC=0.5546, no-resampling/baseline PR-AUC=0.5642; did not materially help.

## 3. Did natural test precision/recall balance improve?
- taiwan: reproduction locked test PR-AUC=0.5645, recall=0.7626, precision=0.3661. V2 operational baseline is a manual-review policy (Best CatBoost); binary reproduction costs are not directly comparable to review-adjusted V2 costs.
- heloc: reproduction locked test PR-AUC=0.8136, recall=0.9708, precision=0.5661. V2 operational baseline is a manual-review policy (Best Scorecard); binary reproduction costs are not directly comparable to review-adjusted V2 costs.

## 4. Did resampling hurt calibration?
Calibration was measured with Brier score and ECE for every validation configuration. Any resampled model with improved PR-AUC but worse ECE/Brier should be treated as a screening model requiring calibration review, not as a ready automatic decision model.

## 5. Is the best config better than V2 manual-review policy?
Not as an operational replacement by default. V2 uses a corrected manual-review cost model and capacity-aware three-way policy; this reproduction experiment mainly tests whether high literature-style binary-model scores survive a leakage-free protocol.

## 6. Is there any leakage-like high score?
No result should be treated as leakage evidence solely because it is high. Leakage suspicion would require implausibly high test AUC/PR-AUC, train/test distribution contamination, or split-before-resampling. This script explicitly prevents split-before-resampling and does not resample test.

## 7. If high scores are not reproduced, what is the likely reason?
The likely explanation is protocol difference: many optimistic literature scores can arise from resampling before splitting, test-set balancing, weak separation between model selection and final evaluation, or reporting threshold-insensitive metrics without natural-distribution precision/cost checks.

## Failed configurations
- heloc / GBDT + ADASYN: cv - No samples will be generated with the provided ratio settings.
