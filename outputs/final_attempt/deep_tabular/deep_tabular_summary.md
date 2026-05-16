# Deep Tabular Reproduction Summary

## Protocol
- Existing processed features only; no new feature engineering.
- Frozen train/validation/test split preserved.
- Train split was internally separated into model_train, early_stop_val, and calibration_val.
- Any KMeansSMOTE or SMOTE-Tomek was applied only to transformed model_train.
- Validation selected the top-3 deep configs; test evaluated only those locked configs.
- The grid is a deterministic coverage grid, not the full Cartesian product of all hyperparameters.

## HELOC top validation deep configs
- mlp_128m64_l2_0.001 / sigmoid / threshold_0_50: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.7297, recall=0.7624, specificity=0.6938, cost=1510.0.
- mlp_128m64_l2_0.001 / sigmoid / manual_review_band_mr_le_0_30: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.7043, recall=0.8208, specificity=0.6262, cost=762.0.
- mlp_128m64_l2_0.001 / sigmoid / f1_optimal: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.6662, recall=0.8802, specificity=0.5216, cost=1068.0.
- mlp_128m64_l2_0.001 / sigmoid / cost_optimal_FN5_FP1: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.5752, recall=0.9718, specificity=0.2218, cost=882.0.
- mlp_128m64_l2_0.001 / sigmoid / precision_ge_0_40: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.5752, recall=0.9718, specificity=0.2218, cost=882.0.
- mlp_128m64_l2_0.001 / sigmoid / precision_ge_0_45: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.5752, recall=0.9718, specificity=0.2218, cost=882.0.
- mlp_128-64_elu_activation / sigmoid / threshold_0_50: PR-AUC=0.8083, ROC-AUC=0.8053, precision=0.7378, recall=0.7644, specificity=0.7054, cost=1489.0.
- mlp_128-64_elu_activation / sigmoid / f1_optimal: PR-AUC=0.8083, ROC-AUC=0.8053, precision=0.6990, recall=0.8500, specificity=0.6030, cost=1146.0.

## TAIWAN top validation deep configs
- mlp_256-128-64_relu_base / sigmoid / threshold_0_50: PR-AUC=0.5519, ROC-AUC=0.7804, precision=0.6871, recall=0.3542, specificity=0.9542, cost=4499.0.
- mlp_256-128-64_relu_base / sigmoid / manual_review_band_mr_le_0_30: PR-AUC=0.5519, ROC-AUC=0.7804, precision=0.5383, recall=0.5396, specificity=0.8686, cost=2785.5.
- mlp_256-128-64_relu_base / sigmoid / f1_optimal: PR-AUC=0.5519, ROC-AUC=0.7804, precision=0.4677, recall=0.6443, specificity=0.7918, cost=3333.0.
- mlp_256-128-64_relu_base / sigmoid / precision_ge_0_45: PR-AUC=0.5519, ROC-AUC=0.7804, precision=0.4621, recall=0.6518, specificity=0.7845, cost=3317.0.
- mlp_256-128-64_relu_base / sigmoid / precision_ge_0_40: PR-AUC=0.5519, ROC-AUC=0.7804, precision=0.4046, recall=0.7257, specificity=0.6968, cost=3237.0.
- mlp_256-128-64_relu_base / sigmoid / cost_optimal_FN5_FP1: PR-AUC=0.5519, ROC-AUC=0.7804, precision=0.3683, recall=0.7800, specificity=0.6202, cost=3235.0.
- mlp_128m64_lr_0.0003_batch_512 / sigmoid / threshold_0_50: PR-AUC=0.5493, ROC-AUC=0.7770, precision=0.6719, recall=0.3580, specificity=0.9504, cost=4492.0.
- mlp_128m64_lr_0.0003_batch_512 / sigmoid / manual_review_band_mr_le_0_30: PR-AUC=0.5493, ROC-AUC=0.7770, precision=0.5558, recall=0.5177, specificity=0.8825, cost=2821.5.

## 1. Did DNN/MLP beat boosting models?

- heloc: best deep validation PR-AUC=0.8109; literature reproduction best=0.7993, advanced boosting best=0.7985.
- taiwan: best deep validation PR-AUC=0.5519; literature reproduction best=0.5642, advanced boosting best=0.5653.

## 2. Did KMeansSMOTE + MLP produce a large jump?

- heloc: KMeansSMOTE best=0.7978, no-resampling best=0.8109; no large leakage-free jump.
- taiwan: KMeansSMOTE best=0.5320, no-resampling best=0.5519; no large leakage-free jump.

## 3. Did focal loss help?

- heloc: focal best=0.8061, binary-crossentropy best=0.8109.
- taiwan: focal best=0.5485, binary-crossentropy best=0.5519.

## 4. Probability calibration

- heloc: best median validation ECE=sigmoid (0.0208).
- taiwan: best median validation ECE=sigmoid (0.0111).

## 5. Precision/recall balance

- taiwan locked mlp_256-128-64_relu_base: test precision=0.6585, recall=0.3459, specificity=0.9491, cost=4578.0.
- taiwan locked mlp_128m64_lr_0.0003_batch_512: test precision=0.6606, recall=0.3580, specificity=0.9478, cost=4504.0.
- taiwan locked mlp_128m64_lr_0.0003_batch_128: test precision=0.6585, recall=0.3647, specificity=0.9463, cost=4466.0.
- heloc locked mlp_128m64_l2_0.001: test precision=0.7258, recall=0.7879, specificity=0.6769, cost=1396.0.
- heloc locked mlp_128-64_elu_activation: test precision=0.7260, recall=0.7782, specificity=0.6811, cost=1442.0.
- heloc locked mlp_128-64_smote_tomek_train_only: test precision=0.7223, recall=0.7792, specificity=0.6748, cost=1443.0.

## 6. Overfitting signs

- heloc: median epochs run=24; early stopping was active on a train-internal early_stop_val split.
- taiwan: median epochs run=42; early stopping was active on a train-internal early_stop_val split.

## 7. Relation to high BP-NN literature scores
The leakage-free reproduction does not justify claiming unusually high BP-NN performance on natural held-out distributions. If much higher scores are reported elsewhere, plausible causes include split-before-resampling, balanced/resampled test sets, or using test evidence during model selection.
