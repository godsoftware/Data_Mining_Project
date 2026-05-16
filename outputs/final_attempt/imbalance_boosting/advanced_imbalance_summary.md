# Advanced Imbalance Boosting Summary

## Protocol
- No new features were created.
- Base models were fit on `model_train`, calibration was fit on `calibration_val`, and policy selection used the frozen validation split.
- Test set was used only for the validation-locked configuration.
- LightGBM probability-quality caveat is active: weighted LightGBM rows must be judged with Brier/ECE after calibration.

## HELOC best validation policies
- CatBoost / catboost_baseline / isotonic / manual_review_band_mr_le_0_30: precision=0.6719, recall=0.8695, specificity=0.5396, PR-AUC=0.7827, cost=747.0, MR rate=0.300.
- CatBoost / catboost_auto_sqrtbalanced / sigmoid / manual_review_band_mr_le_0_30: precision=0.6790, recall=0.8588, specificity=0.5597, PR-AUC=0.7929, cost=751.0, MR rate=0.298.
- CatBoost / catboost_baseline / uncalibrated / manual_review_band_mr_le_0_30: precision=0.6714, recall=0.8695, specificity=0.5385, PR-AUC=0.7955, cost=751.5, MR rate=0.298.
- CatBoost / catboost_auto_sqrtbalanced / uncalibrated / manual_review_band_mr_le_0_30: precision=0.6777, recall=0.8598, specificity=0.5565, PR-AUC=0.7929, cost=751.5, MR rate=0.295.
- CatBoost / catboost_baseline / sigmoid / manual_review_band_mr_le_0_30: precision=0.6674, recall=0.8734, specificity=0.5280, PR-AUC=0.7955, cost=752.0, MR rate=0.299.
- XGBoost / xgboost_spw_3_mds_1 / uncalibrated / manual_review_band_mr_le_0_30: precision=0.6712, recall=0.8724, specificity=0.5364, PR-AUC=0.7876, cost=752.0, MR rate=0.297.
- XGBoost / xgboost_spw_2_mds_1 / sigmoid / manual_review_band_mr_le_0_30: precision=0.6696, recall=0.8744, specificity=0.5322, PR-AUC=0.7886, cost=754.0, MR rate=0.300.
- XGBoost / xgboost_spw_2_mds_1 / uncalibrated / manual_review_band_mr_le_0_30: precision=0.6674, recall=0.8773, specificity=0.5259, PR-AUC=0.7886, cost=754.0, MR rate=0.294.

## TAIWAN best validation policies
- XGBoost / xgboost_scale_pos_weight_5 / sigmoid / manual_review_band_mr_le_0_30: precision=0.5138, recall=0.5878, specificity=0.8421, PR-AUC=0.5619, cost=2648.5, MR rate=0.298.
- XGBoost / xgboost_spw_5_mds_3 / sigmoid / manual_review_band_mr_le_0_30: precision=0.5138, recall=0.5878, specificity=0.8421, PR-AUC=0.5619, cost=2648.5, MR rate=0.298.
- XGBoost / xgboost_spw_5_mds_5 / sigmoid / manual_review_band_mr_le_0_30: precision=0.5138, recall=0.5878, specificity=0.8421, PR-AUC=0.5619, cost=2648.5, MR rate=0.298.
- CatBoost / catboost_class_weight_pos_2.5 / uncalibrated / manual_review_band_mr_le_0_30: precision=0.5433, recall=0.5576, specificity=0.8669, PR-AUC=0.5599, cost=2654.0, MR rate=0.299.
- XGBoost / xgboost_scale_pos_weight_5 / uncalibrated / manual_review_band_mr_le_0_30: precision=0.5177, recall=0.5833, specificity=0.8457, PR-AUC=0.5619, cost=2657.0, MR rate=0.299.
- XGBoost / xgboost_spw_5_mds_3 / uncalibrated / manual_review_band_mr_le_0_30: precision=0.5177, recall=0.5833, specificity=0.8457, PR-AUC=0.5619, cost=2657.0, MR rate=0.299.
- XGBoost / xgboost_spw_5_mds_5 / uncalibrated / manual_review_band_mr_le_0_30: precision=0.5177, recall=0.5833, specificity=0.8457, PR-AUC=0.5619, cost=2657.0, MR rate=0.299.
- CatBoost / catboost_class_weight_pos_2.5 / sigmoid / manual_review_band_mr_le_0_30: precision=0.5399, recall=0.5614, specificity=0.8641, PR-AUC=0.5599, cost=2657.0, MR rate=0.294.

## 1. Did class weighting improve precision?

- heloc: baseline best precision=0.6147, weighted best precision=0.6225.
- taiwan: baseline best precision=0.4235, weighted best precision=0.4338.

## 2. What happened to recall?

- heloc: selected validation recall=0.8695; this is the recall retained after constraints/calibration.
- taiwan: selected validation recall=0.5878; this is the recall retained after constraints/calibration.

## 3. Did calibration degrade or improve?

- heloc / isotonic: median Brier=0.1865, median ECE=0.0313.
- heloc / sigmoid: median Brier=0.1845, median ECE=0.0202.
- heloc / uncalibrated: median Brier=0.1951, median ECE=0.0994.
- taiwan / isotonic: median Brier=0.1342, median ECE=0.0096.
- taiwan / sigmoid: median Brier=0.1338, median ECE=0.0117.
- taiwan / uncalibrated: median Brier=0.1443, median ECE=0.0954.

## 4. Sigmoid vs isotonic

- heloc: lowest median ECE calibration = sigmoid (0.0202).
- taiwan: lowest median ECE calibration = isotonic (0.0096).

## 5-7. Best variant by family

### heloc
- CatBoost: catboost_baseline with isotonic and manual_review_band_mr_le_0_30 (cost=747.0, PR-AUC=0.7827).
- XGBoost: xgboost_spw_3_mds_1 with uncalibrated and manual_review_band_mr_le_0_30 (cost=752.0, PR-AUC=0.7876).
- LightGBM: lightgbm_scale_pos_weight_1.5 with uncalibrated and manual_review_band_mr_le_0_30 (cost=763.0, PR-AUC=0.7732).
### taiwan
- CatBoost: catboost_class_weight_pos_2.5 with uncalibrated and manual_review_band_mr_le_0_30 (cost=2654.0, PR-AUC=0.5599).
- XGBoost: xgboost_scale_pos_weight_5 with sigmoid and manual_review_band_mr_le_0_30 (cost=2648.5, PR-AUC=0.5619).
- LightGBM: lightgbm_scale_pos_weight_2 with uncalibrated and manual_review_band_mr_le_0_30 (cost=2695.5, PR-AUC=0.5535).

## 8. Did any variant beat V2 policy?

- taiwan: best advanced validation operational cost=2648.5; V2 review-adjusted validation cost=2650.5. Advanced variant is lower on this validation cost definition.
- heloc: best advanced validation operational cost=747.0; V2 review-adjusted validation cost=729.0. V2 remains lower or not directly beaten.

## Locked held-out test evidence
- taiwan: XGBoost / xgboost_scale_pos_weight_5 / sigmoid / manual_review_band_mr_le_0_30 test precision=0.4991, recall=0.5968, specificity=0.8299, operational cost=2775.5.
- heloc: CatBoost / catboost_baseline / isotonic / manual_review_band_mr_le_0_30 test precision=0.6647, recall=0.8774, specificity=0.5195, operational cost=768.0.
