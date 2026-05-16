# Interpretable Middle Models Summary

## Protocol
- Frozen train/validation/test split was preserved.
- Model fitting used model_train, calibration used calibration_val, and policy selection used validation only.
- Held-out test rows are locked validation-selected evidence only; they are not used for ranking.
- No new features or datasets were created.
- EBM used the full requested 216-config grid per dataset. max_rounds=1500 with early stopping was fixed for runtime control.
- EBM missing-value warning is documented: interpret-core visualizations do not show missing values directly, but the model score terms retain missing-value bins internally.

## TAIWAN validation best by family

- EBM: ebm_i0_lr0.005_bins128_ob8_ib0_leaf10, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.520, recall=0.568, specificity=0.851, PR-AUC=0.551, ROC-AUC=0.783, selection_cost=2677.5, MR rate=0.297, eligible=True
- Scorecard: scorecard_woe_balanced, policy=manual_review_high_precision_ge_0_45, calibration=sigmoid, precision=0.657, recall=0.396, specificity=0.941, PR-AUC=0.535, ROC-AUC=0.773, selection_cost=2653.0, MR rate=0.496, eligible=False
- Monotonic XGBoost: monotonic_xgboost_base, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.527, recall=0.568, specificity=0.855, PR-AUC=0.561, ROC-AUC=0.788, selection_cost=2672.5, MR rate=0.300, eligible=True
- Monotonic LightGBM: monotonic_lightgbm_conservative, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.508, recall=0.585, specificity=0.839, PR-AUC=0.562, ROC-AUC=0.788, selection_cost=2668.0, MR rate=0.296, eligible=True

## HELOC validation best by family

- EBM: ebm_i10_lr0.01_bins256_ob8_ib4_leaf5, policy=manual_review_high_precision_ge_0_45, calibration=isotonic, precision=0.701, recall=0.839, specificity=0.611, PR-AUC=0.786, ROC-AUC=0.801, selection_cost=726.0, MR rate=0.358, eligible=True
- Scorecard: scorecard_woe_unweighted, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.707, recall=0.825, specificity=0.629, PR-AUC=0.793, ROC-AUC=0.803, selection_cost=731.0, MR rate=0.299, eligible=True
- Monotonic XGBoost: monotonic_xgboost_conservative, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.675, recall=0.874, specificity=0.544, PR-AUC=0.796, ROC-AUC=0.801, selection_cost=747.5, MR rate=0.299, eligible=True
- Monotonic LightGBM: monotonic_lightgbm_conservative, policy=manual_review_mr_le_0_30, calibration=sigmoid, precision=0.669, recall=0.884, specificity=0.525, PR-AUC=0.793, ROC-AUC=0.798, selection_cost=756.0, MR rate=0.300, eligible=True

## Locked held-out evidence

### Taiwan
- Monotonic LightGBM: monotonic_lightgbm_conservative, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.497, recall=0.592, specificity=0.830, PR-AUC=0.559, ROC-AUC=0.782, cost=2765.5, MR rate=0.289
- Monotonic LightGBM: monotonic_lightgbm_base, policy=manual_review_mr_le_0_30, calibration=sigmoid, precision=0.507, recall=0.584, specificity=0.839, PR-AUC=0.558, ROC-AUC=0.780, cost=2750.5, MR rate=0.289
- Monotonic XGBoost: monotonic_xgboost_base, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.517, recall=0.580, specificity=0.847, PR-AUC=0.558, ROC-AUC=0.782, cost=2756.5, MR rate=0.295

### HELOC
- EBM: ebm_i10_lr0.01_bins256_ob8_ib4_leaf5, policy=manual_review_high_precision_ge_0_45, calibration=isotonic, precision=0.690, recall=0.849, specificity=0.586, PR-AUC=0.787, ROC-AUC=0.801, cost=744.0, MR rate=0.341
- EBM: ebm_i10_lr0.01_bins256_ob8_ib4_leaf2, policy=manual_review_high_precision_ge_0_45, calibration=isotonic, precision=0.714, recall=0.815, specificity=0.646, PR-AUC=0.790, ROC-AUC=0.802, cost=732.5, MR rate=0.387
- Scorecard: scorecard_woe_unweighted, policy=manual_review_mr_le_0_30, calibration=uncalibrated, precision=0.690, recall=0.826, specificity=0.597, PR-AUC=0.806, ROC-AUC=0.802, cost=787.0, MR rate=0.273

## Direct answers

1. Is EBM better than Scorecard?
   - Taiwan: yes for balanced eligibility. Best EBM is eligible with validation PR-AUC=0.551, recall=0.568, specificity=0.851; best Scorecard has lower PR-AUC=0.535 and is not eligible because recall/PR-AUC are weak despite low cost.
   - HELOC: mixed. The lowest-cost EBM row has validation cost=726.0, but its manual-review rate is 0.358. Under MR<=0.30, EBM cost is 733.0, while Scorecard cost is 731.0; Scorecard remains the safer governance benchmark.
2. Did EBM approach CatBoost?
   - Taiwan: no. EBM is useful as a transparent middle model but does not reach the V2 CatBoost manual-review baseline cost/PR-AUC. Monotonic LightGBM is closer to CatBoost than EBM.
   - HELOC: partially. EBM PR-AUC is competitive, but the best low-cost rows can require high manual-review workload; Scorecard remains more stable operationally.
3. Did monotonic model performance drop?
   - Taiwan: monotonic LightGBM is the strongest interpretable-middle candidate: validation cost=2668.0, PR-AUC=0.562, ROC-AUC=0.788; monotonicity probe violation rate is 0.0 in selected rows.
   - HELOC: monotonic models are competitive but do not beat Scorecard/EBM under corrected manual-review cost.
4. Did monotonic models give safer explanations?
   - Yes. The constraint report enforces increasing-risk signs for delay/utilization proxies and decreasing-risk signs for payment intensity variables. This is a governance advantage over unconstrained black-box effects.
5. Final interpretable model: Scorecard, EBM, or monotonic model?
   - Taiwan: Monotonic LightGBM is the best middle-ground candidate; EBM is the clearest non-scorecard alternative; Scorecard remains a reason-code benchmark but is weaker.
   - HELOC: Scorecard remains the safest interpretable benchmark; EBM is competitive but workload-sensitive.
6. Automatic rejection?
   - No. These should be framed as screening/manual-review candidates, not automatic rejection systems.

## Failure log
- Failed model rows: 0
