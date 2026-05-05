# Training-Level Imbalance Revision Summary: HELOC

All variants are trained on the train split. Calibration and threshold policies are selected on validation only. Test metrics are held-out final evaluations.

Baseline reference: catboost_baseline + uncalibrated + cost_optimal_FN5_FP1, precision=0.5523, recall=0.9854, FP=821, cost=896.

## Best Variant Per Model Family

| Model Variant | Calibration | Threshold Policy | Precision | Recall | Specificity | FP | FN | Cost | Comment |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| catboost_class_weight_pos_1.5 | uncalibrated | default_0_50 | 0.6926 | 0.8414 | 0.5945 | 384 | 163 | 1199 | More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss. |
| xgboost_sample_weight_pos_1.5 | uncalibrated | default_0_50 | 0.7006 | 0.8307 | 0.6146 | 365 | 174 | 1235 | More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss. |
| lightgbm_is_unbalance | isotonic | default_0_50 | 0.6803 | 0.8259 | 0.5787 | 399 | 179 | 1294 | More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss. |

## Audit Answers

1. Class weighting precision improvement rows: 411/450.
2. Rows where recall increased by more than 0.10 versus baseline CatBoost: 0/450.
3. Weighted models can distort probability quality; inspect Brier/ECE columns by calibration_type.
4. Calibration-improvement comments observed: 300/450.
5. CatBoost Balanced vs SqrtBalanced should be judged by validation-selected test rows in the CSV, not by name alone.
6. XGBoost scale_pos_weight best point is listed in the best-variant table when it survives validation criteria.
7. LightGBM is_unbalance probability quality is visible through validation/test Brier and ECE.
8. Best overall variant by validation-only criteria: catboost_class_weight_pos_1.5 / uncalibrated / default_0_50.
9. Rows with FP reduction versus baseline CatBoost: 417/450.
10. Rows landing in precision 0.40-0.45 band: 0/450.
