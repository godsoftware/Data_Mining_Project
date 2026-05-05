# Training-Level Imbalance Revision Summary: TAIWAN

All variants are trained on the train split. Calibration and threshold policies are selected on validation only. Test metrics are held-out final evaluations.

Baseline reference: catboost_baseline + uncalibrated + cost_optimal_FN5_FP1, precision=0.3616, recall=0.7905, FP=1852, cost=3242.

## Best Variant Per Model Family

| Model Variant | Calibration | Threshold Policy | Precision | Recall | Specificity | FP | FN | Cost | Comment |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| lightgbm_scale_pos_weight_3 | isotonic | precision_0_40 | 0.3994 | 0.7121 | 0.6959 | 1421 | 382 | 3331 | More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss. |
| xgboost_scale_pos_weight_3 | isotonic | precision_0_40 | 0.3944 | 0.7054 | 0.6925 | 1437 | 391 | 3392 | More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss. |
| catboost_class_weight_pos_5 | isotonic | precision_0_40 | 0.4145 | 0.6745 | 0.7295 | 1264 | 432 | 3424 | More balanced than baseline CatBoost: precision improves and FP drops with tolerable recall loss. |

## Audit Answers

1. Class weighting precision improvement rows: 414/450.
2. Rows where recall increased by more than 0.10 versus baseline CatBoost: 0/450.
3. Weighted models can distort probability quality; inspect Brier/ECE columns by calibration_type.
4. Calibration-improvement comments observed: 300/450.
5. CatBoost Balanced vs SqrtBalanced should be judged by validation-selected test rows in the CSV, not by name alone.
6. XGBoost scale_pos_weight best point is listed in the best-variant table when it survives validation criteria.
7. LightGBM is_unbalance probability quality is visible through validation/test Brier and ECE.
8. Best overall variant by validation-only criteria: xgboost_scale_pos_weight_3 / isotonic / precision_0_40.
9. Rows with FP reduction versus baseline CatBoost: 420/450.
10. Rows landing in precision 0.40-0.45 band: 127/450.
