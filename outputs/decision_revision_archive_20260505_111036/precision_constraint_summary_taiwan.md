# Precision-Constrained Threshold Summary: TAIWAN

Thresholds were selected on validation only. Test set metrics are final evaluation of the validation-selected thresholds.

## Best Balanced Policies By Model

| Model | Constraint | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Best CatBoost | A_min_precision_0_35 | 0.1550 | 0.3602 | 0.7928 | 0.6000 | 1869 | 275 | 3244 | balanced FP reduction with tolerable recall loss; FP delta=-113, precision delta=0.0087, recall delta=-0.0166, cost delta=-3; selected on validation. |
| Best XGBoost | A_min_precision_0_35 | 0.1800 | 0.3927 | 0.7076 | 0.6893 | 1452 | 388 | 3392 | balanced FP reduction with tolerable recall loss; FP delta=-746, precision delta=0.0572, recall delta=-0.1289, cost delta=109; selected on validation. |
| Best LightGBM | B_min_precision_0_40 | 0.1800 | 0.3917 | 0.7197 | 0.6826 | 1483 | 372 | 3343 | balanced FP reduction with tolerable recall loss; FP delta=-164, precision delta=0.0163, recall delta=-0.0264, cost delta=11; selected on validation. |
| Best Scorecard | A_min_precision_0_35 | 0.1650 | 0.3831 | 0.7151 | 0.6730 | 1528 | 378 | 3418 | balanced FP reduction with tolerable recall loss; FP delta=-152, precision delta=0.0138, recall delta=-0.0264, cost delta=23; selected on validation. |
| SCRE-Optimized | I_fp_budget_1750 | 0.1650 | 0.3680 | 0.7581 | 0.6302 | 1728 | 321 | 3333 | balanced FP reduction with tolerable recall loss; FP delta=-104, precision delta=0.0078, recall delta=-0.0188, cost delta=21; selected on validation. |
| SCRE-Pareto | A_min_precision_0_35 | 0.1650 | 0.3651 | 0.7528 | 0.6283 | 1737 | 328 | 3377 | balanced FP reduction with tolerable recall loss; FP delta=-100, precision delta=0.0052, recall delta=-0.0256, cost delta=70; selected on validation. |
| Old SCRE-Credit | B_min_precision_0_40 | 0.2000 | 0.3921 | 0.7008 | 0.6914 | 1442 | 397 | 3427 | balanced FP reduction with tolerable recall loss; FP delta=-308, precision delta=0.0277, recall delta=-0.0550, cost delta=57; selected on validation. |

## Taiwan CatBoost Special Analysis

Old cost-threshold baseline: threshold=0.15, precision=0.3514, recall=0.8093, specificity=0.5759, FP=1982, cost=3247.

1. Precision 0.351 -> at least 0.40: YES; the requested precision constraints are feasible on validation and improve test precision where selected.
2. FP count reduction: YES; the best balanced requested policy is F_min_specificity_0_65 with FP delta -330.
3. Recall loss: -0.0686 versus the old cost threshold for that balanced policy.
4. Cost change: 125 versus the old cost threshold for that balanced policy.
5. More defensible model: YES, if framed as constrained screening/manual-review support rather than automatic rejection.
6. Most balanced requested constraint: F_min_specificity_0_65 (min_specificity >= 0.65).

### Requested CatBoost Constraints

| Constraint | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Delta FP | Delta Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_min_precision_0_40 | 0.1800 | 0.4177 | 0.6880 | 0.7276 | 1273 | 414 | 3343 | -709 | 96.0000 |
| C_min_precision_0_45 | 0.2050 | 0.4487 | 0.6526 | 0.7723 | 1064 | 461 | 3369 | -918 | 122 |
| F_min_specificity_0_65 | 0.1650 | 0.3731 | 0.7408 | 0.6465 | 1652 | 344 | 3372 | -330 | 125 |
| J_fp_budget_1500 | 0.1700 | 0.3838 | 0.7234 | 0.6702 | 1541 | 367 | 3376 | -441 | 129 |
| K_precision_0_40_recall_0_65 | 0.1800 | 0.4177 | 0.6880 | 0.7276 | 1273 | 414 | 3343 | -709 | 96.0000 |

## Feasibility

- Total model-constraint rows: 98
- Feasible validation rows: 97
- Infeasible validation rows: 1
