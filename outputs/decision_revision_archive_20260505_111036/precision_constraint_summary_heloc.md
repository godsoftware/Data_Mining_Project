# Precision-Constrained Threshold Summary: HELOC

Thresholds were selected on validation only. Test set metrics are final evaluation of the validation-selected thresholds.

## Best Balanced Policies By Model

| Model | Constraint | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Best CatBoost | A_min_precision_0_35 | 0.1450 | 0.5623 | 0.9747 | 0.1763 | 780 | 26.0000 | 910 | balanced FP reduction with tolerable recall loss; FP delta=-18, precision delta=0.0042, recall delta=-0.0058, cost delta=12; selected on validation. |
| Best XGBoost | E_min_specificity_0_60 | 0.4100 | 0.6956 | 0.8580 | 0.5924 | 386 | 146 | 1116 | balanced FP reduction with tolerable recall loss; FP delta=-230, precision delta=0.0845, recall delta=-0.0837, cost delta=200; selected on validation. |
| Best LightGBM | E_min_specificity_0_60 | 0.4000 | 0.6898 | 0.8327 | 0.5935 | 385 | 172 | 1245 | balanced FP reduction with tolerable recall loss; FP delta=-454, precision delta=0.1433, recall delta=-0.1508, cost delta=321; selected on validation. |
| Best Scorecard | E_min_specificity_0_60 | 0.3900 | 0.6834 | 0.8463 | 0.5744 | 403 | 158 | 1193 | balanced FP reduction with tolerable recall loss; FP delta=-394, precision delta=0.1247, recall delta=-0.1352, cost delta=301; selected on validation. |
| SCRE-Optimized | E_min_specificity_0_60 | 0.4100 | 0.6858 | 0.8599 | 0.5723 | 405 | 144 | 1125 | balanced FP reduction with tolerable recall loss; FP delta=-380, precision delta=0.1244, recall delta=-0.1177, cost delta=225; selected on validation. |
| SCRE-Pareto | E_min_specificity_0_60 | 0.3900 | 0.6876 | 0.8609 | 0.5755 | 402 | 143 | 1117 | balanced FP reduction with tolerable recall loss; FP delta=-342, precision delta=0.1152, recall delta=-0.1080, cost delta=213; selected on validation. |
| Old SCRE-Credit | E_min_specificity_0_60 | 0.4050 | 0.7136 | 0.8191 | 0.6431 | 338 | 186 | 1268 | balanced FP reduction with tolerable recall loss; FP delta=-312, precision delta=0.1126, recall delta=-0.1333, cost delta=373; selected on validation. |

## Feasibility

- Total model-constraint rows: 98
- Feasible validation rows: 98
- Infeasible validation rows: 0
