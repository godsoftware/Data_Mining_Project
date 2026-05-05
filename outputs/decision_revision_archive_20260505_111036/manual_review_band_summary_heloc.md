# Manual Review Band Optimization Summary: HELOC

Low/high thresholds were selected on validation only. Test metrics report the held-out performance of those selected policies.

Remaining expected cost uses FN=5 for low-risk defaults plus FP=1 for high-risk non-defaults. Manual-review cases are assumed deferred to human review rather than automatic accept/reject.

## Best Policy Per Model

| Model | Constraint | t_low | t_high | Manual Review Rate | High-Risk Precision | High-Risk Recall | Low-Risk Default Rate | Cost | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Best CatBoost | C_manual_review_rate_0_30 | 0.1300 | 0.3600 | 0.2699 | 0.6760 | 0.8667 | 0.1048 | 492 | More defensible screening policy: precision and FP improve with bounded review load. |
| Best Scorecard | C_manual_review_rate_0_30 | 0.1600 | 0.3900 | 0.2699 | 0.6834 | 0.8463 | 0.1124 | 498 | More defensible screening policy: precision and FP improve with bounded review load. |
| Best XGBoost | C_manual_review_rate_0_30 | 0.0900 | 0.3300 | 0.2754 | 0.6585 | 0.8911 | 0.1750 | 510 | More defensible screening policy: precision and FP improve with bounded review load. |
| SCRE-Optimized | C_manual_review_rate_0_30 | 0.2200 | 0.4300 | 0.2805 | 0.7006 | 0.8424 | 0.1243 | 485 | More defensible screening policy: precision and FP improve with bounded review load. |
| SCRE-Pareto | C_manual_review_rate_0_30 | 0.1200 | 0.3400 | 0.2795 | 0.6566 | 0.8891 | 0.0000 | 478 | More defensible screening policy: precision and FP improve with bounded review load. |

## Overall Validation-Selected Recommendation

- Model: Best Scorecard
- Constraint: C_manual_review_rate_0_30 (manual_review_rate <= 0.30)
- Validation thresholds: t_low=0.16, t_high=0.39
- Test manual review rate: 0.2699
- Test high-risk precision: 0.6834
- Test high-risk recall: 0.8463
- Test remaining expected cost: 498
