# Manual Review Band Optimization Summary: TAIWAN

Low/high thresholds were selected on validation only. Test metrics report the held-out performance of those selected policies.

Remaining expected cost uses FN=5 for low-risk defaults plus FP=1 for high-risk non-defaults. Manual-review cases are assumed deferred to human review rather than automatic accept/reject.

## Best Policy Per Model

| Model | Constraint | t_low | t_high | Manual Review Rate | High-Risk Precision | High-Risk Recall | Low-Risk Default Rate | Cost | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Best CatBoost | C_manual_review_rate_0_30 | 0.1400 | 0.2800 | 0.2952 | 0.5288 | 0.5750 | 0.0818 | 1820 | More defensible screening policy: precision and FP improve with bounded review load. |
| Best Scorecard | C_manual_review_rate_0_30 | 0.1500 | 0.3400 | 0.2917 | 0.5477 | 0.4755 | 0.0972 | 2026 | More defensible screening policy: precision and FP improve with bounded review load. |
| Best XGBoost | C_manual_review_rate_0_30 | 0.1300 | 0.2200 | 0.2942 | 0.4787 | 0.6277 | 0.0762 | 1857 | More defensible screening policy: precision and FP improve with bounded review load. |
| SCRE-Optimized | C_manual_review_rate_0_30 | 0.1600 | 0.3600 | 0.3018 | 0.5903 | 0.4680 | 0.0944 | 1911 | Precision and FP improve, but manual-review workload needs operational review. |
| SCRE-Pareto | C_manual_review_rate_0_30 | 0.1400 | 0.2500 | 0.2902 | 0.5061 | 0.5938 | 0.0799 | 1849 | More defensible screening policy: precision and FP improve with bounded review load. |

## Overall Validation-Selected Recommendation

- Model: SCRE-Optimized
- Constraint: C_manual_review_rate_0_30 (manual_review_rate <= 0.30)
- Validation thresholds: t_low=0.16, t_high=0.36
- Test manual review rate: 0.3018
- Test high-risk precision: 0.5903
- Test high-risk recall: 0.4680
- Test remaining expected cost: 1911

## Taiwan Operational Questions

1. High-risk precision above old binary precision 0.351: YES (0.5288).
2. Manual review rate realistic: YES (0.2952).
3. Low-risk default rate acceptable under 0.10 target: YES (0.0818).
4. FP reduced vs binary threshold: YES (1302).
5. Cost deterioration vs binary threshold: NO (cost reduction 1427).
6. Defensible framing: YES, as screening + manual review; not as automatic rejection.
