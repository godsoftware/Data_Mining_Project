# Capacity-Aware Review Weakness Closing

## Protocol
- Existing fitted models/probability rankers only.
- All model/capacity candidates are computed on validation.
- Held-out test is evaluated only for policies locked from validation.
- No new model training, feature engineering, threshold learning on test, or resampling.

## TAIWAN validation top-20 ranking

| model | top_10_capture | top_20_capture | top_25_capture | top_30_capture | precision_at_20 | lift_at_20 |
| --- | --- | --- | --- | --- | --- | --- |
| SCRE-Optimized probability ranking | 0.321 | 0.529 | 0.589 | 0.647 | 0.585 | 2.645 |
| SCRE-Pareto probability ranking | 0.318 | 0.526 | 0.589 | 0.651 | 0.582 | 2.630 |
| Best EBM/monotonic model: Monotonic LightGBM | 0.317 | 0.520 | 0.580 | 0.638 | 0.575 | 2.600 |
| Best CatBoost probability ranking | 0.314 | 0.518 | 0.581 | 0.632 | 0.573 | 2.589 |
| Best literature reproduction model | 0.317 | 0.518 | 0.580 | 0.639 | 0.573 | 2.589 |

## TAIWAN locked-test capacity policies

| model | policy_type | capacity_pct | default_capture_rate | precision_at_k | lift_at_k | low_risk_default_rate | review_adjusted_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCRE-Optimized probability ranking | three_bucket_capacity_band | 0.200 | 0.173 | 0.192 | 0.866 | 0.096 | 2920.500 |
| SCRE-Optimized probability ranking | top_k_review | 0.200 | 0.512 | 0.567 | 2.562 | 0.135 | 3835.000 |
| SCRE-Optimized probability ranking | three_bucket_capacity_band | 0.250 | 0.251 | 0.217 | 0.983 | 0.096 | 2858.000 |
| SCRE-Optimized probability ranking | top_k_review | 0.250 | 0.586 | 0.519 | 2.345 | 0.122 | 3495.000 |

## HELOC validation top-20 ranking

| model | top_10_capture | top_20_capture | top_25_capture | top_30_capture | precision_at_20 | lift_at_20 |
| --- | --- | --- | --- | --- | --- | --- |
| SCRE-Optimized probability ranking | 0.182 | 0.340 | 0.419 | 0.500 | 0.884 | 1.698 |
| Best EBM/monotonic model: EBM | 0.169 | 0.338 | 0.408 | 0.481 | 0.878 | 1.689 |
| SCRE-Pareto probability ranking | 0.174 | 0.334 | 0.414 | 0.490 | 0.868 | 1.669 |
| Best Scorecard probability ranking | 0.165 | 0.328 | 0.409 | 0.485 | 0.853 | 1.640 |
| V2 final locked policy ranking | 0.165 | 0.328 | 0.409 | 0.485 | 0.853 | 1.640 |

## HELOC locked-test capacity policies

| model | policy_type | capacity_pct | default_capture_rate | precision_at_k | lift_at_k | low_risk_default_rate | review_adjusted_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCRE-Optimized probability ranking | three_bucket_capacity_band | 0.200 | 0.076 | 0.220 | 0.422 | 0.129 | 800.500 |
| SCRE-Optimized probability ranking | top_k_review | 0.200 | 0.339 | 0.881 | 1.693 | 0.430 | 3597.500 |
| SCRE-Optimized probability ranking | three_bucket_capacity_band | 0.250 | 0.113 | 0.246 | 0.473 | 0.129 | 780.500 |
| SCRE-Optimized probability ranking | top_k_review | 0.250 | 0.411 | 0.856 | 1.645 | 0.409 | 3272.000 |

## Answers

1. Taiwan at 20% capacity: use the validation top-20 table and locked-test rows above; it is a workload-reduction candidate, not an automatic V2 replacement.
2. Taiwan at 25% capacity is closer to the V2 29.5% review load and should be preferred if recall loss at 20% is too high.
3. HELOC at 20% capacity is plausible only if lower review workload is more important than maximum default capture.
4. SCRE ranking is useful only if its validation Lift@K exceeds CatBoost/Scorecard at the same K; check `capacity_model_comparison.csv`.
5. Capacity policy can support or replace the V2 band only after a final audit; for now it is a V3 candidate / appendix evidence.
6. Operational recommendation: discuss 20% and 25% capacity as lower-workload screening alternatives, not automatic rejection policies.

## Availability notes
- All requested non-deep candidate artifacts loaded successfully.
