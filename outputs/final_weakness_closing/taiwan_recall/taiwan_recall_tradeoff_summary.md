# Taiwan Recall-Constrained Policy Search

## Protocol
- Saved model used: `outputs\models\calibrated\taiwan\catboost_isotonic.joblib`
- Selection split: validation only
- Test usage: locked evaluation of the top-3 validation-selected policies only
- New model training: no
- New feature engineering: no
- Test-set threshold selection: no
- H constraint uses V2 validation cost, not V2 test cost, to avoid test-based selection.

## Locked test candidates
| policy_type | constraint_name | review_cost | t_low | t_high | test_recall | test_precision | test_specificity | test_fp | test_fn | test_cost | delta_cost_vs_v2_same_review_cost | test_manual_review_rate | operational_comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| manual_review_band | G_recall_ge_0_60_mr_rate_le_0_30 | 0.100 | 0.125 | 0.215 | 0.633 | 0.463 | 0.791 | 976 | 487 | 1993.2 | -3.9 | 0.295 | Recall improves with bounded FP and manual-review workload; candidate worth audit review. |
| manual_review_band | G_recall_ge_0_60_mr_rate_le_0_30 | 0.100 | 0.130 | 0.235 | 0.619 | 0.480 | 0.809 | 891 | 505 | 2025.6 | 28.5 | 0.291 | Recall improves with bounded FP and manual-review workload; candidate worth audit review. |
| manual_review_band | G_recall_ge_0_60_mr_rate_le_0_30 | 0.100 | 0.130 | 0.230 | 0.622 | 0.478 | 0.807 | 901 | 502 | 2034.3 | 37.2 | 0.289 | Recall improves with bounded FP and manual-review workload; candidate worth audit review. |

## Questions
1. Can Taiwan recall move above 0.60 from the V2 level of 0.575? **YES**. Best locked-test recall is 0.633.
2. What is the FP cost of that recall gain? 976 FP versus V2 680 FP.
3. How much does the manual-review rate change? 0.295 versus V2 0.295.
4. How much does the review-adjusted cost change? 1993.2; same-review-cost delta versus V2 is -3.9.
5. Is the recall gain operationally valuable? **CONDITIONAL**. conditionally useful.
6. Should V2 change now? **V2 should remain final for now** unless the candidate passes a later final audit and improves the replacement-rule criteria.

## Validation search volume
- Feasible validation rows written: 16290
- Locked test rows written: 3
