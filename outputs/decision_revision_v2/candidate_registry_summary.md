# Validation Candidate Registry Summary

Generated: 2026-05-05 11:43:39

## Counts

| dataset | total | binary_threshold | manual_review | eligible_basic | eligible_strict | invalid |
| --- | --- | --- | --- | --- | --- | --- |
| taiwan | 250 | 180 | 70 | 72 | 63 | 2 |
| heloc | 250 | 180 | 70 | 111 | 88 | 1 |

## Audit

1. Total candidates: 500
2. Binary threshold candidates: 360
3. Manual-review candidates: 140
4. Eligible basic: 183
5. Eligible strict: 151
6. Test metric columns present: NO
7. Invalid candidates with missing validation metrics: 3
8. Manual-review adjusted cost missing: YES; missing rows=140

## Invalid Candidates

| candidate_id | dataset | model | policy_type | constraint_name | invalid_reason |
| --- | --- | --- | --- | --- | --- |
| heloc__cand_0147 | heloc | Best imbalance variant | imbalance_training_policy | default_0_50 | validation_accuracy;validation_tn;validation_fp;validation_fn;validation_tp |
| taiwan__cand_0397 | taiwan | Best imbalance variant | imbalance_training_policy | precision_0_40 | validation_accuracy;validation_tn;validation_fp;validation_fn;validation_tp |
| taiwan__cand_0420 | taiwan | Old SCRE-Credit | precision_constrained_threshold | L_precision_0_45_recall_0_60 | validation_accuracy;validation_precision;validation_recall;validation_specificity;validation_f1;validation_roc_auc;validation_pr_auc;validation_brier;validation_ece;validation_tn;validation_fp;validation_fn;validation_tp;validation_expected_cost_binary |

## Notes

- Existing test-derived eligible ranking files were not used as selection evidence.
- Manual-review policies keep binary expected cost separate from review-adjusted cost. Because no explicit review_cost is available, validation_review_adjusted_cost is NA.
- Manual-review rows use validation high-risk bucket precision/recall and derived high-risk-vs-not-high specificity for eligibility screening; they are still not final winners until a validation-only cost rule is defined.