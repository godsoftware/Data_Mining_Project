# Validation-Only Selection Summary

Generated: 2026-05-05 14:46:39

## Protocol Guardrail

No test-set metric columns were read or used. Candidate ranking uses validation-only registry rows and validation-only manual-review cost model rows.

## TAIWAN Selected Policies

| selection_track | model | policy_type | threshold | t_low | t_high | validation_precision | validation_recall | validation_specificity | validation_expected_cost_binary | review_adjusted_cost | manual_review_rate | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_binary_policy | Old SCRE-Credit | precision_constrained_threshold | 0.2000 |  |  | 0.4118 | 0.7197 | 0.7081 | 3224.0 |  |  | Best binary policy by validation expected cost, PR-AUC, ECE, F1, FP. |
| best_manual_review_policy | Best CatBoost | manual_review_band |  | 0.1400 | 0.2800 | 0.5288 | 0.5667 | 0.8566 |  | 2650.5 | 0.2968 | Best manual-review policy by validation review-adjusted cost with review_cost=0.5. |
| best_interpretable_policy | Best Scorecard | manual_review_band |  | 0.1400 | 0.1900 | 0.4332 | 0.6624 | 0.7539 |  | 2993.5 | 0.1945 | Scorecard retained as interpretable benchmark; selected from validation manual-review candidates under corrected workload cost. |
| best_research_framework_policy | SCRE-Pareto | manual_review_band |  | 0.1400 | 0.2500 | 0.5092 | 0.5840 | 0.8401 |  | 2651.5 | 0.2982 | SCRE retained as reliability-aware research framework; selected from revised SCRE validation manual-review candidates. |

Locked final policy: `outputs\decision_revision_v2\locked_final_policy_taiwan.json`

## HELOC Selected Policies

| selection_track | model | policy_type | threshold | t_low | t_high | validation_precision | validation_recall | validation_specificity | validation_expected_cost_binary | review_adjusted_cost | manual_review_rate | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_binary_policy | SCRE-Optimized | SCRE_policy | 0.2200 |  |  | 0.5694 | 0.9825 | 0.1943 | 853.0 |  |  | Best binary policy by validation expected cost, PR-AUC, ECE, F1, FP. |
| best_manual_review_policy | Best Scorecard | manual_review_band |  | 0.1600 | 0.3900 | 0.7010 | 0.8423 | 0.6103 |  | 729.0 | 0.2938 | Best manual-review policy by validation review-adjusted cost with review_cost=0.5. |
| best_interpretable_policy | Best Scorecard | manual_review_band |  | 0.1600 | 0.3900 | 0.7010 | 0.8423 | 0.6103 |  | 729.0 | 0.2938 | Scorecard retained as interpretable benchmark; selected from validation manual-review candidates under corrected workload cost. |
| best_research_framework_policy | SCRE-Optimized | manual_review_band |  | 0.2200 | 0.4300 | 0.7024 | 0.8228 | 0.6220 |  | 732.5 | 0.2882 | SCRE retained as reliability-aware research framework; selected from revised SCRE validation manual-review candidates. |

Locked final policy: `outputs\decision_revision_v2\locked_final_policy_heloc.json`

## Cost Comparability Note

Binary expected cost and manual-review review-adjusted cost are not directly ranked against each other. The locked operational policy is manual-review based because the current project objective is false-positive reduction and screening/manual-review support. Binary winner remains reported as a separate benchmark.