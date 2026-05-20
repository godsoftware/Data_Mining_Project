# Final V2 vs V3 Selection Summary

## Decision rule
V2 can be replaced only if a candidate is validation-only selected, metric-consistent, audit-ready, and improves the requested weakness without introducing unsafe claims.

## Taiwan selection table
| track | candidate | precision | recall | specificity | fp | fn | cost | manual_review_rate | selection_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_best_final_operational_policy | V2 CatBoost manual-review | 0.529 | 0.575 | 0.854 | 680.0 | 564.0 | 2705.5 | 0.295 | SELECTED_FINAL_VERSION |
| B_best_recall_improving_policy | Taiwan recall-constrained CatBoost manual-review | 0.463 | 0.633 | 0.791 | 976.0 | 487.0 | 1993.2 | 0.295 | V3_AUDIT_CANDIDATE_NOT_FINAL |
| C_best_workload_reducing_policy | Segment-aware utilization_terciles | 0.483 | 0.569 | 0.827 | 809.0 | 572.0 | 3400.0 | 0.124 | APPENDIX_ONLY_WORKLOAD_REDUCTION_CANDIDATE |
| E_best_cost_stable_policy | V2 Taiwan CatBoost manual-review | 0.529 | 0.575 | 0.854 | 680.0 | 564.0 | 2705.5 | 0.295 | SUPPORTS_KEEPING_V2 |
| F_best_scre_review_prioritization_role | SCRE-Optimized Top-20% review ranking |  |  |  | nan | nan |  | 0.200 | FRAMEWORK_ROLE_NOT_FINAL_CLASSIFIER |
| G_best_appendix_only_result | Pseudo/order-based temporal diagnostic |  |  |  | nan | nan |  |  | LIMITATION_ONLY |

## HELOC selection table
| track | candidate | precision | recall | specificity | fp | fn | cost | manual_review_rate | selection_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_best_final_operational_policy | V2 Scorecard manual-review | 0.683 | 0.846 | 0.574 | 403.0 | 158.0 | 764.5 | 0.270 | SELECTED_FINAL_VERSION |
| D_best_heloc_specificity_policy | HELOC specificity-focused Scorecard threshold | 0.724 | 0.782 | 0.676 | 307.0 | 224.0 | 1427.0 | 0.000 | APPENDIX_ONLY_SPECIFICITY_CANDIDATE |
| C_best_workload_reducing_policy | Segment-aware revolving_burden_terciles | 0.692 | 0.758 | 0.635 | 346.0 | 249.0 | 866.5 | 0.238 | APPENDIX_ONLY_WORKLOAD_SPECIFICITY_CANDIDATE |
| E_best_cost_stable_policy | V2 HELOC Scorecard manual-review | 0.683 | 0.846 | 0.574 | 403.0 | 158.0 | 764.5 | 0.270 | SUPPORTS_KEEPING_V2 |
| F_best_scre_review_prioritization_role | SCRE-Optimized Top-20% review ranking |  |  |  | nan | nan |  | 0.200 | FRAMEWORK_ROLE_NOT_FINAL_CLASSIFIER |
| G_best_appendix_only_result | Pseudo/order-based temporal diagnostic |  |  |  | nan | nan |  |  | LIMITATION_ONLY |

## Questions
1. V2 kalıyor mu? **YES.** V2 remains the final operational version for both Taiwan and HELOC.
2. V3 var mı? **PARTIAL.** There are V3 candidates, but no audit-ready replacement.
3. Taiwan'da recall zayıflığı kapandı mı? **PARTIALLY.** The recall-constrained candidate reaches recall above 0.60, but precision/specificity drop and it is not audit-ready.
4. Manual-review workload azaldı mı? **PARTIALLY.** Segment/capacity alternatives reduce workload, but they trade off recall/cost and remain appendix/V3 candidates.
5. HELOC specificity iyileşti mi? **YES as candidate evidence.** Specificity-focused Scorecard reaches 0.676, but cost increases substantially; V2 remains final.
6. SCRE daha güçlü framework oldu mu? **YES.** SCRE is stronger as a review-prioritization and reliability-aware framework, not as a final classifier.
7. Real deployment / cost eksikleri ne kadar kapandı? **Only partially.** Cost and instance-dependent sensitivity improve transparency; temporal diagnostics remain limitation-only because no real timestamp exists.
8. Hangi sonuç ana rapora, hangisi appendix'e gidecek? Main report: V2 frozen final results, V2 audit readiness, safe claims, SCRE framework role. Appendix: recall candidate, specificity candidate, capacity review, instance-dependent cost, segment-aware policy, temporal limitation, Final Attempt metric repair.

## Final committee decision
- Taiwan final operational policy: **V2 CatBoost manual-review**.
- HELOC final operational policy: **V2 Scorecard manual-review**.
- V3 status: **candidate-only, not final**.
- Ready for final audit: **YES**.
