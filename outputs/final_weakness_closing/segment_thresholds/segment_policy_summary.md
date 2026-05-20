# Segment-Aware Threshold Policy Summary

## Protocol
- Existing calibrated scores only.
- Segment boundaries are learned from the non-test train/validation pool.
- Segment policies are selected on validation only.
- Locked-test rows are evidence only and contain no rank/winner selection from test.
- Segment-aware thresholding has medium/high overfitting risk because multiple policies are selected from one validation split.

## Taiwan locked-test selected segment policies
| Segment Strategy | Precision | Recall | Specificity | FP | FN | Cost | MR Rate | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| recent_delay_groups | 0.669 | 0.370 | 0.948 | 243 | 836 | 2508.5 | 0.473 | Locked test evaluation of a validation-selected segment policy; no test-set policy selection. |
| limit_bal_terciles | 0.489 | 0.560 | 0.834 | 776 | 584 | 2989.0 | 0.214 | Locked test evaluation of a validation-selected segment policy; no test-set policy selection. |
| utilization_terciles | 0.483 | 0.569 | 0.827 | 809 | 572 | 3400.0 | 0.124 | Locked test evaluation of a validation-selected segment policy; no test-set policy selection. |

Validation constraint status: `limit_bal_terciles` and `utilization_terciles` had passing validation candidates. `recent_delay_groups` did not pass the full Taiwan validation constraints and is diagnostic only despite the locked-test row.

## HELOC locked-test selected segment policies
| Segment Strategy | Precision | Recall | Specificity | FP | FN | Cost | MR Rate | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| revolving_burden_terciles | 0.692 | 0.758 | 0.635 | 346 | 249 | 866.5 | 0.238 | Locked test evaluation of a validation-selected segment policy; no test-set policy selection. |
| percent_never_delq_terciles | 0.872 | 0.286 | 0.955 | 43 | 734 | 871.0 | 0.788 | Locked test evaluation of a validation-selected segment policy; no test-set policy selection. |
| external_risk_terciles | 0.865 | 0.287 | 0.951 | 46 | 733 | 875.0 | 0.784 | Locked test evaluation of a validation-selected segment policy; no test-set policy selection. |

Validation constraint status: only `revolving_burden_terciles` passed the HELOC validation constraints. `external_risk_terciles` and `percent_never_delq_terciles` failed validation constraints because manual-review rate became too high and recall fell too low; they are diagnostic only.

## Answers
1. Segment-aware threshold V2'den iyi mi? **Not enough to replace V2.** Some segment policies improve one dimension but remain higher-complexity V3/appendix candidates.
2. Taiwan recall arttı mı? **NO.** Compare locked-test recall rows above against V2 recall 0.575.
3. HELOC specificity arttı mı? **PARTIALLY.** Compare locked-test specificity rows above against V2 specificity 0.574.
4. MR rate kabul edilebilir mi? **PARTIALLY.** Taiwan valid candidates are mostly within reasonable MR ranges, and the HELOC `revolving_burden_terciles` valid candidate has locked-test MR=0.238. HELOC `external_risk_terciles` and `percent_never_delq_terciles` are not acceptable because locked-test MR is about 0.78 and validation constraints failed.
5. Segment policy overfitting riski taşıyor mu? **YES.** Multiple segment-specific thresholds/bands are selected on a single validation split.
6. Rapor için kullanılabilir mi, yoksa appendix mi? **Appendix / V3 audit candidate.** It should not replace V2 without final audit.

## Candidate for V3
APPENDIX ONLY
