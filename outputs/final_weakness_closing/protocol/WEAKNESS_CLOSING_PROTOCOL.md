# Weakness Closing Protocol

## 1. Objective

This is the final weakness-closing attempt before moving to the conclusion phase.

The objective is not to chase new models. The objective is to examine whether the current V2 final system can be improved in a controlled, audit-safe way using existing evidence, existing scores, and validation-only policy analysis.

The current final operational version remains V2 unless a new candidate satisfies the replacement rule defined below.

## 2. Current V2 Final Status

### Taiwan

- Final policy: V2 CatBoost manual-review
- Precision: 0.529
- Recall: 0.575
- Specificity: 0.854
- FP: 680
- FN: 564
- Cost: 2705.5
- Manual review rate: 0.295
- Audit status: READY

### HELOC

- Final policy: V2 Scorecard manual-review
- Precision: 0.683
- Recall: 0.846
- Specificity: 0.574
- FP: 403
- FN: 158
- Cost: 764.5
- Manual review rate: 0.270
- Audit status: READY

V2 is currently frozen as the final operational evidence in `outputs/final_frozen_v2/`.

## 3. Weaknesses to Be Addressed

1. Taiwan recall is low-to-moderate: 0.575.
2. Manual review rate is high: Taiwan approximately 0.295.
3. HELOC specificity is limited: 0.574.
4. Final Attempt audit is BLOCKED.
5. SCRE-Credit is not a dominant classifier.
6. There is no real deployment evidence.
7. There is no real bank-specific cost matrix.
8. There is no richer financial/behavioral banking dataset.

These weaknesses should be treated as limitations unless a controlled analysis produces audit-ready improvement.

## 4. Forbidden Operations

- Add a new dataset.
- Train a new model family.
- Create new target-derived features.
- Use the test set for model selection.
- Use the test set for threshold selection.
- Use the test set for calibration fitting.
- Use the test set for manual-review band selection.
- Apply SMOTE or any resampling before split.
- Apply resampling to the test set.
- Ignore the Final Attempt `BLOCKED` audit result.
- Present SCRE-Credit as a model that beats everything.
- Produce an automatic rejection claim.
- Promote any result to final without a final audit.

## 5. Allowed Experiments

- Existing-score validation-only threshold / band analysis.
- Manual-review metric repair and denominator clarification.
- Capacity-aware review policy analysis.
- Cost sensitivity analysis.
- Decision curve / net benefit analysis.
- SCRE top-k prioritization analysis.
- Instance-dependent cost sensitivity.
- Segment-aware threshold policy, if based on existing variables and validation-only selection.
- Temporal or pseudo-temporal robustness, if the dataset permits.
- Final audit after any proposed replacement.

## 6. Validation-Only Selection Rule

Any candidate policy must be selected using validation evidence only.

Allowed validation evidence includes:

- validation precision
- validation recall
- validation specificity
- validation expected cost
- validation review-adjusted cost
- validation manual-review rate
- validation capacity metrics
- validation net benefit

Forbidden selection evidence includes:

- test precision
- test recall
- test specificity
- test cost
- test manual-review rate
- test net benefit
- any test-derived rank or winner

## 7. Locked-Test Rule

After a candidate is selected on validation only:

1. The model/policy must be locked.
2. Thresholds or bands must be frozen.
3. Calibration must not be changed.
4. Test set evaluation may be run once as held-out evidence.
5. The policy must not be changed after test results are seen.
6. Test results may not be used to create candidate ranks.

## 8. Final Replacement Rule

V2 can be replaced only if all conditions below are satisfied:

1. The new candidate is selected using validation-only evidence.
2. Locked test evaluation is performed after selection.
3. Final audit verdict is READY.
4. No metric-consistency failure exists.
5. No test-selection, threshold-selection, calibration, or resampling leakage exists.
6. For Taiwan, the candidate meaningfully improves at least one of:
   - recall
   - manual-review workload
   - cost
7. For HELOC, the candidate meaningfully improves at least one of:
   - specificity
   - cost
   - manual-review usability
8. Any improvement must not be achieved by creating an unsafe automatic rejection claim.
9. SCRE-Credit may be promoted only if it is audit-ready and the claim remains framework/review-prioritization oriented, not universal dominance.

Until every condition is satisfied, V2 remains the final operational version.

## Protocol Decision

Ready for metric repair: **YES**

