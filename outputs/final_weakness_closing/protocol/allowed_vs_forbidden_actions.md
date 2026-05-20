# Allowed vs Forbidden Actions

This weakness-closing package is not a new model-hunting phase. It is a controlled validation and policy-cleanup phase built on existing V2 and existing score/probability outputs.

## Forbidden Actions

- Train a new model family.
- Add a new dataset.
- Create new target-derived features.
- Use the test set for model selection.
- Use the test set for threshold selection.
- Use the test set for manual-review band selection.
- Use the test set for calibration fitting.
- Apply SMOTE/resampling before split.
- Apply resampling to the test set.
- Ignore the Final Attempt `BLOCKED` audit result.
- Present SCRE-Credit as a universally superior classifier.
- Claim automatic credit rejection suitability.
- Replace V2 using diagnostic-only or audit-blocked evidence.

## Allowed Actions

- Existing-score validation-only threshold or band analysis.
- Manual-review metric repair and denominator clarification.
- Capacity-aware review policy analysis.
- Cost sensitivity analysis.
- Decision curve / net benefit analysis.
- SCRE top-k prioritization analysis.
- Instance-dependent cost sensitivity.
- Segment-aware threshold policy, if based only on existing variables and validation-only selection.
- Temporal or pseudo-temporal robustness, if the data permits and no test-selection leakage is introduced.
- Final audit of any candidate before replacement.

## Evidence Rules

- Validation evidence may be used for selection.
- Locked test evidence may be used only after the candidate policy is fixed.
- Test metrics must not create ranks or winners.
- Final Attempt locked-test manual-review rows remain non-final while the audit is `BLOCKED`.
- Any new candidate must be compared against the frozen V2 reference in `v2_reference_freeze.csv`.

