# HELOC Specificity-Focused Scorecard Optimization

## Protocol
- Existing scorecard artifacts only: uncalibrated, sigmoid, isotonic.
- Selection split: validation only.
- Test usage: locked evaluation of top-3 validation-selected policies only.
- No new dataset, feature, model training, or test-set threshold selection.
- A8 uses V2 validation cost, not V2 test cost, to avoid test-based selection.

## Locked test candidates
| policy_type | calibration_type | constraint_name | threshold | t_low | t_high | test_specificity | test_recall | test_precision | test_fp | test_fn | test_cost | test_manual_review_rate | operational_comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| threshold_only | sigmoid | A5_specificity_ge_0_65_recall_ge_0_75 | 0.435 |  |  | 0.621 | 0.815 | 0.700 | 359 | 190 | 1309.0 | 0.000 | Specificity gain is limited. |
| threshold_only | uncalibrated | A6_specificity_ge_0_70_recall_ge_0_70 | 0.460 |  |  | 0.676 | 0.782 | 0.724 | 307 | 224 | 1427.0 | 0.000 | Specificity improves while recall remains reasonably high; V3 audit candidate. |
| threshold_only | sigmoid | A7_specificity_ge_0_65_precision_ge_0_70 | 0.440 |  |  | 0.631 | 0.811 | 0.705 | 349 | 194 | 1319.0 | 0.000 | Specificity gain is limited. |

## Calibration comparison
- Best validation Brier/ECE calibration: sigmoid
- Validation Brier: 0.1804
- Validation ECE: 0.0205

## Questions
1. Can HELOC specificity move above 0.65 from V2 0.574? **YES**. Best locked-test specificity is 0.676.
2. How much does recall drop? Best-specificity recall drop is 0.064.
3. Does precision improve? Precision change for best-specificity candidate is 0.040.
4. Does cost increase? Cost change for best-specificity candidate is 662.5.
5. Did calibration help? **YES** if judged by validation Brier/ECE; see `heloc_scorecard_calibration_comparison.csv`.
6. Should Scorecard remain final? **YES for V2 final evidence** until a V3 audit accepts the replacement.
7. Is a HELOC V3 policy recommended? **YES, as audit candidate**.

## Validation search volume
- Feasible validation rows written: 122482
- Locked test rows written: 3
