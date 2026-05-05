# Locked Test Evaluation Summary

Generated: 2026-05-05 15:02:50

## 1. Was The Policy Selected Before Seeing Test Results?

Yes. The locked policies were read from `locked_final_policy_taiwan.json` and `locked_final_policy_heloc.json`. No model, calibration, threshold, or band was changed after opening the test set.

## TAIWAN Locked Test Result

- Model: Best CatBoost
- Policy: manual_review_band
- Threshold/band: threshold=NA, t_low=0.1400, t_high=0.2800
- Precision: 0.5288
- Recall: 0.5750
- Specificity: 0.8545
- FP/FN: 680/564
- Binary diagnostic cost: 3500.0
- Review-adjusted cost: 2705.5
- Manual-review rate: 0.2952

## HELOC Locked Test Result

- Model: Best Scorecard
- Policy: manual_review_band
- Threshold/band: threshold=NA, t_low=0.1600, t_high=0.3900
- Precision: 0.6834
- Recall: 0.8463
- Specificity: 0.5744
- FP/FN: 403/158
- Binary diagnostic cost: 1193.0
- Review-adjusted cost: 764.5
- Manual-review rate: 0.2699

## 4. Comparison To Old Aggressive Cost-Threshold

### TAIWAN

- Old CatBoost precision/recall/specificity: 0.3514/0.8093/0.5759
- Locked precision/recall/specificity: 0.5288/0.5750/0.8545
- FP delta: -1302
- Cost delta vs old CatBoost: -541.5

### HELOC

- Old CatBoost precision/recall/specificity: 0.5581/0.9805/0.1573
- Locked precision/recall/specificity: 0.6834/0.8463/0.5744
- FP delta: -395
- Cost delta vs old CatBoost: -133.5

## 5. Did FP Reduce?

Yes for both locked policies, relative to the old aggressive CatBoost cost-threshold reference.

## 6. Did Precision Improve?

Yes for both locked policies.

## 7. Did Recall Remain Acceptable?

Recall decreased relative to the old aggressive threshold, as expected. It remains a screening/manual-review trade-off rather than an automatic rejection result.

## 8. Is This Suitable For Automatic Rejection?

No. The locked policies should not be presented as automatic rejection systems.

## 9. Is This Suitable For Screening/Manual Review?

Yes. The locked policies are suitable to discuss as screening/manual-review decision-support policies under explicit review-cost assumptions.

## 10. Final Test Interpretation

The validation-locked policies reduce false-positive burden and improve precision on held-out test evidence. The result is now methodologically cleaner because test results were used only after locking the policy.