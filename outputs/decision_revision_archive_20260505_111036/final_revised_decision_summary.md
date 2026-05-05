# Final Revised Decision Summary

## 1. What was wrong with the old cost-threshold model?

Taiwan old CatBoost cost-threshold had precision=0.3514, recall=0.8093, specificity=0.5759, FP=1982. It captured defaults, but generated too many false positives for automatic rejection.

## 2. Did precision-constrained thresholding help?

Yes. CatBoost precision-constrained threshold improved precision to 0.4177 and reduced FP to 1273, with recall=0.6880.

## 3. Did cost sensitivity change the winner?

Yes. Increasing FP cost moves thresholds upward, improves precision/specificity, and changes the winner in several scenarios. FN=5, FP=1 is too aggressive as the only operational assumption.

## 4. Did manual review band improve operational usability?

Yes, conditionally. CatBoost bounded manual-review band reduced high-risk FP to 680 with precision=0.5288 and manual_review_rate=0.2952. Manual-review capacity is the key operational constraint.

## 5. Did class weighting help?

Partially. Best Taiwan imbalance-training row reached precision=0.4012, recall=0.7159, FP=1418. It is a useful binary alternative, but not enough to replace manual-review decision support.

## 6. Are false positives truly risky-looking customers?

Mostly yes. FP profiles are closer to TP than TN across the analyzed systems, especially for delay and repayment behavior. This supports review/screening, not automatic rejection.

## 7. What is the final Taiwan recommendation?

Final Taiwan operational screening policy: CatBoost manual review band (manual_review_band:C_manual_review_rate_0_30, t_low=0.14; t_high=0.28). Use CatBoost precision-constrained or eligible binary SCRE/LightGBM only as fallback when manual review is unavailable.

## 8. What is the final HELOC recommendation?

Final HELOC external-validation policy: CatBoost manual review band (manual_review_band:C_manual_review_rate_0_30, t_low=0.13; t_high=0.36). Scorecard remains the interpretable benchmark.

## 9. Should the model be used for automatic decisioning?

No. The final evidence does not justify fully automatic rejection. Precision and false-positive behavior require human review for high-risk cases.

## 10. Should the model be used for screening/manual review?

Yes. The revised policies are much stronger as screening and manual-review prioritization tools.

## 11. What should be claimed?

- Revised decision policies reduce false-positive burden and improve operational usability.
- Manual-review framing is more defensible than automatic rejection.
- SCRE remains useful as a reliability-aware framework.
- Scorecard remains an interpretable benchmark.

## 12. What should not be claimed?

- Do not claim the model is suitable for fully automatic rejection.
- Do not claim SCRE universally beats all single models.
- Do not claim false positives are harmless; they are risk-like and need review.
- Do not present FN=5, FP=1 as the only valid operational cost assumption.
