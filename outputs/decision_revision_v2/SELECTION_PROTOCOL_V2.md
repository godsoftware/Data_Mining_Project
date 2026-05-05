# Selection Protocol V2

Generated: 2026-05-05 11:10:36
Project root: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project`
Archived prior decision revision outputs: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project\outputs\decision_revision_archive_20260505_111036`

## 1. Problem

The previous final decision audit returned **NOT READY** because the final revised comparison tables ranked candidate systems using held-out test-set evaluation metrics. The resulting numbers are useful as retrospective diagnostic evidence, but they are not valid as a clean validation-only model-selection protocol.

Test-set ranking is a methodological problem because the test set must represent the final untouched estimate of generalization. If candidate models, thresholds, manual-review bands, or operational winners are selected after looking at test precision, recall, specificity, false positives, cost, or manual-review results, the reported test performance becomes optimistic selection evidence rather than a final evaluation.

## 2. Allowed Evidence For Selection

Only the following evidence may be used to choose candidate policies and final winners:

- Validation metrics.
- Validation expected cost.
- Validation precision/specificity/recall constraints.
- Validation manual-review operational cost.
- Validation manual-review workload constraints such as manual-review rate and auto-decision rate.
- Pre-specified business constraints documented before test evaluation.

## 3. Forbidden Evidence For Selection

The following are forbidden for candidate selection, threshold selection, band selection, final ranking, or final winner choice:

- Test accuracy.
- Test precision.
- Test recall.
- Test specificity.
- Test F1.
- Test ROC-AUC or PR-AUC.
- Test Brier or ECE.
- Test expected cost.
- Test false positives, false negatives, true positives, or true negatives.
- Test manual-review result.
- Test net benefit.
- Any all-candidate test table used as winner evidence.

## 4. Test Set Usage

The test set may be used only after the final policy is locked from validation evidence. The allowed sequence is:

1. Generate model/policy candidates from training and validation only.
2. Select thresholds, cost-matrix policy, manual-review band, and final winner using validation only.
3. Write a locked policy record before opening test results.
4. Evaluate only the locked final policy on the test set.
5. If all-candidate test tables are shown, label them **diagnostic retrospective evidence only**, never **selection evidence**.

## 5. Manual-Review Cost Rule

Manual-review cost is not the same as binary FN/FP expected cost. A manual-review policy has three outcomes: low risk, manual review, and high risk. The current `remaining_expected_cost` counts low-risk false negatives plus high-risk false positives, but it does not monetize manual-review workload, reviewer capacity, review latency, or downstream override decisions.

Therefore:

- Binary expected cost and manual-review remaining cost must be reported as separate quantities.
- Manual-review workload cost must be modeled explicitly if it is used for operational cost ranking.
- Manual-review policies cannot be declared cheaper than binary policies unless the cost definition is stated clearly.

## 6. Final Readiness Criteria

The project becomes ready for final reporting only when all of the following are true:

- No test-based ranking is used for model or policy selection.
- A validation-only selection log exists.
- A locked final policy exists before test evaluation.
- Test evaluation is run only after locking the final policy.
- All-candidate test tables, if retained, are explicitly marked diagnostic retrospective evidence.
- Manual-review cost scope is clearly separated from binary FN/FP expected cost.
