# SCRE-Credit V2 READY Backup

Created: 2026-05-05 16:35

Project root frozen as V2: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project`

Backup root: `C:\Users\AKTS\Desktop\Resul\project_backups`

V1 reference kept: `C:\Users\AKTS\Desktop\Resul\project_backups\SCRE_Credit_homework_freeze_20260429_1627`

V2 backup target: `C:\Users\AKTS\Desktop\Resul\project_backups\SCRE_Credit_v2_READY_20260505_1633`

## 1. Backup cleanup performed

Before creating V2, `project_backups` contained three near-duplicate homework freezes:

- `SCRE_Credit_homework_freeze_20260429_1625`
- `SCRE_Credit_homework_freeze_20260429_1627`
- `SCRE_Credit_homework_freeze_20260429_1628`

All three had the same core project files, same file count, and same main result hashes. The only differences were audit timestamp/path metadata. I kept the most consistent/latest homework freeze reference:

- Kept: `SCRE_Credit_homework_freeze_20260429_1627`

Deleted as redundant duplicates:

- Deleted: `SCRE_Credit_homework_freeze_20260429_1625`
- Deleted: `SCRE_Credit_homework_freeze_20260429_1628`

## 2. V1 to V2 difference summary

The V1 backup was the homework-ready SCRE-Credit freeze from 2026-04-29. It captured the earlier project state before the later decision-revision V2 methodology cleanup.

The current project is now marked as **V2 READY** because it fixes the previous NOT READY issue: final recommendations are no longer ranked by held-out test metrics.

File-level diff artifacts:

- `C:\Users\AKTS\Desktop\Resul\project_backups\v2_vs_homework_v1_file_diff_20260505_1633.csv`
- `C:\Users\AKTS\Desktop\Resul\project_backups\v2_vs_homework_v1_diff_summary_20260505_1633.csv`
- `C:\Users\AKTS\Desktop\Resul\project_backups\v2_vs_homework_v1_dir_diff_summary_20260505_1633.csv`
- `C:\Users\AKTS\Desktop\Resul\project_backups\v2_important_artifacts_20260505_1633.csv`

Diff status counts:

| status | count |
| --- | --- |
| added_in_current_v2 | 183 |
| removed_from_current_v2 | 3 |
| unchanged | 773 |

Directory-level changed-file summary:

| top_level | status | count |
| --- | --- | --- |
| catboost_info | added_in_current_v2 | 5 |
| experiments | added_in_current_v2 | 9 |
| outputs | added_in_current_v2 | 169 |
| outputs | removed_from_current_v2 | 3 |

Important note: `.git`, `__pycache__`, `.pytest_cache`, and notebook checkpoint/cache files are excluded from the comparison CSV to keep the methodological diff readable. The V2 backup copy itself is a full folder copy and includes hidden project files such as `.git`.

## 3. What changed conceptually from V1 to V2

### 3.1 Previous state in V1

V1 was a homework/report freeze. It already included Taiwan and HELOC cleaning, model training, SCRE-Credit variants, calibration, threshold/cost analysis, XAI, statistical tests, ablation, and homework delivery outputs.

However, later validation revealed a methodological issue: some final comparison/recommendation tables ranked candidate systems using held-out test-set metrics. Those tables were useful as retrospective diagnostics, but not valid for clean final model selection.

### 3.2 Core V2 methodology reset

V2 introduced a strict separation:

1. Candidate generation and model/policy selection use validation evidence only.
2. Threshold/manual-review band selection uses validation evidence only.
3. Locked final policies are written before held-out test evaluation.
4. Held-out test metrics are reported only as final evidence.
5. Held-out test tables contain no candidate ranking and no winner column.
6. Old all-candidate test ranking tables are retained only as diagnostic retrospective evidence and marked not for selection.

## 4. What was tried and what happened

### 4.1 Baseline false-positive problem

The original cost-minimizing Taiwan CatBoost policy used a low threshold around 0.15. It gave high recall but poor precision/specificity and too many false positives. This was unsafe to present as an automatic rejection model.

Outcome: the project shifted from pure cost minimization to false-positive-aware screening and manual-review policy design.

### 4.2 Precision-constrained thresholds

Precision/specificity/FP-budget constrained threshold searches were run without retraining models. Thresholds were selected on validation and evaluated on test only after selection.

Outcome: precision and specificity could be improved, but recall dropped. This made the model more operationally defensible as screening, not automatic rejection.

### 4.3 Cost matrix sensitivity

Multiple cost scenarios were evaluated: FN/FP ratios including `FN=2, FP=1`, `FN=3, FP=1`, `FN=5, FP=1`, `FN=5, FP=2`, `FN=5, FP=3`, and others.

Outcome: increasing FP cost pushes thresholds upward, improves precision/specificity, and lowers FP burden, but recall can fall. This exposed that `FN=5, FP=1` alone was too aggressive for operational discussion.

### 4.4 Decision curve / net benefit

Decision curve analysis was added to ask whether models provide useful decision benefit versus treat-all/treat-none rules.

Outcome: model policies can be useful over ranges of threshold probability, but this supports decision support, not automatic rejection.

### 4.5 Manual-review band optimization

Binary classification was converted into three actions: low risk, manual review, high risk.

Outcome: manual-review policies better manage false positives by moving uncertainty into review. They are more realistic for credit risk operations, but review workload must be priced.

### 4.6 Training-level imbalance revision

CatBoost, LightGBM, and XGBoost class-weighting / imbalance variants were tested. This created additional exploratory evidence.

Outcome: class weighting can improve some recall/precision trade-offs, but it can distort probability calibration. It is not the final selected operational policy.

### 4.7 Eligible model selection

Minimum precision/recall/specificity/PR-AUC/ROC-AUC constraints were applied to decide which systems are realistic enough to discuss.

Outcome: pure cost-minimizing models are not automatically operationally acceptable. Eligibility constraints strengthened the case for manual-review policies.

### 4.8 False-positive error analysis

FP/TP/TN/FN groups were analyzed to see whether false positives looked like genuine risk profiles or pure noise.

Outcome: FPs are generally TP-like risky profiles, not purely harmless alarms. But because they are non-default outcomes, automatic rejection remains unsafe.

### 4.9 Final revised comparison initially failed audit

The first final revised comparison still had a methodological problem: it appeared to rank systems using test metrics.

Outcome: audit verdict was NOT READY. This triggered the V2 reset.

### 4.10 V2 protocol reset

V2 created `outputs/decision_revision_v2` and archived old outputs as diagnostic history.

Outcome: selection protocol now explicitly forbids test-based ranking or policy selection.

### 4.11 Validation-only candidate registry

All candidate model/policy rows were rebuilt with validation metrics only.

Outcome:

| dataset | total | invalid | eligible_basic | eligible_strict |
| --- | --- | --- | --- | --- |
| heloc | 250 | 1 | 111 | 88 |
| taiwan | 250 | 2 | 72 | 63 |

Invalid candidates remained marked, not hidden.

### 4.12 Manual-review cost correction

Manual-review cost was separated into three definitions:

- Automation-only residual cost.
- Workload-adjusted review cost.
- Imperfect-review adjusted cost.

Outcome: manual-review cost is no longer treated as equivalent to binary expected cost.

Top workload-adjusted candidates under `FN=5`, `FP=1`, `review_cost=0.5`:

| dataset | model | source_constraint_name | manual_review_rate | review_adjusted_cost | binary_reference_cost |
| --- | --- | --- | --- | --- | --- |
| heloc | Best Scorecard | C_manual_review_rate_0_30 | 0.2938 | 729 | 871 |
| heloc | Best Scorecard | N_review_0_30_precision_0_45_low_default_0_10 | 0.2938 | 729 | 871 |
| heloc | SCRE-Optimized | C_manual_review_rate_0_30 | 0.2882 | 732.5 | 853 |
| heloc | SCRE-Optimized | N_review_0_30_precision_0_45_low_default_0_10 | 0.2882 | 732.5 | 853 |
| heloc | Best XGBoost | C_manual_review_rate_0_30 | 0.2958 | 734 | 884 |
| taiwan | SCRE-Optimized | C_manual_review_rate_0_30 | 0.2958 | 2640.5 | 3107 |
| taiwan | SCRE-Optimized | N_review_0_30_precision_0_45_low_default_0_10 | 0.2958 | 2640.5 | 3107 |
| taiwan | Best CatBoost | C_manual_review_rate_0_30 | 0.2968 | 2650.5 | 3144 |
| taiwan | Best CatBoost | N_review_0_30_precision_0_45_low_default_0_10 | 0.2968 | 2650.5 | 3144 |
| taiwan | SCRE-Pareto | C_manual_review_rate_0_30 | 0.2982 | 2651.5 | 3176 |

### 4.13 Validation-only final selector

Final winners were selected only from validation evidence.

Selected validation rows:

| dataset | selection_track | candidate_id | model | policy_type | calibration_type | threshold | t_low | t_high | validation_precision | validation_recall | validation_specificity | validation_pr_auc | validation_roc_auc | validation_ece | validation_expected_cost_binary | review_adjusted_cost | manual_review_rate | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | best_binary_policy | taiwan__cand_0410 | Old SCRE-Credit | precision_constrained_threshold | weighted_sum_plus_isotonic | 0.2 |  |  | 0.4118 | 0.7197 | 0.7081 | 0.555 | 0.7909 | 0 | 3224 |  |  | Best binary policy by validation expected cost, PR-AUC, ECE, F1, FP. |
| taiwan | best_manual_review_policy | taiwan__cand_0264 | Best CatBoost | manual_review_band | isotonic |  | 0.14 | 0.28 | 0.5288 | 0.5667 | 0.8566 | 0.5601 | 0.7863 | 0.0059 |  | 2650.5 | 0.2968 | Best manual-review policy by validation review-adjusted cost with review_cost=0.5. |
| taiwan | best_interpretable_policy | taiwan__cand_0328 | Best Scorecard | manual_review_band | isotonic |  | 0.14 | 0.19 | 0.4332 | 0.6624 | 0.7539 | 0.532 | 0.7734 | 0.0129 |  | 2993.5 | 0.1945 | Scorecard retained as interpretable benchmark; selected from validation manual-review candidates under corrected workload cost. |
| taiwan | best_research_framework_policy | taiwan__cand_0475 | SCRE-Pareto | manual_review_band | weighted_sum_of_calibrated_pareto_front_probabilities |  | 0.14 | 0.25 | 0.5092 | 0.584 | 0.8401 | 0.5623 | 0.7888 | 0.01 |  | 2651.5 | 0.2982 | SCRE retained as reliability-aware research framework; selected from revised SCRE validation manual-review candidates. |
| heloc | best_binary_policy | heloc__cand_0173 | SCRE-Optimized | SCRE_policy | weighted_sum_of_calibrated_pareto_front_probabilities_no_final_calibration | 0.22 |  |  | 0.5694 | 0.9825 | 0.1943 | 0.8056 | 0.808 | 0.0454 | 853 |  |  | Best binary policy by validation expected cost, PR-AUC, ECE, F1, FP. |
| heloc | best_manual_review_policy | heloc__cand_0080 | Best Scorecard | manual_review_band | sigmoid |  | 0.16 | 0.39 | 0.701 | 0.8423 | 0.6103 | 0.7951 | 0.8034 | 0.0205 |  | 729 | 0.2938 | Best manual-review policy by validation review-adjusted cost with review_cost=0.5. |
| heloc | best_interpretable_policy | heloc__cand_0080 | Best Scorecard | manual_review_band | sigmoid |  | 0.16 | 0.39 | 0.701 | 0.8423 | 0.6103 | 0.7951 | 0.8034 | 0.0205 |  | 729 | 0.2938 | Scorecard retained as interpretable benchmark; selected from validation manual-review candidates under corrected workload cost. |
| heloc | best_research_framework_policy | heloc__cand_0186 | SCRE-Optimized | manual_review_band | weighted_sum_of_calibrated_pareto_front_probabilities_no_final_calibration |  | 0.22 | 0.43 | 0.7024 | 0.8228 | 0.622 | 0.8056 | 0.808 | 0.0454 |  | 732.5 | 0.2882 | SCRE retained as reliability-aware research framework; selected from revised SCRE validation manual-review candidates. |

### 4.14 Locked held-out test evaluation

Locked policies were evaluated once on held-out test. No policy was changed after test.

Held-out evidence:

| dataset | selected_model | selected_policy_type | calibration_type | threshold | t_low | t_high | test_precision | test_recall | test_specificity | test_f1 | test_pr_auc | test_roc_auc | test_ece | test_fp | test_fn | test_review_adjusted_cost | test_manual_review_rate | delta_precision_vs_old_catboost | delta_recall_vs_old_catboost | delta_specificity_vs_old_catboost | delta_fp_vs_old_catboost | delta_cost_vs_old_catboost | policy_changed_after_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | Best CatBoost | manual_review_band | isotonic |  | 0.14 | 0.28 | 0.5288 | 0.575 | 0.8545 | 0.5509 | 0.5639 | 0.7853 | 0.0078 | 680 | 564 | 2705.5 | 0.2952 | 0.1773 | -0.2344 | 0.2786 | -1302 | -541.5 | False |
| heloc | Best Scorecard | manual_review_band | sigmoid |  | 0.16 | 0.39 | 0.6834 | 0.8463 | 0.5744 | 0.7562 | 0.8039 | 0.8015 | 0.0142 | 403 | 158 | 764.5 | 0.2699 | 0.1253 | -0.1342 | 0.4171 | -395 | -133.5 | False |

### 4.15 Final tables rebuilt

Final result tables were split into:

- `final_validation_selection_table.csv`: validation-only selection table, rank/winner allowed.
- `final_heldout_test_evidence_table.csv`: held-out locked-policy evidence, no rank/winner.
- `retrospective_all_candidate_test_diagnostics.csv`: old all-candidate test diagnostics, marked not for model selection.

Outcome: previous test-ranking issue was removed.

### 4.16 FP profile completion

Missing columns were added:

- `fp_closer_to`
- `fp_risky_profile_comment`

Outcome: Taiwan and HELOC FPs are TP-like risky profiles, but this supports manual review, not automatic rejection.

### 4.17 Calibration-policy caveat

The same-validation calibration/threshold caveat was documented. A separated calibration/policy check was added for final non-SCRE policies.

Outcome: no test leakage found. SCRE full-stack separation remains future work because it requires re-running the complete ensemble optimization stack.

### 4.18 Final safe recommendation

A safe final recommendation was written.

Outcome:

- Taiwan operational screening: Best CatBoost manual-review band.
- Taiwan interpretable benchmark: Scorecard.
- Taiwan research framework: SCRE-Pareto.
- HELOC operational screening: Best Scorecard manual-review band.
- HELOC interpretable benchmark: Scorecard.
- HELOC research framework: SCRE-Optimized.
- Automatic rejection: not suitable.

### 4.19 Final audit V2

Final audit V2 returned READY.

Audit result counts:

| severity | status | count |
| --- | --- | --- |
| CRITICAL | PASS | 11 |
| HIGH | PASS | 9 |
| MEDIUM | PASS | 19 |

## 5. Current V2 final position

The project should be described as a validation-selected credit-risk screening and manual-review prioritization framework.

Final safe framing:

Pure cost minimization produced an overly aggressive screening model with high recall but excessive false positives. The revised validation-selected policies reduce false-positive pressure and improve operational interpretability through constrained thresholding, manual-review bands, and review-workload-aware cost analysis. The models should be presented as screening and manual-review prioritization tools, not automatic credit rejection systems.

## 6. Unsafe claims to avoid

Do not claim:

- The model is suitable for automatic credit rejection.
- SCRE-Credit outperforms every individual model.
- Manual-review cost is the same as binary FN/FP expected cost.
- Test set was used for final model selection.
- SHAP, LIME, or engineered features prove causal effects.
- Taiwan results automatically generalize to HELOC without domain-shift caveats.
- Calibration necessarily improves AUC.

## 7. Files that define V2 READY state

Core V2 files:

| relative_path | exists | size |
| --- | --- | --- |
| outputs/decision_revision_v2/SELECTION_PROTOCOL_V2.md | True | 3851 |
| outputs/decision_revision_v2/validation_candidate_registry_all.csv | True | 355144 |
| outputs/decision_revision_v2/manual_review_cost_model_all.csv | True | 3359585 |
| outputs/decision_revision_v2/validation_only_selection_taiwan.csv | True | 24224 |
| outputs/decision_revision_v2/validation_only_selection_heloc.csv | True | 24455 |
| outputs/decision_revision_v2/locked_final_policy_taiwan.json | True | 1379 |
| outputs/decision_revision_v2/locked_final_policy_heloc.json | True | 1251 |
| outputs/decision_revision_v2/locked_test_evaluation_taiwan.csv | True | 1775 |
| outputs/decision_revision_v2/locked_test_evaluation_heloc.csv | True | 1763 |
| outputs/decision_revision_v2/final_validation_selection_table.csv | True | 27071 |
| outputs/decision_revision_v2/final_heldout_test_evidence_table.csv | True | 1295 |
| outputs/decision_revision_v2/false_positive_profile_taiwan_completed.csv | True | 3981 |
| outputs/decision_revision_v2/false_positive_profile_heloc_completed.csv | True | 2668 |
| outputs/decision_revision_v2/calibration_threshold_caveat.md | True | 3128 |
| outputs/decision_revision_v2/final_safe_recommendation.md | True | 8043 |
| outputs/decision_revision_v2/final_audit_v2.md | True | 3252 |
| outputs/decision_revision_v2/final_audit_v2.csv | True | 10490 |

## 8. Restore instructions

To restore this V2 state later, copy the V2 backup folder back over a working project directory. Recommended safe approach:

1. Close editors, notebooks, Python terminals, and processes using the project.
2. Rename the current working folder instead of deleting it, for example `Data_Mining_Project_before_restore`.
3. Copy `C:\Users\AKTS\Desktop\Resul\project_backups\SCRE_Credit_v2_READY_20260505_1633` to `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project`.
4. Verify these files exist after restore:
   - `outputs/decision_revision_v2/final_audit_v2.md`
   - `outputs/decision_revision_v2/final_safe_recommendation.md`
   - `outputs/decision_revision_v2/locked_final_policy_taiwan.json`
   - `outputs/decision_revision_v2/locked_final_policy_heloc.json`

## 9. V2 backup note

This README was written before the final copy. After copy verification, the backup folder should contain this README as `README_V2_STATUS.md`.
