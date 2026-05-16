# Final Attempt Audit

Generated: 2026-05-06 14:51:22

## 1. Executive verdict

- Verdict: **BLOCKED**
- Critical failures: 9
- High issues/warnings: 4
- Medium warnings: 0
- No detected test-selection leakage, no split-before-resampling failure in the audited tables, and no test ranking/winner columns in locked-test outputs.
- Blocking issue: several final locked-test manual-review rows do not satisfy the requested confusion-matrix metric identities. In those rows, `test_precision` may match TP/(TP+FP), but `test_recall`, `test_specificity`, or `test_f1` use a different manual-review denominator than the displayed `test_tn/test_fp/test_fn/test_tp` columns.

## 2. Critical issues

- **Metric consistency / taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_recall**: calc=0.782608695652174, observed=0.5968349660889224
- **Metric consistency / taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_specificity**: calc=0.7536411527734738, observed=0.8298737427776589
- **Metric consistency / taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_f1**: calc=0.6094651789149672, observed=0.543582704186685
- **Metric consistency / taiwan:Interpretable benchmark:Monotonic LightGBM conservative_recall**: calc=0.7803180914512923, observed=0.5915599095704597
- **Metric consistency / taiwan:Interpretable benchmark:Monotonic LightGBM conservative_specificity**: calc=0.7565164060104262, observed=0.8300877380697624
- **Metric consistency / taiwan:Interpretable benchmark:Monotonic LightGBM conservative_f1**: calc=0.6073500967117988, observed=0.5402615278733655
- **Metric consistency / heloc:Interpretable benchmark:WOE scorecard unweighted_recall**: calc=0.9691780821917808, observed=0.8258754863813229
- **Metric consistency / heloc:Interpretable benchmark:WOE scorecard unweighted_specificity**: calc=0.31663685152057247, observed=0.5966209081309398
- **Metric consistency / heloc:Interpretable benchmark:WOE scorecard unweighted_f1**: calc=0.8058851447555766, observed=0.751660026560425

## 3. High issues

- **Manual-review cost / taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_review_cost_auditability**: WARN. final locked table has MR rate=0.2935 and adjusted cost=2775.5, but no manual review count/autofp/autofn decomposition; formula not fully auditable from this final table 
- **Manual-review cost / taiwan:Interpretable benchmark:Monotonic LightGBM conservative_review_cost_auditability**: WARN. final locked table has MR rate=0.2888333333333333 and adjusted cost=2765.5, but no manual review count/autofp/autofn decomposition; formula not fully auditable from this final table 
- **Manual-review cost / heloc:Final operational screening:Best Scorecard_review_cost_auditability**: WARN. final locked table has MR rate=0.2698734177215189 and adjusted cost=764.5, but no manual review count/autofp/autofn decomposition; formula not fully auditable from this final table 
- **Manual-review cost / heloc:Interpretable benchmark:WOE scorecard unweighted_review_cost_auditability**: WARN. final locked table has MR rate=0.2734177215189873 and adjusted cost=787.0, but no manual review count/autofp/autofn decomposition; formula not fully auditable from this final table 

## 4. Medium issues

- None detected.

## 5. Leakage status

- literature_test_set_used_for_selection: PASS (unique=['false'])
- advanced_imbalance_test_set_used_for_selection: PASS (unique=['false'])
- deep_tabular_test_set_used_for_selection: PASS (unique=['false'])
- interpretable_test_set_used_for_selection: PASS (unique=['false'])
- capacity_taiwan_test_set_used_for_selection: PASS (unique=['false'])
- capacity_heloc_test_set_used_for_selection: PASS (unique=['false'])
- dca_taiwan_test_set_used_for_selection: PASS (unique=['false'])
- dca_heloc_test_set_used_for_selection: PASS (unique=['false'])
- final_selection_taiwan_test_set_used_for_selection: PASS (unique=['false'])
- final_selection_heloc_test_set_used_for_selection: PASS (unique=['false'])
- literature_resampling_inside_train_fold: PASS (bad_rows=0)
- literature_split_before_resampling: PASS (bad_rows=0)
- advanced_no_new_features: PASS (new_feature_rows=0)
- deep_no_test_resampling: PASS (test_resampling_rows=0)
- deep_no_new_features: PASS (new_feature_rows=0)
- taiwan_no_test_metric_cols: PASS (bad_test_cols=[])
- taiwan_selection_rank_scope: PASS (rank/winner cols=[]; selection table is validation-only)
- heloc_no_test_metric_cols: PASS (bad_test_cols=[])
- heloc_selection_rank_scope: PASS (rank/winner cols=[]; selection table is validation-only)
- json_test_selection_flag: PASS (False)
- do_not_change_after_locking: PASS (True)
- taiwan_no_rank_winner: PASS (rank/winner cols=[])
- taiwan_test_selection_flag: PASS (bad_rows=0)
- taiwan_policy_changed_after_test: PASS (bad_rows=0)
- heloc_no_rank_winner: PASS (rank/winner cols=[])
- heloc_test_selection_flag: PASS (bad_rows=0)
- heloc_policy_changed_after_test: PASS (bad_rows=0)

## 6. Metric consistency

- FAIL. Displayed final locked-test metrics are not always recomputable from displayed `test_tn/test_fp/test_fn/test_tp`. This violates the requested final audit identities.
  - taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_recall: calc=0.782608695652174, observed=0.5968349660889224
  - taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_specificity: calc=0.7536411527734738, observed=0.8298737427776589
  - taiwan:Final operational screening:XGBoost xgboost_scale_pos_weight_5_f1: calc=0.6094651789149672, observed=0.543582704186685
  - taiwan:Interpretable benchmark:Monotonic LightGBM conservative_recall: calc=0.7803180914512923, observed=0.5915599095704597
  - taiwan:Interpretable benchmark:Monotonic LightGBM conservative_specificity: calc=0.7565164060104262, observed=0.8300877380697624
  - taiwan:Interpretable benchmark:Monotonic LightGBM conservative_f1: calc=0.6073500967117988, observed=0.5402615278733655
  - heloc:Interpretable benchmark:WOE scorecard unweighted_recall: calc=0.9691780821917808, observed=0.8258754863813229
  - heloc:Interpretable benchmark:WOE scorecard unweighted_specificity: calc=0.31663685152057247, observed=0.5966209081309398
  - heloc:Interpretable benchmark:WOE scorecard unweighted_f1: calc=0.8058851447555766, observed=0.751660026560425

## 7. Capacity consistency

- taiwan_capacity_formula: PASS (bad_count=0; sample=[])
- heloc_capacity_formula: PASS (bad_count=0; sample=[])

## 8. Decision curve consistency

- taiwan_net_benefit_formula: PASS (bad_count=0; sample=[])
- heloc_net_benefit_formula: PASS (bad_count=0; sample=[])

## 9. Literature reproduction validity

- PASS on audited leakage controls: `test_set_used_for_selection=False`, resampling inside train fold, and split-before-resampling flags are clean in `literature_reproduction_validation_all.csv`.
- No evidence in the audited tables that SMOTE/KMeansSMOTE was applied before split or to the test set.
- Interpretation remains valid only as leakage-free reproduction under this project split; it should not claim to disprove all literature results.

## 10. Final selected policy validity

- Selection protocol itself is valid: final selection tables contain validation metrics and `test_set_used_for_selection=False`; locked-test tables contain no rank/winner columns.
- Final locked evidence table is not report-ready because several manual-review rows fail metric identity checks.
- Held-out evidence weakens the Taiwan final operational switch: XGBoost improves recall but worsens precision, specificity, FP count, and review-adjusted cost versus V2 CatBoost.
- HELOC final operational policy remains V2 Scorecard; SCRE-Optimized is useful as review-prioritization evidence, not as an operational binary winner.

## 11. Ready for report?

- **NO** for the final-attempt locked-test tables in their current form.
- The leakage-free experiment narratives are usable, but the final locked-test evidence table must be corrected/re-expressed before it can support report claims.

## 12. Safe final claim

The final-attempt experiments did not justify replacing the V2 operational recommendation on held-out evidence. They support a cautious conclusion: V2 remains the safer operational baseline, while SCRE-Optimized is useful for review prioritization and the final attempt provides additional leakage-free stress tests of resampling, imbalance, deep tabular, interpretable, capacity, and decision-curve analyses. Current final locked-test metric tables require correction before formal reporting.

## 13. Must fix list

1. Rebuild final locked-test manual-review rows so `test_tn/test_fp/test_fn/test_tp` and `test_precision/test_recall/test_specificity/test_f1` use one explicit denominator and are mutually consistent.
2. Add manual-review decomposition columns: manual_review_count, auto_FP, auto_FN, auto_TP, auto_TN, manual_review_defaults, manual_review_nondefaults, review_cost, and explicit formula output.
3. Rename or explain manual-review `test_accuracy` as auto-decision accuracy if manual-review cases are excluded.
4. Do not claim the final-attempt Taiwan XGBoost policy beats V2; held-out evidence says it does not on precision/specificity/cost.
5. Keep SCRE-Optimized as review-prioritization / research-framework evidence, not binary operational dominance.
