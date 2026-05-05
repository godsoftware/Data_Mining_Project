# Final Table Methodology Note

## 1. Why there are two tables
The revised workflow separates model/policy selection evidence from held-out test evidence. The old final comparison mixed candidate ranking with test-set metrics, which made the previous audit verdict NOT READY. The new structure prevents test-set results from influencing final policy selection.

## 2. What the validation table is for
`final_validation_selection_table.csv` is the only table that can be used for final model/policy selection. It contains validation-side metrics, validation ranks, eligibility, and the validation-selected winner flag. These ranks are allowed because they were produced before held-out test evaluation.

## 3. What the test evidence table is for
`final_heldout_test_evidence_table.csv` reports only the performance of the locked policies from `locked_final_policy_taiwan.json` and `locked_final_policy_heloc.json`. It is held-out evidence after locking, not selection evidence.

## 4. Why the test table has no rank
The held-out test table intentionally contains no model ranking and no winner column. It includes `rank_column_allowed = NO` and `winner_column_allowed = NO` as explicit audit guards. Test results can support retrospective interpretation, but they must not choose or reorder candidate policies.

## 5. How the old NOT READY issue was fixed
The old retrospective all-candidate tables were moved into `retrospective_all_candidate_test_diagnostics.csv` and marked with `not_for_model_selection = TRUE`. Final selection now comes from validation-only evidence, while test metrics are shown only for the locked policies. This resolves the specific audit issue where test-set metrics appeared to drive operational ranking.

## Output files
- `outputs\decision_revision_v2\final_validation_selection_table.csv`
- `outputs\decision_revision_v2\final_heldout_test_evidence_table.csv`
- `outputs\decision_revision_v2\retrospective_all_candidate_test_diagnostics.csv`
