# Final Attempt Protocol

Generated: 2026-05-06 09:54:18

Project root: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project`

Output root for this final attempt: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project\outputs\final_attempt`

## Objective

This is the final major experiment package on top of the V2 READY project state. The objective is to run one last set of strong, leakage-free, methodologically clean experiments that test whether the current V2 recommendation can be improved without changing the research scope.

The final attempt will focus on:

1. Reproducing high scores from the literature under strict leakage controls.
2. Evaluating advanced imbalance strategies on the natural held-out distribution rather than resampled test distributions.
3. Testing stronger interpretable or monotonic models such as EBM-style and monotonic model families.
4. Building capacity-aware manual-review policies instead of unconstrained review bands.
5. Measuring decision usefulness with decision curve / net benefit analysis.
6. Selecting the final model and policy using validation evidence only.
7. Evaluating the locked final policy on held-out test only after selection is complete.

## Datasets

- **Taiwan / UCI Default of Credit Card Clients**: primary dataset. It remains the main model-development and primary evaluation dataset.
- **FICO HELOC**: external validation dataset. It remains the external robustness and domain-shift check. It must not be treated as a replacement for Taiwan or as a source for Taiwan-specific feature engineering.

No additional datasets will be added in this final attempt.

## Forbidden operations

The following operations are forbidden for the final attempt:

- Using the test set for model selection.
- Using the test set for policy selection.
- Using the test set for threshold selection.
- Using the test set for manual-review band selection.
- Using the test set for calibration fitting.
- Using the test set for resampling, SMOTE, undersampling, or any distribution-shaping operation.
- Applying SMOTE/SMOTENC/oversampling before train/validation/test splitting.
- Applying resampling to validation or test splits.
- Creating new raw features not already present in the project scope.
- Creating target-derived features.
- Creating future-information features.
- Reintroducing ID as a model feature.
- Adding new datasets.
- Ranking final candidates by held-out test metrics.
- Declaring any all-candidate test table as selection evidence.
- Presenting any model as suitable for automatic credit rejection without manual-review caveats.
- Claiming SCRE-Credit or any new final-attempt method dominates all baselines unless validation selection and held-out evidence both support that statement without test-selection bias.

## Allowed experiments

The following experiments are allowed, provided they obey the selection and test rules below:

- **Strict literature reproduction**: reproduce published or commonly reported high-scoring approaches using the existing Taiwan and HELOC processed datasets while documenting every preprocessing, split, and leakage-control assumption.
- **Advanced imbalance training**: try leakage-free class weighting, focal-style losses, balanced bagging, threshold-free imbalance-aware objectives, and resampling only inside training folds or training pipelines.
- **Deep tabular reproduction**: if used, reproduce tabular deep-learning baselines without new features or target-derived transformations; validation-only selection remains mandatory.
- **EBM / monotonic interpretable models**: evaluate explainable boosting machine style models, generalized additive model families, monotonic boosting, or monotonic constraints using existing features only.
- **Capacity-aware manual review**: optimize manual-review policies under explicit review capacity and review-cost constraints.
- **Decision curve**: evaluate net benefit / standardized net benefit over clinically or operationally meaningful threshold-probability ranges.
- **Final validation-only selector**: select final model/policy from validation evidence only, with a written locked-policy artifact before held-out test evaluation.

## Selection rule

Final selection must use validation evidence only.

Allowed selection evidence includes:

- Validation PR-AUC / ROC-AUC.
- Validation precision, recall, specificity, F1.
- Validation Brier and ECE.
- Validation expected cost under pre-specified FN/FP costs.
- Validation review-adjusted cost under pre-specified review-cost assumptions.
- Validation manual-review rate and capacity constraints.
- Validation decision curve / net benefit summaries.
- Pre-specified interpretability or monotonicity constraints.

Forbidden selection evidence includes all held-out test metrics, test confusion matrices, test costs, test manual-review rates, and test net-benefit summaries.

## Test rule

The test set may be used only once the final policy is locked.

The required sequence is:

1. Generate candidate models and policies using train/dev/validation evidence only.
2. Select final model/policy using validation evidence only.
3. Write locked final policy JSON/CSV before opening held-out test evidence.
4. Evaluate only the locked final policy on held-out test.
5. Store held-out test evidence in `outputs/final_attempt/locked_test/` with no rank column and no winner column.

If any all-candidate test diagnostics are produced, they must be labeled diagnostic only and must include a `not_for_model_selection = TRUE` guard column.

## Baseline V2 reference

The current V2 READY state is frozen as the baseline before final-attempt experiments.

- V2 audit verdict: **READY**.
- V2 critical failures: `0`.
- V2 high failures: `0`.
- V2 medium fail/warn count: `0`.
- V2 baseline freeze file: `outputs/final_attempt/protocol/v2_baseline_freeze.csv`.
- V2 input inventory: `outputs/final_attempt/protocol/final_attempt_input_inventory.csv`.

V2 results may be used as benchmark context. However, V2 held-out test metrics must not be used to select new final-attempt models or policies.

The V2 locked baseline is:

| Dataset | V2 locked model | Policy | Calibration | Band/threshold | Validation selection basis | Held-out evidence role |
|---|---|---|---|---|---|---|
| Taiwan | Best CatBoost | manual_review_band | isotonic | t_low=0.14, t_high=0.28 | validation review-adjusted cost with review_cost=0.5 | evidence only, no ranking |
| HELOC | Best Scorecard | manual_review_band | sigmoid | t_low=0.16, t_high=0.39 | validation review-adjusted cost with review_cost=0.5 | evidence only, no ranking |

## Output isolation rule

All new final-attempt outputs must be written under `outputs/final_attempt/`.

The following V2 directories and files are read-only references for the final attempt:

- `outputs/decision_revision_v2/`
- `outputs/decision_revision_archive_*/`
- `outputs/homework_delivery_pack/`
- `project_backups/SCRE_Credit_v2_READY_*`

No final-attempt script should overwrite V2 READY artifacts.
