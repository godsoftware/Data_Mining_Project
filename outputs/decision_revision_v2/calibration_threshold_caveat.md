# Calibration and Threshold Selection Caveat

## 1. What is the problem?
Some decision-revision experiments, especially the imbalance-training revision, fit a probability calibrator on the validation split and then choose thresholds/policies on that same validation split. This is not test leakage, but it can make validation-side policy estimates mildly optimistic.

## 2. Is there test leakage?
No. The held-out test split was not used to fit the model, fit the calibrator, choose thresholds, choose manual-review bands, or select policies.

## 3. Why is this a caveat?
Calibration and policy selection are two separate tuning operations. Reusing the same validation observations for both can overfit the validation split even though the test set remains clean.

## 4. Was separate validation applied?
Partially yes. A separated protocol check was applied for the final selected non-SCRE model and scorecard benchmark rows. SCRE was documented but not fully recomputed because it requires re-running the full ensemble-weight optimization stack.

## 5. What changed in the separated check?
- taiwan / final_selected_model / Best CatBoost: selected band t_low=0.11, t_high=0.21; policy_val high-risk precision=0.453, recall=0.636, review-adjusted cost=1710.0; test precision=0.459, recall=0.641, review-adjusted cost=2812.5.
- taiwan / scorecard_benchmark / Best Scorecard: selected band t_low=0.13, t_high=0.24; policy_val high-risk precision=0.506, recall=0.555, review-adjusted cost=1742.5; test precision=0.515, recall=0.553, review-adjusted cost=2858.0.
- taiwan / scre_policy / SCRE-Pareto: not_recomputed - SCRE policy depends on previously optimized ensemble weights and calibrated base-model probabilities; full separation would require re-running the SCRE ensemble optimization stack. Documented as caveat, not used to replace locked policy.
- heloc / final_selected_model / Best Scorecard: selected band t_low=0.19, t_high=0.43; policy_val high-risk precision=0.714, recall=0.822, review-adjusted cost=428.5; test precision=0.694, recall=0.827, review-adjusted cost=780.5.
- heloc / scorecard_benchmark / Best Scorecard: selected band t_low=0.19, t_high=0.43; policy_val high-risk precision=0.714, recall=0.822, review-adjusted cost=428.5; test precision=0.694, recall=0.827, review-adjusted cost=780.5.
- heloc / scre_policy / SCRE-Optimized: not_recomputed - SCRE policy depends on previously optimized ensemble weights and calibrated base-model probabilities; full separation would require re-running the SCRE ensemble optimization stack. Documented as caveat, not used to replace locked policy.

## 6. Safe report wording
Primary held-out test claims do not involve test leakage. However, some exploratory threshold/calibration experiments reused the validation split for both calibration fitting and policy selection. A separated calibration/policy validation check was added for the final non-SCRE policies; future work should run the full SCRE and imbalance-training stack with nested calibration-policy validation before making strong selection claims from those exploratory variants.
