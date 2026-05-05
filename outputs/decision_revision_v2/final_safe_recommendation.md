# Final Safe Recommendation

## 1. Final selection protocol

Final model and policy selection was made using validation-only evidence. The selection table is `final_validation_selection_table.csv`, where ranking and winner flags are allowed because they are based on validation metrics only.

The held-out test set was used only after the final policies were locked in `locked_final_policy_taiwan.json` and `locked_final_policy_heloc.json`. The test evidence table is `final_heldout_test_evidence_table.csv`; it contains no model ranking and no winner column.

No final model, threshold, calibration method, manual-review band, or policy was changed after seeing held-out test results.

## 2. Taiwan recommendation

Operational screening policy: use the validation-selected Best CatBoost manual-review band with isotonic calibration, `t_low=0.14` and `t_high=0.28`. On validation this policy had precision `0.529`, recall `0.567`, specificity `0.857`, manual-review rate `0.297`, and review-adjusted cost `2650.5` under `FN=5`, `FP=1`, `review_cost=0.5`. Held-out test evidence was precision `0.529`, recall `0.575`, specificity `0.854`, manual-review rate `0.295`, and review-adjusted cost `2705.5`.

Interpretable benchmark: keep the Scorecard manual-review policy as the interpretable comparator. On validation it used `t_low=0.14`, `t_high=0.19`, with precision `0.433`, recall `0.662`, specificity `0.754`, manual-review rate `0.195`, and review-adjusted cost `2993.5`.

Research framework: retain SCRE-Pareto as the reliability-aware framework candidate. It was close to CatBoost on validation manual-review cost (`2651.5`) but should not be described as strictly superior to CatBoost.

Automatic rejection suitability: not suitable. The policy is better described as risk screening and manual-review prioritization, not automated rejection.

## 3. HELOC recommendation

Operational screening policy: use the validation-selected Best Scorecard manual-review band with sigmoid calibration, `t_low=0.16` and `t_high=0.39`. On validation this policy had precision `0.701`, recall `0.842`, specificity `0.610`, manual-review rate `0.294`, and review-adjusted cost `729.0` under `FN=5`, `FP=1`, `review_cost=0.5`. Held-out test evidence was precision `0.683`, recall `0.846`, specificity `0.574`, manual-review rate `0.270`, and review-adjusted cost `764.5`.

Interpretable benchmark: Scorecard is also the primary interpretable benchmark for HELOC. Its strong validation and held-out evidence make it the most defensible external-validation model for explanation-oriented reporting.

Research framework: retain SCRE-Optimized as the HELOC reliability-aware framework candidate. It had strong binary validation performance and competitive manual-review evidence, but the final manual-review operational policy favored Scorecard after workload-adjusted review cost.

Automatic rejection suitability: not suitable. HELOC evidence supports screening/manual-review prioritization rather than automatic adverse decisions.

## 4. Why not automatic rejection?

Precision remains limited, especially for Taiwan. A false positive is a customer who did not default in the observed outcome window, so converting the model signal directly into automatic rejection would be too strong.

False positives still matter operationally. Even when FP cases look closer to TP-like risky profiles than TN-like profiles, they remain non-default outcomes. The right interpretation is elevated-risk screening, not final credit denial.

Manual review is necessary because the model is strongest as a triage layer. The high-risk bucket can prioritize review, while the manual-review band absorbs borderline uncertainty.

The dataset signal is limited. Taiwan and HELOC differ in feature definitions, target construction, and domain context, so the model should not be presented as a universal automated credit-decision engine.

## 5. What improved compared with old aggressive cost threshold?

Precision improved. Taiwan moved from the older aggressive CatBoost precision around `0.351` to locked held-out precision `0.529`. HELOC improved from the old aggressive CatBoost reference around `0.558` to locked held-out precision `0.683`.

Specificity improved. Taiwan held-out specificity reached `0.854`; HELOC held-out specificity reached `0.574`.

False positives were reduced. Taiwan FP count decreased by `1302` versus the old aggressive CatBoost reference. HELOC FP count decreased by `395`.

Operational practicality improved because high-risk decisions are no longer treated as automatic rejection. The revised policy explicitly includes a manual-review band and prices review workload separately from binary expected cost.

## 6. What got worse?

Recall can decrease when precision and specificity are constrained. This is an expected trade-off when moving away from aggressive pure cost-minimizing thresholds.

Cost can increase under some review-cost assumptions. Manual-review policies are not automatically cheaper than binary policies once workload cost and imperfect review are considered.

Manual-review workload is created. A review rate around `0.27` to `0.30` is operationally heavy but plausible if the system is framed as screening. It should be justified with review-capacity assumptions.

Calibration and policy selection also carry a caveat: some exploratory imbalance-training experiments reused the same validation split for calibration fitting and policy selection. This is not test leakage, but future full-stack SCRE work should use nested calibration-policy validation. A separated check was added for the final non-SCRE policies.

## 7. SCRE role

SCRE-Credit should not be claimed as a model that beats every individual baseline. The revised evidence supports SCRE as a reliability-aware framework for organizing performance, calibration, cost, stability, faithfulness, and external-validation evidence.

For Taiwan, CatBoost remains the safer operational screening recommendation under the validation-selected manual-review policy. SCRE-Pareto remains useful as the research framework comparator.

For HELOC, Scorecard is the safer operational and interpretable recommendation under the validation-selected manual-review policy. SCRE-Optimized remains useful as the reliability-aware framework comparator.

## 8. Scorecard role

Scorecard should be retained as the interpretable benchmark. It is especially important because it gives a transparent, credit-risk-compatible comparator against CatBoost, XGBoost, LightGBM, and SCRE variants.

On HELOC, Scorecard is not merely a weak interpretability baseline; it is the selected manual-review operational policy under the corrected validation-only protocol. This strengthens the external-validation story without overclaiming black-box superiority.

For deployment-style discussion, Scorecard is valuable because it is easier to explain, audit, and defend than a purely black-box model.

## 9. Safe final claim

Pure cost minimization produced an overly aggressive screening model with high recall but excessive false positives. The revised validation-selected decision policies reduce false-positive pressure and improve operational interpretability by combining constrained thresholding, manual-review bands, and cost-sensitivity analysis. The final models should be used as screening and manual-review prioritization tools rather than automatic rejection systems.

## 10. Unsafe claims

Do not claim: "The model is suitable for automatic credit rejection."

Do not claim: "SCRE-Credit outperforms all individual models."

Do not claim: "Manual-review cost is directly the same as binary expected cost."

Do not claim: "The test set was used for final model selection."

Do not claim: "The engineered features or SHAP explanations prove causal effects."

Do not claim: "The Taiwan result automatically generalizes to HELOC without domain-shift caveats."

Do not claim: "Calibration necessarily improves AUC." Calibration is probability reliability work, not an AUC-improvement mechanism.
