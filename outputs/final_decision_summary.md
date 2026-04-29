# Final Decision Summary

## 1. What wins on Taiwan?

Best operational model on Taiwan is **Best CatBoost** with expected cost 3247, PR-AUC 0.5639, ROC-AUC 0.7853, recall 0.8093, and ECE 0.0078.

For raw predictive ranking, **SCRE-Optimized** has the highest PR-AUC (0.5669), but it does not beat CatBoost on operational expected cost.

## 2. What wins on HELOC?

Best operational model on HELOC is **Best Scorecard** with expected cost 892. The raw PR-AUC winner is **SCRE-Optimized** with PR-AUC 0.8130. This difference should be treated as a domain-shift and objective-tradeoff finding, not as an error.

## 3. Did SCRE-Credit improve after revision?

Yes, methodologically. Old SCRE was revised into SCRE-Pareto and SCRE-Optimized. SCRE-Optimized improved the SCRE-family discrimination profile: Taiwan SCRE-Optimized PR-AUC is 0.5669, and HELOC SCRE-Optimized PR-AUC is 0.8130.

However, revision did not make SCRE the best operational cost model. On Taiwan, CatBoost vs SCRE-Optimized expected-cost difference is 65 against SCRE-Optimized, p=0.126.

## 4. Is SCRE-Credit the best predictor?

No. It is not the best standalone predictor under the main operational cost criterion. Taiwan CatBoost is stronger for deployment-style cost, and HELOC Scorecard is the external-validation cost winner.

Old SCRE is significantly worse than CatBoost on Taiwan expected cost in the paired bootstrap summary: difference 123, p=0.014.

## 5. Is SCRE-Credit still useful as a framework?

Yes. SCRE-Optimized is useful as a reliability-aware research framework because it integrates calibrated probabilities, validation-only objective weighting, Pareto model selection, cost-sensitive thresholding, and explicit guardrails against test leakage.

Its correct positioning is: **a reliability-aware ensemble framework**, not **the universally best classifier**.

## 6. What should be claimed?

- Taiwan operational winner: Best CatBoost.
- HELOC external-validation cost winner: Best Scorecard.
- Best interpretable benchmark: Scorecard.
- Best proposed SCRE-family framework: SCRE-Optimized.
- SCRE-Optimized can improve PR-AUC within the SCRE family and provides a structured reliability-aware framework.
- Manual-review bands can reduce remaining auto-decision cost, but they require operational review capacity assumptions.

## 7. What should not be claimed?

- Do not claim SCRE-Credit beats CatBoost on Taiwan operational expected cost.
- Do not claim SCRE-Credit is the best predictor across all metrics.
- Do not claim LightGBM SHAP stability proves CatBoost, Scorecard, or SCRE ensemble stability.
- Do not claim HELOC results are identical to Taiwan results; they show domain shift.
- Do not claim statistical significance means the candidate model won; direction matters.

## 8. Ready for report writing: YES/NO

**YES**, with careful claim boundaries. The report should be written as an objective-specific model validation study: CatBoost for Taiwan operational cost, Scorecard as interpretable/external-validation benchmark, and SCRE-Optimized as the proposed reliability-aware framework.
