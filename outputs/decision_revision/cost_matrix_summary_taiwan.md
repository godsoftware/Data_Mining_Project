# Cost Matrix Sensitivity Summary: TAIWAN

Thresholds were selected on validation only for each cost scenario. Test metrics are holdout evaluations of those selected policies.

## Validation-Selected Winners By Cost Scenario

| Scenario | Best Model | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FN=2, FP=1 | SCRE-Pareto | 0.3500 | 0.5834 | 0.4770 | 0.9033 | 452 | 694 | 1840 | More conservative cost ratio; lower FP burden is prioritized. |
| FN=2, FP=2 | SCRE-Optimized | 0.4650 | 0.6582 | 0.3715 | 0.9452 | 256 | 834 | 2180 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=3, FP=1 | Best XGBoost | 0.2900 | 0.5415 | 0.5456 | 0.8688 | 613 | 603 | 2422 | Reference-like cost tradeoff; use alongside precision/specificity constraints. |
| FN=3, FP=2 | Old SCRE-Credit | 0.4650 | 0.6012 | 0.4612 | 0.9131 | 406 | 715 | 2957 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=5, FP=1 | SCRE-Optimized | 0.1600 | 0.3601 | 0.7769 | 0.6080 | 1832 | 296 | 3312 | Reference-like cost tradeoff; use alongside precision/specificity constraints. |
| FN=5, FP=2 | Best XGBoost | 0.2900 | 0.5415 | 0.5456 | 0.8688 | 613 | 603 | 4241 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=5, FP=3 | Old SCRE-Credit | 0.4650 | 0.6012 | 0.4612 | 0.9131 | 406 | 715 | 4793 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=10, FP=1 | Best XGBoost | 0.1200 | 0.3108 | 0.8855 | 0.4423 | 2606 | 152 | 4126 | Higher FN cost keeps an aggressive screening threshold and protects recall. |
| FN=10, FP=2 | SCRE-Optimized | 0.1600 | 0.3601 | 0.7769 | 0.6080 | 1832 | 296 | 6624 | Reference-like cost tradeoff; use alongside precision/specificity constraints. |
| FN=10, FP=3 | SCRE-Pareto | 0.2100 | 0.4514 | 0.6473 | 0.7766 | 1044 | 468 | 7812 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |

## Audit Answers

1. FP cost 1 -> 2 raises threshold for 7/7 tracked model policies.
2. Precision increases for 7/7 tracked model policies.
3. Specificity increases for 7/7 tracked model policies.
4. Average recall change when moving FN=5, FP=1 -> FN=5, FP=2: -0.2351.
5. CatBoost is the validation-cost winner in 0/10 scenarios.
6. Scorecard winner scenarios: 0. It remains useful as an interpretable benchmark even when not cost-winning.
7. SCRE winner scenarios: 5. Use SCRE sensitivity rows to discuss reliability-aware integration rather than guaranteed superiority.
8. Recommended operational discussion scenario: FN=5, FP=2 as a defensible sensitivity scenario, paired with precision constraints.
