# Cost Matrix Sensitivity Summary: HELOC

Thresholds were selected on validation only for each cost scenario. Test metrics are holdout evaluations of those selected policies.

## Validation-Selected Winners By Cost Scenario

| Scenario | Best Model | Threshold | Precision | Recall | Specificity | FP | FN | Cost | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| FN=2, FP=1 | Best XGBoost | 0.3050 | 0.6503 | 0.9027 | 0.4731 | 499 | 100 | 699 | More conservative cost ratio; lower FP burden is prioritized. |
| FN=2, FP=2 | Old SCRE-Credit | 0.5000 | 0.7448 | 0.7383 | 0.7254 | 260 | 269 | 1058 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=3, FP=1 | Old SCRE-Credit | 0.3050 | 0.6251 | 0.9261 | 0.3970 | 571 | 76.0000 | 799 | Reference-like cost tradeoff; use alongside precision/specificity constraints. |
| FN=3, FP=2 | Best Scorecard | 0.3650 | 0.6740 | 0.8687 | 0.5438 | 432 | 135 | 1269 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=5, FP=1 | SCRE-Optimized | 0.2200 | 0.5615 | 0.9776 | 0.1711 | 785 | 23.0000 | 900 | Reference-like cost tradeoff; use alongside precision/specificity constraints. |
| FN=5, FP=2 | Best XGBoost | 0.3050 | 0.6503 | 0.9027 | 0.4731 | 499 | 100 | 1498 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=5, FP=3 | Best Scorecard | 0.3650 | 0.6740 | 0.8687 | 0.5438 | 432 | 135 | 1971 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |
| FN=10, FP=1 | Best XGBoost | 0.0900 | 0.5276 | 0.9932 | 0.0348 | 914 | 7.0000 | 984 | Higher FN cost keeps an aggressive screening threshold and protects recall. |
| FN=10, FP=2 | SCRE-Optimized | 0.2200 | 0.5615 | 0.9776 | 0.1711 | 785 | 23.0000 | 1800 | Reference-like cost tradeoff; use alongside precision/specificity constraints. |
| FN=10, FP=3 | Old SCRE-Credit | 0.3050 | 0.6251 | 0.9261 | 0.3970 | 571 | 76.0000 | 2473 | Higher FP cost pushes threshold upward; precision/specificity improve at the cost of recall. |

## Audit Answers

1. FP cost 1 -> 2 raises threshold for 7/7 tracked model policies.
2. Precision increases for 7/7 tracked model policies.
3. Specificity increases for 7/7 tracked model policies.
4. Average recall change when moving FN=5, FP=1 -> FN=5, FP=2: -0.0593.
5. CatBoost is the validation-cost winner in 0/10 scenarios.
6. Scorecard winner scenarios: 2. It remains useful as an interpretable benchmark even when not cost-winning.
7. SCRE winner scenarios: 2. Use SCRE sensitivity rows to discuss reliability-aware integration rather than guaranteed superiority.
8. Recommended operational discussion scenario: FN=5, FP=2 or FN=3, FP=2, because HELOC false positives are operationally costly.
