# Manual-Review Cost Interpretation

Generated: 2026-05-05 14:42:32

## 1. Why Old Manual-Review Cost Was Not Directly Comparable

The old manual-review result used remaining expected cost: low-risk false negatives plus high-risk false positives. It excluded the cost of sending cases to human review. This is not the same object as binary expected cost, where every customer receives an automatic decision. A manual-review policy can look artificially cheap if it moves many hard cases into the review band and does not price that workload.

## 2. Winner Movement Under Workload Cost

### TAIWAN

- Automation-only residual winner: SCRE-Pareto (D_high_risk_precision_0_40, MR=0.993, automation_residual_cost=11.0).
- Workload-adjusted winner at review_cost=0.5: SCRE-Optimized (C_manual_review_rate_0_30, MR=0.296, review_adjusted_cost=2640.5).
- Workload-adjusted winner at review_cost=1.0: SCRE-Optimized (A_manual_review_rate_0_20, MR=0.191, review_adjusted_cost=3350.0).
- Imperfect-review winner at review_cost=0.5 and error rates 0.05/0.05: Best CatBoost (C_manual_review_rate_0_30, MR=0.297, review_adjusted_cost_imperfect=2810.9).

### HELOC

- Automation-only residual winner: SCRE-Optimized (D_high_risk_precision_0_40, MR=0.905, automation_residual_cost=21.0).
- Workload-adjusted winner at review_cost=0.5: Best Scorecard (C_manual_review_rate_0_30, MR=0.294, review_adjusted_cost=729.0).
- Workload-adjusted winner at review_cost=1.0: SCRE-Optimized (A_manual_review_rate_0_20, MR=0.189, review_adjusted_cost=942.0).
- Imperfect-review winner at review_cost=0.5 and error rates 0.05/0.05: SCRE-Optimized (B_manual_review_rate_0_25, MR=0.249, review_adjusted_cost_imperfect=784.0).

## 3. Workload-Adjusted FN5/FP1 Winners By Review Cost

| dataset | review_cost | best_model | constraint | manual_review_rate | review_adjusted_cost | binary_reference_cost |
| --- | --- | --- | --- | --- | --- | --- |
| taiwan | 0.1000 | Best CatBoost | D_high_risk_precision_0_40 | 0.9808 | 605.5 | 3144.0 |
| taiwan | 0.2500 | Best CatBoost | D_high_risk_precision_0_40 | 0.9808 | 1488.2 | 3144.0 |
| taiwan | 0.5000 | SCRE-Optimized | C_manual_review_rate_0_30 | 0.2958 | 2640.5 | 3107.0 |
| taiwan | 1.0000 | SCRE-Optimized | A_manual_review_rate_0_20 | 0.1913 | 3350.0 | 3107.0 |
| taiwan | 2.0000 | SCRE-Optimized | A_manual_review_rate_0_20 | 0.1913 | 4498.0 | 3107.0 |
| heloc | 0.1000 | SCRE-Optimized | D_high_risk_precision_0_40 | 0.9048 | 199.6 | 853.0 |
| heloc | 0.2500 | SCRE-Pareto | I_high_risk_recall_0_60 | 0.6018 | 444.0 | 885.0 |
| heloc | 0.5000 | Best Scorecard | C_manual_review_rate_0_30 | 0.2938 | 729.0 | 871.0 |
| heloc | 1.0000 | SCRE-Optimized | A_manual_review_rate_0_20 | 0.1890 | 942.0 | 853.0 |
| heloc | 2.0000 | SCRE-Optimized | A_manual_review_rate_0_20 | 0.1890 | 1315.0 | 853.0 |

## 4. Operational Threshold For Heavy Review Load

A manual-review rate above roughly 0.30 should be treated as operationally heavy unless a review-capacity analysis justifies it. Rates above 0.50 are not credible for routine automated screening without a large review operation.

## 5. SCRE Claim Control

SCRE policies may have low automation-only residual cost when many cases are pushed into manual review. That is not a safe cost-superiority claim unless review workload cost and review error are included. SCRE can still be described as a reliability-aware framework, but not as automatically cheaper solely from residual cost.

## 6. CatBoost Manual-Review Band

CatBoost manual-review bands remain operationally plausible when the manual-review rate stays near the bounded 0.20-0.30 region and adjusted cost remains competitive after review workload cost is included. The final selection still must be made by validation-only ranking in the next step.