# Minimum Acceptable Precision Model Selection

No model was trained in this step. Existing result tables were consolidated and ranked by eligibility criteria.

## Old CatBoost Cost Threshold

- Taiwan: NO basic, NO strict (precision=0.3514, recall=0.8093, specificity=0.5759).
- HELOC: YES basic, NO strict (precision=0.5581, recall=0.9805, specificity=0.1573).

## Taiwan Top Eligible Policies

| Rank | Model/Policy | Precision | Recall | Specificity | FP | FN | Cost | Strict? | Comment |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | SCRE-Optimized / manual_review_band:I_high_risk_recall_0_60 | 0.4951 | 0.6112 | 0.8230 | 827 | 516 | 832 | YES | Strict winner: strongest operationally eligible policy under minimum acceptable criteria. |
| 2 | Best XGBoost / manual_review_band:I_high_risk_recall_0_60 | 0.4881 | 0.6172 | 0.8162 | 859 | 508 | 879 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 3 | SCRE-Pareto / manual_review_band:I_high_risk_recall_0_60 | 0.4828 | 0.6240 | 0.8102 | 887 | 499 | 892 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 4 | Best CatBoost / manual_review_band:I_high_risk_recall_0_60 | 0.4780 | 0.6217 | 0.8072 | 901 | 502 | 906 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 5 | Best Scorecard / manual_review_band:I_high_risk_recall_0_60 | 0.4620 | 0.6134 | 0.7971 | 948 | 513 | 958 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 6 | Best XGBoost / manual_review_band:J_high_risk_recall_0_65 | 0.4477 | 0.6609 | 0.7685 | 1082 | 450 | 1102 | NO | Basic eligible three-way decision-support policy; not an automatic rejection rule. |
| 7 | Best Scorecard / manual_review_band:J_high_risk_recall_0_65 | 0.4291 | 0.6473 | 0.7554 | 1143 | 468 | 1153 | NO | Basic eligible three-way decision-support policy; not an automatic rejection rule. |
| 8 | SCRE-Optimized / manual_review_band:J_high_risk_recall_0_65 | 0.4345 | 0.6654 | 0.7541 | 1149 | 444 | 1154 | NO | Basic eligible three-way decision-support policy; not an automatic rejection rule. |

## HELOC Top Eligible Policies

| Rank | Model/Policy | Precision | Recall | Specificity | FP | FN | Cost | Strict? | Comment |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | SCRE-Pareto / manual_review_band:L_review_0_25_low_default_0_10 | 0.6344 | 0.9183 | 0.4256 | 544 | 84.0000 | 544 | YES | Basic winner: lowest-cost eligible policy under minimum acceptable criteria. |
| 2 | Best XGBoost / manual_review_band:B_manual_review_rate_0_25 | 0.6377 | 0.9125 | 0.4372 | 533 | 90.0000 | 568 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 3 | Best XGBoost / manual_review_band:K_review_0_25_precision_0_40 | 0.6377 | 0.9125 | 0.4372 | 533 | 90.0000 | 568 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 4 | Best XGBoost / manual_review_band:L_review_0_25_low_default_0_10 | 0.6377 | 0.9125 | 0.4372 | 533 | 90.0000 | 568 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 5 | SCRE-Pareto / manual_review_band:A_manual_review_rate_0_20 | 0.6435 | 0.9095 | 0.4530 | 518 | 93.0000 | 603 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 6 | Best CatBoost / manual_review_band:A_manual_review_rate_0_20 | 0.6461 | 0.9056 | 0.4615 | 510 | 97.0000 | 610 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 7 | Best Scorecard / manual_review_band:A_manual_review_rate_0_20 | 0.6392 | 0.9066 | 0.4446 | 526 | 96.0000 | 621 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |
| 8 | SCRE-Optimized / manual_review_band:A_manual_review_rate_0_20 | 0.6456 | 0.9056 | 0.4604 | 511 | 97.0000 | 626 | YES | Strict eligible; usable candidate, but not the lowest-cost ranked winner. |

## Direct Answers

1. Old CatBoost cost-threshold eligible? Taiwan: NO basic, NO strict (precision=0.3514, recall=0.8093, specificity=0.5759). HELOC: YES basic, NO strict (precision=0.5581, recall=0.9805, specificity=0.1573).
2. Precision constraint altında winner değişiyor mu? YES; Taiwan old CatBoost cost threshold fails precision eligibility, so the winner moves to constrained/manual-review policies.
3. Manual review policy eligible oluyor mu? YES; Taiwan eligible manual-review rows=25, HELOC eligible manual-review rows=9.
4. Scorecard daha dengeli mi? Taiwan eligible Scorecard rows=11; HELOC eligible Scorecard rows=16. HELOC tarafında daha dengeli ve savunulabilir.
5. SCRE hangi koşullarda mantıklı? SCRE rows passing basic eligibility: Taiwan=25, HELOC=56; özellikle manual-review veya stricter FP-control bandlarında anlamlı.
6. En gerçekçi final operational model/policy: Taiwan için manual-review destekli veya precision-constrained policy; HELOC için SCRE/Scorecard/CatBoost arasında eligibility-passing decision-support policy.

## Final Revised Winners

| Dataset | Scope | Model | Policy | Precision | Recall | Specificity | FP | FN | Cost |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| taiwan | basic | SCRE-Optimized | manual_review_band:I_high_risk_recall_0_60 | 0.4951 | 0.6112 | 0.8230 | 827 | 516 | 832 |
| taiwan | strict | SCRE-Optimized | manual_review_band:I_high_risk_recall_0_60 | 0.4951 | 0.6112 | 0.8230 | 827 | 516 | 832 |
| heloc | basic | SCRE-Pareto | manual_review_band:L_review_0_25_low_default_0_10 | 0.6344 | 0.9183 | 0.4256 | 544 | 84.0000 | 544 |
| heloc | strict | SCRE-Pareto | manual_review_band:C_manual_review_rate_0_30 | 0.6566 | 0.8891 | 0.4952 | 478 | 114 | 478 |