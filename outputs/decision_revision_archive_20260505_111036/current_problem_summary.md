# Current Decision Problem Summary

## 1. Current Taiwan problem

The frozen Taiwan operational winner is Best CatBoost at threshold 0.15. It has high recall (0.8093) but low precision (0.3514) and only moderate specificity (0.5759). The confusion matrix is TN=2691, FP=1982, FN=253, TP=1074. This is a screening-style configuration: it catches many likely defaults but sends many non-default cases into the positive/high-risk bucket.

At threshold 0.50, the same CatBoost probability model has much higher precision (0.6739) and specificity (0.9516), but recall falls to 0.3519. That contrast is the core reason for the next false-positive reduction experiments.

The most FP-heavy frozen Taiwan cost-optimal row among the tracked final candidates is Best XGBoost with FP=2198, precision=0.3356, specificity=0.5296, and recall=0.8365.

## 2. Current HELOC problem

The frozen HELOC expected-cost/interpretable winner is Best Scorecard at threshold 0.16. It has very high recall (0.9815) but low specificity (0.1584) and FP=797. Best CatBoost is close in cost and calibration, with threshold 0.14, recall=0.9805, precision=0.5581, specificity=0.1573, and FP=798.

HELOC has a different class balance and domain structure than Taiwan, so the same threshold intuition should not be transferred blindly.

## 3. Why accuracy is misleading

Taiwan's majority-class accuracy is 0.7788 by predicting class 0 for everyone, because most customers are non-default. HELOC's majority-class accuracy is 0.5203 by predicting class 1. A model can look accurate by avoiding positive/default predictions, while missing costly defaults. Conversely, a low-threshold screening policy can lower accuracy by increasing false positives while still reducing FN-weighted cost.

Accuracy therefore hides the precision-recall-specificity tradeoff that matters for credit-risk decisioning.

## 4. Why precision is low

The cost-sensitive thresholds are low because FN is weighted five times FP. Lowering the threshold moves more customers into the predicted-default/high-risk bucket. This raises recall, but many marginal non-default customers also cross the threshold, so precision drops.

For Taiwan CatBoost, threshold 0.15 is intentionally aggressive: recall=0.8093, precision=0.3514, FP=1982. This should be interpreted as a high-sensitivity screening policy, not as a final automatic rejection rule.

## 5. Why false positives matter

False positives are non-default customers flagged as high risk. In a real credit workflow they can create unnecessary manual review, customer friction, opportunity cost, and potential unfair treatment concerns. The original FN=5, FP=1 cost setup made sense for default capture, but the next revision should explicitly constrain FP volume, precision, specificity, and review capacity.

## 6. Why this model should not be presented as automatic rejection

The current low-threshold models are not precise enough for automatic rejection. A Taiwan CatBoost precision of 0.3514 means many predicted high-risk cases are actually non-default. The safer framing is: use the model as a screening/risk-prioritization system with manual review, not as a standalone automatic deny system.

## 7. What the next experiments will try to fix

The next experiments should search for constrained operating points that reduce FP, improve precision and specificity, preserve a minimum recall floor, and keep expected cost within an acceptable tolerance. The natural next step is constrained threshold search and manual-review band optimization with explicit constraints on precision, specificity, recall, FP count, and review coverage.
