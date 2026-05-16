# Manual-Review Metric Definitions

Manual-review policy is not the same as ordinary binary classification.

In a binary classifier, every customer is assigned to one of two actions:

1. Predicted non-default
2. Predicted default / high risk

In a manual-review policy, the model produces three decision buckets:

1. Low risk
2. Manual review
3. High risk

Therefore, manual-review results should ideally be reported as a three-bucket decision table:

| Decision Bucket | Actual Non-default | Actual Default |
|---|---:|---:|
| Low risk | low-risk non-defaults | low-risk defaults |
| Manual review | reviewed non-defaults | reviewed defaults |
| High risk | high-risk non-defaults | high-risk defaults |

## Core Definitions

### high_risk_precision

Among customers automatically placed in the high-risk bucket, the share that truly defaulted.

Formula:

`high_risk_precision = high_risk_defaults / (high_risk_defaults + high_risk_nondefaults)`

Interpretation:

Higher is better. This measures whether the automatic high-risk bucket is too noisy.

### high_risk_recall

Among all actual defaults, the share captured in the high-risk bucket.

Formula:

`high_risk_recall = high_risk_defaults / total_defaults`

Interpretation:

Higher is better. This measures how many true defaults are automatically captured as high risk.

### low_risk_default_rate

Among customers automatically placed in the low-risk bucket, the share that actually defaulted.

Formula:

`low_risk_default_rate = low_risk_defaults / (low_risk_defaults + low_risk_nondefaults)`

Interpretation:

Lower is better. This measures default leakage into the automatically safe bucket.

### manual_review_rate

Share of customers routed to manual review.

Formula:

`manual_review_rate = manual_review_count / total_customers`

Interpretation:

Lower is usually better for operational workload, but too low a review rate may force too many automatic decisions.

### auto_decision_rate

Share of customers receiving an automatic low-risk or high-risk decision.

Formula:

`auto_decision_rate = 1 - manual_review_rate`

Interpretation:

Higher means more automation, but it should not come at the cost of unsafe low-risk leakage or excessive false positives.

### review_adjusted_cost

Operational cost that includes automatic-decision errors and manual-review workload.

Typical formula under the assumption that manual review resolves reviewed cases correctly:

`review_adjusted_cost = auto_FP * fp_cost + auto_FN * fn_cost + manual_review_count * review_cost`

Where:

- `auto_FP`: non-default customers automatically placed in high risk
- `auto_FN`: default customers automatically placed in low risk
- `fp_cost`: cost of falsely flagging a non-default as high risk
- `fn_cost`: cost of missing an actual default in the low-risk bucket
- `review_cost`: workload cost per manually reviewed case

## Important Warning

Binary precision, recall, specificity, and F1 should not be mixed with manual-review bucket metrics in the same table unless the denominator is explicitly stated.

For manual-review policies:

- binary confusion-matrix metrics describe a two-class collapse of a three-action policy;
- high-risk metrics describe only the high-risk automatic bucket;
- low-risk default leakage describes the automatic safe bucket;
- review-adjusted cost describes operational workflow cost.

If these definitions are mixed without clear denominators, the table becomes audit-unsafe. This is why Final Attempt locked-test manual-review rows should not be used as final evidence while the metric-consistency audit remains BLOCKED.
