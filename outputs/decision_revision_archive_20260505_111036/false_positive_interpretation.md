# False Positive Error Analysis: TAIWAN

No model was trained. Existing selected probability policies were evaluated on the held-out test split.

## System Comparison

| System | FP Count | FP Closer To | FP Looks Risky? | Overall FP-vs-TP SMD | Overall FP-vs-TN SMD | Comment |
|---|---:|---|---|---:|---:|---|
| Best manual-review high-risk bucket | 827 | TP | YES | 0.2516 | 0.6971 | FP profile is closer to true defaults than true non-defaults. |
| Best SCRE policy | 827 | TP | YES | 0.2516 | 0.6971 | FP profile is closer to true defaults than true non-defaults. |
| Best Scorecard policy | 948 | TP | YES | 0.2820 | 0.7090 | FP profile is closer to true defaults than true non-defaults. |
| Best precision-constrained CatBoost | 1273 | TP | YES | 0.3072 | 0.5535 | FP profile is closer to true defaults than true non-defaults. |
| Old Best CatBoost cost-threshold | 1982 | TP | YES | 0.3291 | 0.4572 | FP profile is closer to true defaults than true non-defaults. |

## FP Profile Indicators

| System | PAY_0 delayed | delay_count>=2 | high utilization | low payment/bill | Mean delay_count |
|---|---:|---:|---:|---:|---:|
| Old Best CatBoost cost-threshold | 0.3199 | 0.2709 | 0.2447 | 0.5590 | 1.1529 |
| Best precision-constrained CatBoost | 0.4682 | 0.4116 | 0.2506 | 0.5923 | 1.6787 |
| Best manual-review high-risk bucket | 0.6312 | 0.5683 | 0.2781 | 0.6771 | 2.2624 |
| Best SCRE policy | 0.6312 | 0.5683 | 0.2781 | 0.6771 | 2.2624 |
| Best Scorecard policy | 0.6055 | 0.5274 | 0.2954 | 0.6825 | 2.1392 |

## Interpretation

- Systems where FP is closer to TP than TN: 5/5.
- Low precision should not automatically be read as meaningless alarms; FP borrowers can still exhibit high-risk characteristics.
- Automatic rejection is not appropriate from FP profile alone. Manual review is the safer operational framing.
