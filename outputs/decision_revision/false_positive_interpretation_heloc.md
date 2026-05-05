# False Positive Error Analysis: HELOC

No model was trained. Existing selected probability policies were evaluated on the held-out test split.

## System Comparison

| System | FP Count | FP Closer To | FP Looks Risky? | Overall FP-vs-TP SMD | Overall FP-vs-TN SMD | Comment |
|---|---:|---|---|---:|---:|---|
| Best Scorecard policy | 526 | TP | YES | 0.3077 | 0.8018 | FP profile is closer to true defaults than true non-defaults. |
| Best manual-review high-risk bucket | 544 | TP | YES | 0.3130 | 0.7861 | FP profile is closer to true defaults than true non-defaults. |
| Best SCRE policy | 544 | TP | YES | 0.3130 | 0.7861 | FP profile is closer to true defaults than true non-defaults. |
| Best precision-constrained CatBoost | 780 | TP | YES | 0.4273 | 0.7200 | FP profile is closer to true defaults than true non-defaults. |
| Old Best CatBoost cost-threshold | 798 | TP | YES | 0.4339 | 0.7034 | FP profile is closer to true defaults than true non-defaults. |

## FP Profile Indicators

| System | High utilization signal | Negative trade signal | Mean ExternalRiskEstimate | Mean revolving burden |
|---|---:|---:|---:|---:|
| Old Best CatBoost cost-threshold | 0.3985 | 0.5288 | 75.2005 | 26.2374 |
| Best precision-constrained CatBoost | 0.4077 | 0.5385 | 74.9718 | 26.6899 |
| Best manual-review high-risk bucket | 0.4945 | 0.6415 | 71.5699 | 33.2546 |
| Best SCRE policy | 0.4945 | 0.6415 | 71.5699 | 33.2546 |
| Best Scorecard policy | 0.5114 | 0.6426 | 71.2205 | 34.0192 |

## Interpretation

- Systems where FP is closer to TP than TN: 5/5.
- Low precision should not automatically be read as meaningless alarms; FP borrowers can still exhibit high-risk characteristics.
- Automatic rejection is not appropriate from FP profile alone. Manual review is the safer operational framing.
