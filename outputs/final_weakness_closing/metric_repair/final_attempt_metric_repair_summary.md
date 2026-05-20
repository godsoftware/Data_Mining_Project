# Final Attempt Manual-Review Metric Repair

## Scope
This repair does not train a model, select a threshold, change a policy, or use the test set for decision-making. It only rewrites existing Final Attempt locked-test manual-review rows as explicit 3x2 decision tables.

## Correct metric definition
Manual-review policies are three-way decision policies, not ordinary binary classifiers. The repaired files separate Low risk, Manual review, and High risk buckets, then compute high-risk precision/recall, low-risk default rate, manual-review rate, auto-decision rate, automation residual cost, and review-adjusted cost.

## Taiwan repaired rows
| dataset | system_name | model | high_risk_precision_expected | high_risk_recall_expected | low_risk_default_rate_expected | manual_review_rate_expected | review_adjusted_cost_expected | binary_metrics_mixed_denominator_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | Final operational screening | XGBoost xgboost_scale_pos_weight_5 | 0.499055 | 0.596835 | 0.082956 | 0.293500 | 2775.500000 | True |
| taiwan | Interpretable benchmark | Monotonic LightGBM conservative | 0.497150 | 0.591560 | 0.082217 | 0.288833 | 2765.500000 | True |

## HELOC repaired rows
| dataset | system_name | model | high_risk_precision_expected | high_risk_recall_expected | low_risk_default_rate_expected | manual_review_rate_expected | review_adjusted_cost_expected | binary_metrics_mixed_denominator_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heloc | Final operational screening | Best Scorecard | 0.683425 | 0.846304 | 0.112426 | 0.269873 | 764.500000 | False |
| heloc | Interpretable benchmark | WOE scorecard unweighted | 0.689683 | 0.825875 | 0.132353 | 0.273418 | 787.000000 | True |

## Audit result after repair
- Repaired 3x2 metric identity failures: 0
- Original binary denominator mismatch flags documented: 3
- Final Attempt final operational status after repair: NOT FINAL
- Intended use: appendix / robustness / stress-test evidence only, pending any future full audit.

## Interpretation
The previous BLOCKED audit mixed binary confusion-matrix checks with manual-review bucket metrics. The repaired tables make the bucket definitions explicit. This removes the manual-review cost auditability gap for the repaired rows, but it does not make Final Attempt the final operational version. V2 remains the final operational evidence unless a later full audit explicitly replaces it.
