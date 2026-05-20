# Instance-Dependent Cost Summary

## Protocol
- Existing V2 and weakness-closing candidate scores/policies only.
- No model training, no feature generation, no threshold selection, and no test-set policy selection.
- Costs are proxy-based sensitivity analyses, not observed bank losses.
- Manual-review policies use low-risk defaults as automatic FN, high-risk non-defaults as automatic FP, and a fixed review friction cost of 0.5 per reviewed case.

## Taiwan locked-test proxy-cost leaders
| cost_definition | system | total_instance_dependent_cost | delta_vs_v2 | high_exposure_default_missed_count | cost_rank |
| --- | --- | --- | --- | --- | --- |
| taiwan_bill_weighted | Taiwan recall candidate MR band | 3398.9 | -242.0 | 57 | 1.0 |
| taiwan_bill_weighted | V2 Taiwan CatBoost manual-review | 3640.9 | 0.0 | 76 | 2.0 |
| taiwan_bill_weighted | Taiwan capacity SCRE top25 | 5125.0 | 1484.1 | 122 | 3.0 |
| taiwan_bill_weighted | Taiwan capacity SCRE top20 | 5539.5 | 1898.6 | 129 | 4.0 |
| taiwan_capped_hybrid | Taiwan recall candidate MR band | 2766.5 | -2.0 | 58 | 1.0 |
| taiwan_capped_hybrid | V2 Taiwan CatBoost manual-review | 2768.5 | 0.0 | 73 | 2.0 |
| taiwan_capped_hybrid | Taiwan capacity SCRE top25 | 3409.7 | 641.2 | 111 | 3.0 |
| taiwan_capped_hybrid | Taiwan capacity SCRE top20 | 3666.0 | 897.5 | 117 | 4.0 |
| taiwan_limit_weighted | Taiwan recall candidate MR band | 3587.8 | -207.4 | 65 | 1.0 |
| taiwan_limit_weighted | V2 Taiwan CatBoost manual-review | 3795.2 | 0.0 | 78 | 2.0 |
| taiwan_limit_weighted | Taiwan capacity SCRE top25 | 5556.0 | 1760.7 | 145 | 3.0 |
| taiwan_limit_weighted | Taiwan capacity SCRE top20 | 6138.3 | 2343.0 | 160 | 4.0 |
| taiwan_utilization_weighted | Taiwan recall candidate MR band | 3266.9 | -261.6 | 27 | 1.0 |
| taiwan_utilization_weighted | V2 Taiwan CatBoost manual-review | 3528.5 | 0.0 | 43 | 2.0 |
| taiwan_utilization_weighted | Taiwan capacity SCRE top25 | 5266.0 | 1737.4 | 136 | 3.0 |
| taiwan_utilization_weighted | Taiwan capacity SCRE top20 | 5823.3 | 2294.8 | 159 | 4.0 |

## HELOC locked-test proxy-cost leaders
| cost_definition | system | total_instance_dependent_cost | delta_vs_v2 | high_exposure_default_missed_count | cost_rank |
| --- | --- | --- | --- | --- | --- |
| heloc_capped_hybrid | V2 HELOC Scorecard manual-review | 737.1 | 0.0 | 0 | 1.0 |
| heloc_capped_hybrid | SCRE-Optimized ranking threshold 0.50 | 971.7 | 234.7 | 0 | 2.0 |
| heloc_capped_hybrid | HELOC specificity candidate threshold | 1239.0 | 501.9 | 5 | 3.0 |
| heloc_capped_hybrid | SCRE-Pareto ranking threshold 0.50 | 1280.3 | 543.3 | 4 | 4.0 |
| heloc_revolving_burden_weighted | V2 HELOC Scorecard manual-review | 768.9 | 0.0 | 0 | 1.0 |
| heloc_revolving_burden_weighted | SCRE-Optimized ranking threshold 0.50 | 1270.0 | 501.2 | 6 | 2.0 |
| heloc_revolving_burden_weighted | HELOC specificity candidate threshold | 1793.0 | 1024.1 | 15 | 3.0 |
| heloc_revolving_burden_weighted | SCRE-Pareto ranking threshold 0.50 | 1875.1 | 1106.2 | 13 | 4.0 |
| heloc_risk_exposure_proxy | V2 HELOC Scorecard manual-review | 765.4 | 0.0 | 0 | 1.0 |
| heloc_risk_exposure_proxy | SCRE-Optimized ranking threshold 0.50 | 1310.2 | 544.8 | 0 | 2.0 |
| heloc_risk_exposure_proxy | HELOC specificity candidate threshold | 1801.0 | 1035.6 | 1 | 3.0 |
| heloc_risk_exposure_proxy | SCRE-Pareto ranking threshold 0.50 | 1891.0 | 1125.6 | 2 | 4.0 |

## Answers
1. Sabit maliyet ile instance-dependent maliyet aynı final kararı veriyor mu? **Taiwan: PARTIALLY.** The recall-improving candidate has lower proxy-cost in all Taiwan proxy definitions, but it is still a V3/audit candidate. **HELOC: YES.** V2 remains best under all HELOC proxy definitions.
2. V2 yüksek exposure defaultları kaçırıyor mu? **YES, but not catastrophically.** Taiwan V2 misses up to 78 high-exposure defaults depending on the proxy; HELOC V2 misses up to 0.
3. Recall-improving policy yüksek exposure defaultlarda daha iyi mi? **PARTIALLY.** Taiwan recall candidate reduces the worst-case high-exposure missed-default count to 65, but it increases false alarms and remains a V3/audit candidate.
4. Instance-dependent cost final policy'yi değiştiriyor mu? **PARTIALLY - some candidates lower proxy-cost in specific definitions (Taiwan recall candidate MR band), but this is sensitivity evidence, not an audited replacement.** For HELOC: **NO - V2 remains the safest final operational evidence under proxy-cost sensitivity.**
5. Bu analiz gerçek maliyet eksikliğini ne kadar kapatıyor? It improves transparency by testing exposure-weighted cost assumptions with existing financial proxies, but it remains a sensitivity analysis rather than a true bank loss model.
