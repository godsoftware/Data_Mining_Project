# Cost Sensitivity and Decision Curve Summary

## Protocol
- Existing locked policies and candidate policies only.
- No model training, feature creation, or test-set policy selection.
- Cost sensitivity is reported for validation and locked-test evidence separately.
- Cost rank uses review-adjusted cost for manual-review/top-K policies and binary expected cost for threshold-only policies.
- Decision curves are computed from existing probability scores across threshold probabilities 0.01-0.80.

## Taiwan cost stability
| System | Stable? | Mean rank | Best Cost Scenarios | Weak Scenarios | Comment |
| --- | --- | --- | --- | --- | --- |
| V2 Taiwan CatBoost manual-review | YES | 2.33 | FN=2,FP=1,RC=0.1; FN=3,FP=1,RC=0.1; FN=3,FP=1,RC=0.25; FN=3,FP=1,RC=0.5 | FN=2,FP=1,RC=0.5; FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=1 | Review-cost sensitive |
| Taiwan capacity SCRE top25 | YES | 2.98 | FN=2,FP=1,RC=0.1; FN=2,FP=1,RC=0.25; FN=2,FP=1,RC=0.5; FN=5,FP=2,RC=0.1 | FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=1; FN=3,FP=1,RC=2 | Review-cost sensitive |
| Taiwan recall candidate MR band | NO | 3.33 | FN=3,FP=1,RC=0.1; FN=3,FP=1,RC=0.25; FN=3,FP=1,RC=0.5; FN=5,FP=1,RC=0.1 | FN=2,FP=1,RC=0.5; FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=1 | Binary/score benchmark |
| Taiwan capacity SCRE top20 | NO | 3.70 | FN=2,FP=1,RC=0.25; FN=2,FP=1,RC=0.5; FN=5,FP=3,RC=0.1; FN=5,FP=3,RC=0.25 | FN=5,FP=1,RC=2 | Review-cost sensitive |
| SCRE-Optimized ranking threshold 0.50 | NO | 4.22 | FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=1; FN=3,FP=1,RC=2 | none | Binary/score benchmark |
| CatBoost benchmark threshold 0.50 | NO | 5.22 | FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=1; FN=3,FP=1,RC=2 | none | Binary/score benchmark |
| Scorecard benchmark threshold 0.50 | NO | 6.22 | none | none | Binary/score benchmark |

## HELOC cost stability
| System | Stable? | Mean rank | Best Cost Scenarios | Weak Scenarios | Comment |
| --- | --- | --- | --- | --- | --- |
| V2 HELOC Scorecard manual-review | YES | 1.65 | FN=2,FP=1,RC=0.1; FN=2,FP=1,RC=0.25; FN=2,FP=1,RC=0.5; FN=3,FP=1,RC=0.1 | FN=2,FP=1,RC=0.5; FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=1 | Review-cost sensitive |
| SCRE-Optimized ranking threshold 0.50 | YES | 1.80 | FN=2,FP=1,RC=0.1; FN=2,FP=1,RC=0.25; FN=2,FP=1,RC=0.5; FN=2,FP=1,RC=1 | none | Binary/score benchmark |
| CatBoost benchmark threshold 0.50 | NO | 3.23 | FN=2,FP=1,RC=1; FN=2,FP=1,RC=2; FN=3,FP=1,RC=2; FN=5,FP=2,RC=2 | none | Binary/score benchmark |
| HELOC specificity candidate threshold | NO | 3.48 | FN=5,FP=1,RC=2 | none | Binary/score benchmark |
| Scorecard benchmark threshold 0.50 | NO | 4.85 | none | none | Binary/score benchmark |
| HELOC capacity SCRE top25 | NO | 6.03 | none | FN=2,FP=1,RC=2 | Review-cost sensitive |
| HELOC capacity SCRE top20 | NO | 6.97 | none | none | Review-cost sensitive |

## Decision curve interpretation
- Taiwan validation threshold ranges led by: SCRE-Optimized ranking threshold 0.50: 0.01-0.80; Taiwan capacity SCRE top20: 0.01-0.80; Taiwan capacity SCRE top25: 0.01-0.80.
- HELOC validation threshold ranges led by: HELOC capacity SCRE top20: 0.01-0.80; HELOC capacity SCRE top25: 0.01-0.80; SCRE-Optimized ranking threshold 0.50: 0.01-0.80.
- A model is useful only where its net benefit is above both treat-all and treat-none.

## Questions
1. V2 hangi cost senaryolarinda hala iyi? See the V2 rows in the cost stability tables; V2 remains defensible when review_cost is not too high and FN cost dominates FP cost.
2. Review cost yukselince manual-review policy bozuluyor mu? **YES.** Review-adjusted cost rises directly with review_count, so high review_cost weakens manual-review and capacity policies.
3. Taiwan recall-improving candidate hangi senaryoda mantikli? It is most defensible when FN cost is high and low review_cost is plausible; otherwise extra FP/review exposure weakens it.
4. HELOC specificity candidate hangi senaryoda mantikli? It is most defensible when FP cost or specificity pressure is high; it sacrifices recall and can raise FN-driven cost.
5. Decision curve hangi sistemi destekliyor? Use the validation ranges above; SCRE ranking is strongest in some ranking-oriented ranges, while CatBoost/Scorecard remain core benchmarks.
6. Treat-all / treat-none karsisinda model faydali mi? **YES in useful threshold ranges**, not uniformly across all thresholds.
7. Cost belirsizligi sonucu degistiriyor mu? **PARTIALLY.** It does not automatically replace V2, but it identifies V3 candidates under explicit cost/workload assumptions.
