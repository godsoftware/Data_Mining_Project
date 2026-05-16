# Week 13 Progress Report

**Course:** ECE 565 - Data Mining
**Student:** Resul Ozkale
**Student ID:** 2540041007
**Week:** Week 13 - Final Attempt Stress Testing, Validation Protocol Audit, and Report-Readiness Check
**Date:** May 11, 2026
**Project:** SCRE-Credit: Stability- and Cost-Regularized Ensemble for Credit Risk

## Executive Summary
This week extended the Week 12 final-evaluation work into a stricter validation and audit phase. The main achievement was not a new dataset or a new feature set, but a cleaner decision protocol: model and policy selection must be validation-only, while the test set is reserved for locked held-out evidence. The V2 revision reached READY status under this protocol. A later final-attempt package tested literature-style resampling, advanced imbalance boosting, deep tabular models, interpretable middle models, capacity-aware review, and decision curves. Those experiments produced useful negative and diagnostic evidence, but the final-attempt audit ended BLOCKED because some manual-review locked-test metric rows were not internally consistent. Therefore, the safest current report position is: V2 is the defensible report-ready baseline; the final-attempt experiments are useful stress tests but should not replace V2 until the metric-consistency issue is fixed.

## 1. Work Completed This Week
- **V2 protocol reset (Completed):** Separated validation-only model/policy selection from held-out test evaluation. Previous test-ranking problem was explicitly corrected.
- **Manual-review cost correction (Completed):** Separated binary FN/FP cost from review-adjusted operational cost. Manual review is no longer treated as the same object as binary classification cost.
- **Validation candidate registry (Completed):** Created validation-only candidate registries for Taiwan and HELOC without test columns.
- **Locked policy evaluation (Completed):** Locked final policies were evaluated on test only after validation selection. No policy change was allowed after test evidence.
- **Final audit V2 (Completed):** V2 audit result was READY with no critical, high, or medium issues.
- **Strict literature reproduction (Completed):** SMOTE, BorderlineSMOTE, SMOTE-Tomek, KMeansSMOTE, boosting, and class strategies were tested under leakage-free protocol.
- **Advanced imbalance boosting (Completed):** CatBoost, XGBoost, and LightGBM class-weight and calibration variants were tested.
- **Deep tabular reproduction (Completed):** MLP/BP neural network variants were tested to check literature-style neural claims.
- **Interpretable middle models (Completed):** EBM/monotonic/scorecard-style alternatives were evaluated as governance-friendly options.
- **Capacity-aware review and decision curve (Completed):** Models were evaluated as review-prioritization systems using capture@K, lift@K, and net benefit.
- **Final attempt audit (Completed but blocked):** No leakage was detected, but locked-test manual-review metric consistency failed, so the final attempt is not report-ready yet.

## 2. Key Numerical Evidence
### V2 locked held-out baseline
| Dataset | Final V2 model/policy | Precision | Recall | Specificity | FP | FN | Cost | Manual review rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Taiwan | Best CatBoost manual-review band | 0.529 | 0.575 | 0.854 | 680 | 564 | 2705.500 | 0.295 |
| HELOC | Best Scorecard manual-review band | 0.683 | 0.846 | 0.574 | 403 | 158 | 764.500 | 0.270 |

### Final attempt locked test evidence
| Dataset | Final attempt system | Precision | Recall | Specificity | FP | FN | Cost | Manual review rate | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Taiwan | XGBoost manual-review final operational system | 0.499 | 0.597 | 0.830 | 795 | 220 | 2775.500 | 0.293 | Slight recall gain, but lower precision/specificity and higher cost than V2. |
| HELOC | Scorecard manual-review final operational system | 0.683 | 0.846 | 0.574 | 403 | 158 | 764.500 | 0.270 | Essentially preserves the V2 HELOC recommendation. |

## 3. What Is New Compared With Week 12?
- **Validation-only selection protocol - New:** Week 12 still reported final results, but Week 13 explicitly repaired the risk that final ranking could be read as test-based.
- **Manual-review cost model - New:** Week 13 separates automation residual cost, review workload cost, and imperfect-review assumptions.
- **False-positive reduction framing - New:** The project moved from pure cost minimization to precision, specificity, FP reduction, and manual-review usability.
- **Strict literature reproduction - New:** Week 13 tested whether high literature scores can be reproduced without leakage and with natural test distribution.
- **Advanced imbalance boosting - New:** Week 13 tested class-weight, scale_pos_weight, max_delta_step, and calibration variants.
- **Deep tabular / BP-NN reproduction - New:** Week 13 tested whether MLP/BP-style neural models improve over boosting/scorecard baselines.
- **EBM and monotonic middle models - New:** Week 13 searched for a stronger but more explainable alternative between Scorecard and black-box boosting.
- **Capacity-aware review - New:** Week 13 reframed the system as review prioritization using top-10/20/30 percent capture and lift.
- **Final brutal audit - New:** Final attempt was blocked because metric rows were inconsistent, even though no test leakage was detected.

## 4. Results and Interpretation
The V2 revision is currently the strongest defensible basis for the homework report. It improved the methodological protocol compared with Week 12 by preventing test-set ranking and by locking policies before test evaluation. Taiwan's V2 CatBoost manual-review policy reduced the extremely aggressive false-positive behavior seen in the old cost-threshold model. HELOC remained best represented by the Scorecard manual-review policy, which is also the most interpretable benchmark.

The final attempt did not clearly replace V2. Literature-inspired resampling did not reproduce large performance jumps under the leakage-free protocol. KMeansSMOTE did not produce a reliable breakthrough. Deep tabular models improved some raw metrics in places but did not deliver a better operational policy. Interpretable middle models, especially monotonic LightGBM and Scorecard, remain valuable for governance, but they did not make the project a fully automatic rejection system. The capacity-aware analysis is one of the most useful additions: it supports describing the model as a screening and manual-review prioritization tool.

## 5. Problems Encountered
- **Final attempt audit BLOCKED:** The final locked-test manual-review rows failed metric-consistency checks. Precision/recall/specificity/F1 could not be recomputed cleanly from the displayed confusion matrix columns.
- **Validation winner did not hold on Taiwan test:** The final-attempt XGBoost manual-review system improved recall slightly but worsened precision, specificity, FP count, and cost compared with V2.
- **Manual-review denominators are complex:** Manual-review policies need separate auto-decision confusion matrices, full-population rates, and review workload cost. These should not be collapsed into one binary cost table.
- **SCRE should not be overclaimed:** SCRE remains useful as a reliability-aware framework and ranking tool, but it does not dominate CatBoost or Scorecard in all operational metrics.
- **Automatic rejection is not justified:** Precision and false-positive behavior still require manual review. The safe use case is screening and prioritization, not automatic credit denial.

## 6. Literature Comparison
Week 13 directly addressed the concern that some published Taiwan credit-risk papers report very high scores using resampling or neural networks. Under the stricter protocol used here, resampling is applied only inside training folds and the test set remains at the natural class distribution. Under this setup, the large jumps often associated with SMOTE/KMeansSMOTE or BP neural networks were not observed. This suggests that at least part of the gap between this project and some literature scores may come from protocol differences, thresholding choices, resampled test distributions, or less strict leakage controls. This is an important methodological contribution rather than a failure.

## 7. Plan for Next Week
1. Rebuild the final-attempt locked-test manual-review tables with explicit denominator definitions: full-population metrics, auto-decision-only metrics, and manual-review workload metrics must be separated.
2. Re-run the final-attempt brutal audit after the metric-consistency repair.
3. Decide whether the report should use V2 as the final result or include the final attempt only as an additional stress-test appendix.
4. Begin writing the final project report using safe claims: validation-only selection, held-out evidence, no automatic rejection claim, and SCRE as a framework rather than a universally dominant predictor.
5. Prepare final figures and tables: V2 locked results, manual-review workflow, capacity@K/lift curves, decision curve, and claim-control matrix.

## 8. Self Assessment
I completed the planned Week 12 follow-up work: false-positive reduction, cost sensitivity, manual-review policies, imbalance experiments, and decision-curve analysis. The project is technically stronger because it now has a clear validation-only selection protocol and a strong audit trail. However, I should not present the final-attempt locked-test table as report-ready until the metric-consistency issue is fixed. My honest estimate is that the project is about 90 percent ready for final report writing if V2 is used as the main result, and about 80 to 85 percent ready if the final-attempt package must be included as the final main result.

## Safe Current Claim
The project does not support automatic credit rejection. It supports a credit-risk screening and manual-review prioritization framework. The safest final position is that V2 provides the current defensible operational baseline, while the final-attempt experiments show that leakage-free literature reproduction, deeper models, and stronger imbalance strategies do not clearly replace the V2 manual-review decision policy. SCRE-Credit remains useful as a reliability-aware framework and review-ranking component, not as a universally superior standalone predictor.