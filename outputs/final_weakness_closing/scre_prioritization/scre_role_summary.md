# SCRE Review-Prioritization Role Summary

## Protocol
- SCRE is not promoted as the final classifier.
- No new model training, feature creation, or test-set policy selection was performed.
- Evidence is limited to existing SCRE probabilities, capacity-review outputs, SCRE weights, and existing stability/faithfulness artifacts.

## Top-K review prioritization

### Taiwan
| Dataset | SCRE Version | Capture@10 | Capture@20 | Precision@20 | Lift@20 | Comment |
| --- | --- | --- | --- | --- | --- | --- |
| taiwan | SCRE-Optimized | 0.321 | 0.529 | 0.585 | 2.645 | Useful as review ranking evidence, not automatic rejection. |
| taiwan | SCRE-Pareto | 0.318 | 0.526 | 0.582 | 2.630 | Useful as review ranking evidence, not automatic rejection. |

### HELOC
| Dataset | SCRE Version | Capture@10 | Capture@20 | Precision@20 | Lift@20 | Comment |
| --- | --- | --- | --- | --- | --- | --- |
| heloc | SCRE-Optimized | 0.182 | 0.340 | 0.884 | 1.698 | Useful as review ranking evidence, not automatic rejection. |
| heloc | SCRE-Pareto | 0.174 | 0.334 | 0.868 | 1.669 | Useful as review ranking evidence, not automatic rejection. |

## Reliability score decomposition
- SCRE weight/decomposition rows are stored in `scre_score_decomposition.csv`.
- Highest-weight components by dataset/version: heloc/SCRE-Optimized: catboost=0.595; heloc/SCRE-Optimized: woe_scorecard_logistic_regression=0.285; heloc/SCRE-Pareto: catboost=0.478; heloc/SCRE-Pareto: optuna_best_xgboost=0.309; taiwan/SCRE-Optimized: optuna_best_catboost=0.489; taiwan/SCRE-Optimized: monotonic_xgboost=0.307; taiwan/SCRE-Pareto: xgboost=0.237; taiwan/SCRE-Pareto: catboost=0.217.
- Stability/faithfulness component values in some SCRE weight tables are neutral placeholders where explicitly marked; they should not be overclaimed as direct SCRE-specific SHAP stability.

## Rank stability
- SCRE-Optimized vs SCRE-Pareto validation rank agreement: taiwan: top10=0.975, top20=0.975, tau=0.957, spearman=0.997; heloc: top10=0.919, top20=0.959, tau=0.959, spearman=0.998.
- `scre_rank_stability.csv` also includes existing SHAP Kendall's W rows as reference evidence. Those rows are model-family stability references, not proof that the SCRE ensemble itself was retrained over seeds.

## Questions
1. SCRE final classifier mı? **NO.** V2 CatBoost/Scorecard policies remain final operational evidence.
2. SCRE review prioritization için işe yarıyor mu? **YES.** It provides useful top-k ranking evidence, especially for review-capacity discussion.
3. SCRE top-20% default capture değerinde iyi mi? **YES as ranking support.** Taiwan: validation Capture@20=0.529, Precision@20=0.585, Lift@20=2.645; locked-test Capture@20=0.512. HELOC: validation Capture@20=0.340, Precision@20=0.884, Lift@20=1.698; locked-test Capture@20=0.339.
4. SCRE calibration/stability/faithfulness açısından ne katıyor? It gives a structured reliability framework that combines performance, calibration, cost, and available stability/faithfulness evidence, but not all stability/faithfulness values are direct SCRE-specific measurements.
5. SCRE hangi claim ile sunulmalı? SCRE should be presented as a reliability-aware framework for model comparison, weighted integration, and review prioritization.
6. SCRE hangi claim ile sunulmamalı? It should not be claimed as a universally superior classifier or automatic rejection model.
