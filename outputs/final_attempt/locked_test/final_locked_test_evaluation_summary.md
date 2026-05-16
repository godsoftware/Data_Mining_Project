# Final Locked Test Evaluation Summary

This file reports held-out evidence only. The final policies were locked in `final_selected_locked_policies.json` before this evaluation. No threshold, model, calibration, or policy was changed after seeing test results.

## 1. Yeni final system V2'den iyi mi?

- Taiwan: not uniformly. The new XGBoost manual-review policy has slightly higher recall, but lower precision/specificity and higher review-adjusted cost than the V2 CatBoost manual-review reference.
- HELOC: unchanged operationally; the final operational policy remains the V2 Scorecard manual-review policy.

## 2. Hangi metrikte iyi?

- Taiwan operational XGBoost improves recall by +0.0219 versus V2.
- SCRE-Optimized review prioritization improves Taiwan capture@20 by +0.0023 and HELOC capture@20 by +0.0068.

## 3. Hangi metrikte kotu?

- Taiwan operational XGBoost has precision delta -0.0297, specificity delta -0.0246, and review-adjusted cost delta +70.0 versus V2.
- HELOC operational system is the same as V2, so there is no operational degradation from V2 on the locked final policy.

## 4. Skor artisi mi, operational denge mi saglandi?

- The strongest improvement is not a simple score increase. It is a clearer separation of roles: XGBoost/Scorecard for operational screening, SCRE-Optimized for review prioritization, and interpretable benchmarks for governance.

## 5. Automatic rejection uygun mu?

- No. Precision remains limited and manual-review policies explicitly route uncertain cases to review. These results should not be framed as automatic credit rejection.

## 6. Screening/manual review uygun mu?

- Yes, conditionally. The evidence supports screening and review prioritization, especially using manual-review bands and top-k review capacity analysis.

## Held-out Operational Rows

| Dataset | System | Precision | Recall | Specificity | FP | FN | Review Cost | MR Rate | Capture@20 | Comment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| taiwan | Final operational screening / XGBoost xgboost_scale_pos_weight_5 | 0.4991 | 0.5968 | 0.8299 | 795 | 220 | 2775.5 | 0.2935 | NA | vs V2: precision -0.0297, recall +0.0219, specificity -0.0246, cost +70.0 |
| taiwan | Review prioritization / SCRE-Optimized probability ranking | NA | NA | NA | NA | NA | NA | NA | 0.5124 | vs V2: capture@20 +0.0023 |
| heloc | Final operational screening / Best Scorecard | 0.6834 | 0.8463 | 0.5744 | 403 | 158 | 764.5 | 0.2699 | 0.3317 | vs V2: precision +0.0000, recall +0.0000, specificity +0.0000, cost +0.0, capture@20 +0.0000 |
| heloc | Review prioritization / SCRE-Optimized probability ranking | NA | NA | NA | NA | NA | NA | NA | 0.3385 | vs V2: capture@20 +0.0068 |

## Leakage Status

- Test set used for selection: NO.
- Test set used for threshold changes: NO.
- Test set used for calibration changes: NO.
- Test ranking produced: NO.
