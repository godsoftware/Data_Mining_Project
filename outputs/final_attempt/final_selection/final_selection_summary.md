# Final Multi-Objective Selection Summary

Selection evidence: validation-only final-attempt outputs, validation capacity analysis, and validation decision-curve summaries.
Forbidden evidence: held-out test metrics, test ranks, test winners, or test-threshold tuning.

## TAIWAN Track Decisions

- Track A (Best raw classifier): XGBoost xgboost_spw_5_mds_1.
  Reason: Highest validation PR-AUC after tie-breaking by ROC-AUC, F1, and calibration. No test metric was consulted.
  Validation metrics: PR-AUC=0.5653, ROC-AUC=0.7862, precision=0.5231, recall=0.5727, specificity=0.8517, cost=2664.0.
- Track B (Best operational binary policy): XGBoost xgboost_scale_pos_weight_5.
  Reason: Lowest validation binary expected cost among eligible binary policies, with PR-AUC/ECE/F1 used only as validation tie-breakers.
  Validation metrics: PR-AUC=0.5619, ROC-AUC=0.7878, precision=0.4012, recall=0.7408, specificity=0.6861, cost=3187.0.
- Track C (Best manual-review policy): XGBoost xgboost_scale_pos_weight_5.
  Reason: Lowest validation review-adjusted cost under review_cost=0.5 among manual-review candidates satisfying MR-rate and high-risk quality constraints.
  Validation metrics: PR-AUC=0.5619, ROC-AUC=0.7878, precision=0.5138, recall=0.5878, specificity=0.8421, cost=2648.5.
- Track D (Best capacity-aware reviewer): SCRE-Optimized probability ranking.
  Reason: Highest validation default capture@20%, then precision@20%, lift@20%, and net value@20%.
  Capacity@20: default capture=0.5290, precision@20=0.5850, lift@20=2.6451.
- Track E (Best interpretable model): Monotonic LightGBM conservative.
  Reason: Best validation-governed interpretable candidate after enforcing manual-review feasibility; scorecard is preferred when it remains competitive and feasible.
  Validation metrics: PR-AUC=0.5616, ROC-AUC=0.7876, precision=0.5082, recall=0.5848, specificity=0.8393, cost=2668.0.
- Track F (Best research framework): SCRE-Optimized.
  Reason: SCRE variant selected for reliability integration, validation AUC/PR-AUC, and capacity-review support. This is a framework track, not a claim of universal predictive dominance.
  Validation metrics: PR-AUC=0.5628, ROC-AUC=0.7882, precision=0.3747, recall=0.7980, specificity=0.6219, cost=3107.0.
- Track G (Final recommended system): Operational screening: XGBoost xgboost_scale_pos_weight_5 (manual_review_band_mr_le_0_30); review prioritization: SCRE-Optimized probability ranking; interpretable benchmark: Monotonic LightGBM conservative; research framework: SCRE-Optimized..
  Reason: Taiwan changes partially from V2: the validation-only manual-review winner shifts marginally from V2 CatBoost to advanced XGBoost (review-adjusted cost 2648.5), while SCRE-Optimized is kept for top-k review prioritization and Monotonic LightGBM is the strongest interpretable middle model.

## HELOC Track Decisions

- Track A (Best raw classifier): MLP/BP Neural Network mlp_128m64_l2_0.001.
  Reason: Highest validation PR-AUC after tie-breaking by ROC-AUC, F1, and calibration. No test metric was consulted.
  Validation metrics: PR-AUC=0.8109, ROC-AUC=0.8044, precision=0.6662, recall=0.8802, specificity=0.5216, cost=1068.0.
- Track B (Best operational binary policy): WOE scorecard unweighted.
  Reason: Lowest validation binary expected cost among eligible binary policies, with PR-AUC/ECE/F1 used only as validation tie-breakers.
  Validation metrics: PR-AUC=0.7927, ROC-AUC=0.8026, precision=0.5890, recall=0.9698, specificity=0.2661, cost=850.0.
- Track C (Best manual-review policy): Best Scorecard.
  Reason: Lowest validation review-adjusted cost under review_cost=0.5 among manual-review candidates satisfying MR-rate and high-risk quality constraints.
  Validation metrics: PR-AUC=0.7951, ROC-AUC=0.8034, precision=0.7010, recall=0.8423, specificity=0.6103, cost=729.0.
- Track D (Best capacity-aware reviewer): SCRE-Optimized probability ranking.
  Reason: Highest validation default capture@20%, then precision@20%, lift@20%, and net value@20%.
  Capacity@20: default capture=0.3398, precision@20=0.8835, lift@20=1.6983.
- Track E (Best interpretable model): WOE scorecard unweighted.
  Reason: Best validation-governed interpretable candidate after enforcing manual-review feasibility; scorecard is preferred when it remains competitive and feasible.
  Validation metrics: PR-AUC=0.7927, ROC-AUC=0.8026, precision=0.7070, recall=0.8247, specificity=0.6294, cost=731.0.
- Track F (Best research framework): SCRE-Optimized.
  Reason: SCRE variant selected for reliability integration, validation AUC/PR-AUC, and capacity-review support. This is a framework track, not a claim of universal predictive dominance.
  Validation metrics: PR-AUC=0.8056, ROC-AUC=0.8080, precision=0.5694, recall=0.9825, specificity=0.1943, cost=853.0.
- Track G (Final recommended system): Operational screening: Best Scorecard (manual_review_band); review prioritization: SCRE-Optimized probability ranking; interpretable benchmark: WOE scorecard unweighted; research framework: SCRE-Optimized..
  Reason: HELOC operational selection remains close to V2: Scorecard-style manual review remains the cleanest validation-only operational policy, while SCRE-Optimized ranks best for capacity@20 review prioritization.

## Prompt Questions

1. En yüksek skor modeli hangisi?
   - Taiwan: XGBoost xgboost_spw_5_mds_1. HELOC: MLP/BP Neural Network mlp_128m64_l2_0.001.
2. En dengeli operational model hangisi?
   - Taiwan: XGBoost xgboost_scale_pos_weight_5. HELOC: Best Scorecard.
3. Manual-review için en iyi model hangisi?
   - Taiwan: XGBoost xgboost_scale_pos_weight_5. HELOC: Best Scorecard.
4. Capacity@20 için en iyi model hangisi?
   - Taiwan: SCRE-Optimized probability ranking. HELOC: SCRE-Optimized probability ranking.
5. En iyi yorumlanabilir model hangisi?
   - Taiwan: Monotonic LightGBM conservative. HELOC: WOE scorecard unweighted.
6. SCRE'nin rolü ne?
   - SCRE-Optimized is retained as a reliability-aware research framework and as a strong review-prioritization ranker, not as a universal winner over all single models.
7. V2 final policy hala en iyi mi?
   - Partially. Taiwan operational screening changes marginally under validation-only cost; HELOC operational screening remains effectively V2 Scorecard/manual-review.
8. Yeni deneyler final kararı değiştirdi mi?
   - Partially. They add stronger Taiwan XGBoost manual-review evidence and stronger capacity@20 evidence for SCRE-Optimized, while keeping the HELOC Scorecard decision stable.
9. Final test için hangi policy kilitleniyor?
   - Taiwan: XGBoost xgboost_scale_pos_weight_5. HELOC: Best Scorecard. These are locked before any future held-out test evaluation.

## Methodological Lock

- Test set used for selection: NO.
- Test metrics read by this selector: NO.
- Held-out test evaluation may be run only after these locked policies are accepted.
