# Homework Freeze Audit

## Freeze Metadata

- Freeze date: 2026-04-29 16:27
- Project version: homework_v1_scre_credit_20260429
- Git commit hash: not available; Git command was not available in PATH.
- Backup path: `C:\Users\AKTS\Desktop\Resul\project_backups\SCRE_Credit_homework_freeze_20260429_1627`
- Delivery zip: `C:\Users\AKTS\Desktop\Resul\Data_Mining_Project\outputs\SCRE_Credit_homework_delivery_pack_20260429.zip`

## Main Datasets

- Primary dataset: UCI Default of Credit Card Clients / Taiwan.
- External validation dataset: FICO HELOC.

## Final Taiwan Result Summary

- Taiwan operational winner: CatBoost.
- CatBoost ROC-AUC: 0.7853
- CatBoost PR-AUC: 0.5639
- CatBoost recall: 0.8093
- CatBoost expected cost FN=5, FP=1: 3247
- Taiwan PR-AUC winner: SCRE-Optimized, PR-AUC 0.5669, expected cost 3312.

## Final HELOC Result Summary

- HELOC cost/interpretable winner: Scorecard.
- Scorecard ROC-AUC: 0.8015
- Scorecard PR-AUC: 0.8039
- Scorecard recall: 0.9815
- Scorecard expected cost FN=5, FP=1: 892
- HELOC SCRE-Optimized PR-AUC: 0.8130, expected cost 900.

## Final Model Decision

- Taiwan operational winner: CatBoost.
- Taiwan PR-AUC winner: SCRE-Optimized.
- HELOC cost/interpretable winner: Scorecard.
- Proposed framework: SCRE-Optimized.

## Safe Claim

Best single calibrated CatBoost provides the strongest Taiwan operational result under the explicit FN=5, FP=1 cost scenario, while SCRE-Credit provides a structured reliability-aware framework for comparing and integrating performance, calibration, cost, stability, and faithfulness evidence.

## Unsafe Claims

- Do not claim SCRE-Credit outperforms all individual models.
- Do not claim SCRE-Credit is the best model on Taiwan.
- Do not claim SCRE-Credit is the best model on HELOC.
- Do not claim SHAP/LIME evidence is causal.
- Do not claim HELOC proves universal external robustness.
- Do not claim this is a new fundamental machine-learning algorithm.

## Files Included In Delivery Pack

- `outputs\homework_delivery_pack\tables\final_decision_matrix.csv` from `outputs\tables\final_decision_matrix.csv`
- `outputs\homework_delivery_pack\tables\scre_revision_global_comparison.csv` from `outputs\tables\scre_revision_global_comparison.csv`
- `outputs\homework_delivery_pack\tables\final_winners_taiwan.csv` from `outputs\tables\final_winners_taiwan.csv`
- `outputs\homework_delivery_pack\tables\final_winners_heloc.csv` from `outputs\tables\final_winners_heloc.csv`
- `outputs\homework_delivery_pack\tables\statistical_tests_cleaned.csv` from `outputs\tables\statistical_tests_cleaned.csv`
- `outputs\homework_delivery_pack\tables\kendalls_w_stability_taiwan.csv` from `outputs\tables\kendalls_w_stability_taiwan.csv`
- `outputs\homework_delivery_pack\tables\kendalls_w_stability_heloc.csv` from `outputs\tables\kendalls_w_stability_heloc.csv`
- `outputs\homework_delivery_pack\tables\claim_control_matrix.csv` from `outputs\tables\claim_control_matrix.csv`
- `outputs\homework_delivery_pack\project_metadata\final_positioning_statement.md` from `outputs\final_positioning_statement.md`
- `outputs\homework_delivery_pack\project_metadata\final_revision_audit.md` from `outputs\final_revision_audit.md`
- `outputs\homework_delivery_pack\project_metadata\PROJECT_SCOPE.md` from `PROJECT_SCOPE.md`
- `outputs\homework_delivery_pack\project_metadata\README.md` from `README.md`
- `outputs\homework_delivery_pack\project_metadata\requirements.txt` from `requirements.txt`
- `outputs\homework_delivery_pack\project_metadata\environment.yml` from `environment.yml`
- `outputs\homework_delivery_pack\project_metadata\run_all.py` from `run_all.py`
- `outputs\homework_delivery_pack\figures\roc_curves.png` from `outputs\figures\roc_curves.png`
- `outputs\homework_delivery_pack\figures\precision_recall_curves.png` from `outputs\figures\precision_recall_curves.png`
- `outputs\homework_delivery_pack\figures\calibration_curve.png` from `outputs\figures\calibration_curve.png`
- `outputs\homework_delivery_pack\figures\calibration_curve_taiwan.png` from `outputs\figures\calibration_curve_taiwan.png`
- `outputs\homework_delivery_pack\figures\calibration_curve_heloc.png` from `outputs\figures\calibration_curve_heloc.png`
- `outputs\homework_delivery_pack\figures\cost_curve.png` from `outputs\figures\cost_curve.png`
- `outputs\homework_delivery_pack\figures\cost_curve_taiwan.png` from `outputs\figures\cost_curve_taiwan.png`
- `outputs\homework_delivery_pack\figures\cost_curve_heloc.png` from `outputs\figures\cost_curve_heloc.png`
- `outputs\homework_delivery_pack\figures\kendalls_w_comparison.png` from `outputs\figures\kendalls_w_comparison.png`
- `outputs\homework_delivery_pack\figures\shap_rank_stability_heatmap.png` from `outputs\figures\shap_rank_stability_heatmap.png`
- `outputs\homework_delivery_pack\figures\manual_review_band_distribution_taiwan.png` from `outputs\figures\manual_review_band_distribution_taiwan.png`
- `outputs\homework_delivery_pack\figures\manual_review_band_distribution_heloc.png` from `outputs\figures\manual_review_band_distribution_heloc.png`
- `outputs\homework_delivery_pack\figures\taiwan_vs_heloc_metrics.png` from `outputs\figures\taiwan_vs_heloc_metrics.png`
- `outputs\homework_delivery_pack\figures\taiwan_vs_heloc_calibration.png` from `outputs\figures\taiwan_vs_heloc_calibration.png`
- `outputs\homework_delivery_pack\figures\ablation_results_barplot.png` from `outputs\figures\ablation_results_barplot.png`
- `outputs\homework_delivery_pack\figures\ablation_cost_vs_auc.png` from `outputs\figures\ablation_cost_vs_auc.png`
- `outputs\homework_delivery_pack\figures\shap\global_bar.png` from `outputs\shap\global_bar.png`
- `outputs\homework_delivery_pack\figures\shap\global_beeswarm.png` from `outputs\shap\global_beeswarm.png`
- `outputs\homework_delivery_pack\figures\shap\external_validation_heloc\global_bar.png` from `outputs\shap\external_validation_heloc\global_bar.png`
- `outputs\homework_delivery_pack\figures\shap\external_validation_heloc\global_beeswarm.png` from `outputs\shap\external_validation_heloc\global_beeswarm.png`
- `outputs\homework_delivery_pack\delivery_pack_file_inventory.csv` from `outputs\homework_delivery_pack\delivery_pack_file_inventory.csv`
- `outputs\homework_delivery_pack\HOMEWORK_FREEZE_AUDIT.md` from `outputs\homework_delivery_pack\HOMEWORK_FREEZE_AUDIT.md`
- `outputs\homework_delivery_pack\FREEZE_SUMMARY.txt` from `outputs\homework_delivery_pack\FREEZE_SUMMARY.txt`

## Missing Files

- None

## Can This Version Be Used For Homework Report?

YES
