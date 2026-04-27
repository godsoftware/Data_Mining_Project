# SCRE-Credit Experiment Plan

## Phase 0 - Project Cleaning And Scope Lock

Status:

- German Credit removed from active scope and kept only under archive.
- Old report/presentation generation material archived.
- README kept as a technical project README.
- Final report generation paused.
- Project claim fixed as SCRE-Credit.
- Main dataset fixed as Taiwan.
- External validation dataset fixed as HELOC.
- Conformal prediction excluded.

Expected outputs:

- outputs/project_scope.md
- outputs/experiment_plan.md
- archive/old_german_credit_option/
- archive/old_report_drafts/

## Phase 1 - Reproducibility And Environment

Goal:

Make the project runnable from a fixed environment with traceable experiment outputs.

Required controls:

- requirements.txt includes all required Python packages.
- environment.yml pins Python 3.10 for the tf210win workflow.
- Random seed is centralized in src/config/experiment_config.yaml.
- run_all.py runs the reproducible core workflow.
- Experiment scripts write records to outputs/experiment_logs/runs.jsonl.
- New SCRE-Credit model/table artifacts receive timestamp and experiment_id copies.

Core command:

```powershell
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' run_all.py
```

Smoke-test command:

```powershell
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' src\run_scre_credit_experiments.py --dataset both --quick --n-bootstraps 5
```

Verification commands:

```powershell
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' experiments\00_environment_check.py
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' -m pytest tests
```

## Phase 2 - Taiwan Main Modelling

Status:

- Taiwan data pipeline completed.
- Cleaned interim dataset generated.
- Taiwan data audit table generated.
- Taiwan target distribution figure generated.
- ID is retained in the interim audit dataset and excluded from the model-ready dataset.

Run:

```powershell
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' experiments\01_prepare_taiwan.py
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' experiments\03_train_base_models_taiwan.py
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' experiments\04_train_scre_credit_taiwan.py
```

Expected outputs:

- cleaned and model-ready Taiwan data,
- baseline/enhanced model comparison tables,
- SCRE-Credit candidate pool,
- SCRE-Credit reliability weights,
- SCRE-Credit validation metrics,
- threshold and cost tables,
- bootstrap confidence intervals,
- manual-review decision-band outputs,
- trained SCRE-Credit model artifact.

## Phase 3 - HELOC External Robustness

Status:

- HELOC raw CSV is stored under data/raw/heloc/.
- Target is converted to bad_flag with Good = 0 and Bad = 1.
- Special HELOC missing codes -9, -8, and -7 are detected on raw data and converted to NaN when present.
- Cleaned HELOC data preserves missing values for train-fold imputation in model pipelines.
- HELOC data audit, data dictionary, and target distribution figure are generated.
- HELOC is compatible with the same binary metric interface used by Taiwan.

Run:

```powershell
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' experiments\02_prepare_heloc.py
& 'C:\Users\AKTS\anaconda3\envs\tf210win\python.exe' experiments\05_external_validation_heloc.py
```

Purpose:

HELOC does not validate a Taiwan-trained model directly. It validates whether the same reliability-weighted framework can be reproduced on a different credit-risk dataset.

Expected outputs:

- cleaned and model-ready HELOC data,
- reduced HELOC candidate pool,
- HELOC SCRE-Credit reliability weights,
- HELOC threshold/cost/review-band outputs,
- external validation summary table.

## Phase 4 - Reliability Extensions

Feature-engineering status:

- Taiwan row-wise payment-behavior features completed.
- HELOC conceptually aligned row-wise features completed.
- Feature dictionaries generated for both datasets.
- No target encoding, global mean encoding, or dataset-level statistics are used in pre-split feature generation.
- Missing-value imputation remains inside model pipelines after splitting.

## Phase 5 - Split And Leakage Control

Status:

- Stratified train/validation/test split summaries generated for Taiwan and HELOC.
- Split policy is centralized as 60% train, 20% validation, and 20% test.
- The 80/20 train/test + CV-based calibration strategy remains documented as an allowed alternative.
- Resampling, scaling, encoding, and imputation are implemented inside train-fitted pipelines.
- SCRE-Credit threshold selection now uses internal validation probabilities; test predictions are reserved for final metrics.
- Leakage checklist generated and all current checks pass.

Expected outputs:

- outputs/tables/split_summary_taiwan.csv
- outputs/tables/split_summary_heloc.csv
- outputs/tables/leakage_checklist.csv

## Phase 6 - Baseline Models

Status:

- Majority baseline, Logistic Regression, class-weighted Logistic Regression, WOE / scorecard Logistic Regression, Decision Tree, and Random Forest baselines are implemented for Taiwan and HELOC.
- Baselines are fit on the train split and reported on the validation split.
- Test split remains reserved and is not used in baseline model selection.
- Preprocessing is fit inside train-fitted pipelines.
- Baseline model artifacts are saved under outputs/models/baselines/.

Expected outputs:

- outputs/tables/baseline_results_taiwan.csv
- outputs/tables/baseline_results_heloc.csv
- outputs/models/baselines/

## Phase 7 - Black-Box And Constrained Models

Status:

- XGBoost, LightGBM, CatBoost, SMOTENC + XGBoost, class-weighted LightGBM, monotonic XGBoost, and monotonic LightGBM are implemented for Taiwan and HELOC.
- Models are fit on the train split and reported on the validation split.
- Test split remains reserved and is not used for model selection.
- Monotonic priors are explicit in src/monotonic_models.py.
- LIMIT_BAL is constrained as neutral (0) to avoid an over-strong assumption.

Expected outputs:

- outputs/tables/model_results_taiwan.csv
- outputs/tables/model_results_heloc.csv
- outputs/models/boosting/
- outputs/models/monotonic/

## Phase 8 - Hyperparameter Optimization

Status:

- Optuna optimization is implemented with PR-AUC as the primary objective.
- ROC-AUC, recall, Brier score, and FN-weighted cost are stored as secondary trial metrics.
- RepeatedStratifiedKFold is used with n_splits = 5 and n_repeats = 3.
- Taiwan default search uses 50 trials; HELOC default search uses 30 trials.
- The current implementation searches the black-box family as a conditional model space: XGBoost, LightGBM, and CatBoost.
- Tuning uses the train split with repeated CV; validation split is used only to evaluate the selected tuned model; test split remains reserved.

Expected outputs:

- outputs/tables/hyperparameter_search_results.csv
- outputs/tables/best_params.csv
- outputs/figures/optuna_optimization_history.png
- outputs/figures/optuna_param_importance.png

## Phase 9 - Calibration

Status:

- Uncalibrated, sigmoid-calibrated, and isotonic-calibrated probabilities are implemented for the Phase 6 baseline pool, Phase 7 model pool, and the Phase 8 tuned best model when available.
- Calibration is fit only on the train split with CalibratedClassifierCV(cv=3) for sigmoid and isotonic variants.
- Validation split is used to report Brier score, ECE, calibration slope, calibration intercept, PR-AUC, ROC-AUC, and cost.
- Test split remains reserved and is not used for calibration fitting or calibration metric reporting.

Expected outputs:

- outputs/tables/calibration_results_taiwan.csv
- outputs/tables/calibration_results_heloc.csv
- outputs/figures/calibration_curve_taiwan.png
- outputs/figures/calibration_curve_heloc.png

## Phase 10 - Cost-Sensitive Threshold And Manual Review Band

Status:

- Taiwan cost-sensitive threshold search is implemented on validation probabilities with thresholds from 0.01 to 0.99 in 0.01 steps.
- Scenario A uses FN=2, FP=1; Scenario B uses FN=5, FP=1; Scenario C uses FN=10, FP=1.
- Threshold analysis covers the Phase 6 baseline pool, Phase 7 model pool, and the Phase 8 tuned best model across uncalibrated, sigmoid, and isotonic probability variants.
- Manual review band search is implemented for the primary Taiwan decision candidate selected by highest validation PR-AUC with Brier-score tie-break.
- Manual review policy assumes auto-approved low-risk false negatives, auto-flagged high-risk false positives, and a central manual-review cost from config.
- Test split remains reserved and is not used for threshold or manual-review-band selection.

Expected outputs:

- outputs/tables/threshold_analysis_taiwan.csv
- outputs/tables/manual_review_band_results_taiwan.csv
- outputs/figures/cost_curve_taiwan.png
- outputs/figures/threshold_tradeoff_taiwan.png

Required before strong research claims:

- 30/50/100-seed SHAP stability runs.
- Fold-level SHAP stability.
- Top-k overlap, Spearman rank correlation, Kendall's W.
- SHAP background sensitivity.
- Expanded deletion/insertion/perturbation faithfulness tests.
- Ablation study:
  performance-only ensemble vs performance+calibration vs performance+calibration+cost vs full SCRE-Credit.
- Subgroup reliability:
  calibration, recall, false-negative rate, and cost by demographic/payment-behavior subgroups.
- Statistical model comparison with bootstrap confidence intervals and paired tests where appropriate.

## Phase 5 - Final Reporting

Not active now.

Final report writing starts only after the technical pipeline, reliability extensions, and external validation checks are complete.
