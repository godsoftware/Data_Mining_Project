# SCRE-Credit Research Pipeline

This repository is an experimental machine-learning research pipeline, not a
final report or presentation draft. The active scope is fixed to:

- Primary dataset: UCI Default of Credit Card Clients / Taiwan
- External validation dataset: FICO HELOC
- Proposed method: SCRE-Credit, a Stability- and Cost-Regularized Ensemble for
  Credit Risk

German Credit / Option A and older report drafts are archived and are not part
of the active experimental scope. See `PROJECT_SCOPE.md` for the full scope,
out-of-scope claims, datasets, and research questions.

## Reproducible Workflow

Create the environment from `requirements.txt`, place the raw Taiwan and HELOC
files under `data/raw/`, then run the audited Prompt 0-12 pipeline:

```powershell
C:\Users\AKTS\anaconda3\envs\tf210win\python.exe run_all.py
```

Full Optuna tuning is intentionally opt-in because it is CPU-expensive:

```powershell
C:\Users\AKTS\anaconda3\envs\tf210win\python.exe experiments\08_optuna_tuning.py --dataset both --taiwan-trials 100 --heloc-trials 50 --models xgboost lightgbm catboost
```

`run_all.py --include-optuna` runs the same tuning step inside the full workflow.

## Main Outputs

- Cleaned data: `data/interim/`
- Model-ready data: `data/processed/`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/`
- Models: `outputs/models/`
- Experiment logs: `outputs/experiment_logs/`

The test set is reserved for final evaluation. Model selection, calibration,
and threshold selection are performed on train/validation or cross-validation
splits only.
