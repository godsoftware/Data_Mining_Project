# Reproducibility Checklist

## Environment

- Use Anaconda environment: `tf210win`.
- Python path checked: `C:\Users\AKTS\anaconda3\envs\tf210win\python.exe`.
- Record package versions before final experiments.

## Data

- Primary raw data remains under `data/raw/`.
- HELOC external data remains under `data/external/heloc/`.
- Processed data is generated, not manually edited.
- Taiwan and HELOC must not be merged.

## Splits

- Use stratified train/test splits.
- Fit preprocessing only on train folds.
- Keep SMOTE/SMOTENC inside imblearn pipelines.
- Select thresholds on validation or OOF predictions, then report test performance.

## Randomness

- Use fixed base seed `42`.
- For stability experiments, run planned seed grids: 30, 50, then 100 if runtime allows.

## Reporting

- Do not claim publishable breakthrough.
- Do not claim direct external validation across incompatible feature schemas.
- State that HELOC is a methodology robustness dataset.
- Report bootstrap confidence intervals where possible.
- Report remaining limitations explicitly.

