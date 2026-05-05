# Decision Curve Summary: HELOC

Decision curves are evaluated on the held-out test split. No model training or threshold optimization is performed here.

## Net Benefit Winner Ranges

| Threshold Range | Best Model | Better Than Treat-All? | Better Than Treat-None? | Max Net Benefit | Comment |
|---|---|---|---|---:|---|
| 0.01-0.02 | Tie: Best CatBoost; Best Scorecard; Best XGBoost; SCRE-Optimized; SCRE-Pareto | NO | YES | 0.5157 | Model is top-ranked but not superior to both defaults |
| 0.03-0.11 | Best CatBoost | YES | YES | 0.5057 | Useful decision-support range |
| 0.12 | SCRE-Pareto | YES | YES | 0.4573 | Useful decision-support range |
| 0.13 | Best CatBoost | YES | YES | 0.4507 | Useful decision-support range |
| 0.14-0.16 | Best Scorecard | YES | YES | 0.4451 | Useful decision-support range |
| 0.17 | SCRE-Pareto | YES | YES | 0.4284 | Useful decision-support range |
| 0.18 | Best Scorecard | YES | YES | 0.4229 | Useful decision-support range |
| 0.19-0.22 | Best CatBoost | YES | YES | 0.4174 | Useful decision-support range |
| 0.23-0.25 | SCRE-Pareto | YES | YES | 0.3975 | Useful decision-support range |
| 0.26 | Best XGBoost | YES | YES | 0.3807 | Useful decision-support range |
| 0.27-0.28 | SCRE-Pareto | YES | YES | 0.3754 | Useful decision-support range |
| 0.29-0.31 | Best XGBoost | YES | YES | 0.3669 | Useful decision-support range |
| 0.32 | SCRE-Pareto | YES | YES | 0.3506 | Useful decision-support range |
| 0.33-0.42 | Best XGBoost | YES | YES | 0.3453 | Useful decision-support range |
| 0.43 | SCRE-Pareto | YES | YES | 0.2986 | Useful decision-support range |
| 0.44-0.45 | SCRE-Optimized | YES | YES | 0.2923 | Useful decision-support range |
| 0.46 | SCRE-Pareto | YES | YES | 0.2790 | Useful decision-support range |
| 0.47-0.49 | Best CatBoost | YES | YES | 0.2750 | Useful decision-support range |
| 0.50 | Tie: Best CatBoost; SCRE-Optimized | YES | YES | 0.2547 | Useful decision-support range |
| 0.51 | SCRE-Optimized | YES | YES | 0.2492 | Useful decision-support range |
| 0.52-0.55 | Best CatBoost | YES | YES | 0.2423 | Useful decision-support range |
| 0.56-0.57 | SCRE-Pareto | YES | YES | 0.2241 | Useful decision-support range |
| 0.58 | SCRE-Optimized | YES | YES | 0.2131 | Useful decision-support range |
| 0.59-0.60 | Best Scorecard | YES | YES | 0.2054 | Useful decision-support range |
| 0.61-0.63 | SCRE-Pareto | YES | YES | 0.1919 | Useful decision-support range |
| 0.64-0.66 | Best XGBoost | YES | YES | 0.1741 | Useful decision-support range |
| 0.67 | SCRE-Optimized | YES | YES | 0.1548 | Useful decision-support range |
| 0.68-0.69 | SCRE-Pareto | YES | YES | 0.1500 | Useful decision-support range |
| 0.70-0.76 | Best Scorecard | YES | YES | 0.1352 | Useful decision-support range |
| 0.77-0.79 | SCRE-Pareto | YES | YES | 0.0926 | Useful decision-support range |
| 0.80 | Best XGBoost | YES | YES | 0.0709 | Useful decision-support range |

## Peak Net Benefit By Model

| Model | Best Threshold Probability | Max Net Benefit | Better Than Treat-All? | Better Than Treat-None? |
|---|---:|---:|---|---|
| Best CatBoost | 0.0100 | 0.5157 | NO | YES |
| Best Scorecard | 0.0100 | 0.5157 | NO | YES |
| Best XGBoost | 0.0100 | 0.5157 | NO | YES |
| SCRE-Optimized | 0.0100 | 0.5157 | NO | YES |
| SCRE-Pareto | 0.0100 | 0.5157 | NO | YES |

## Useful Ranges By Model

- Best CatBoost: beats both defaults over 0.03-0.80.
- Best XGBoost: beats both defaults over 0.04-0.07, 0.13-0.80.
- Best Scorecard: beats both defaults over 0.07-0.08, 0.11-0.80.
- SCRE-Optimized: beats both defaults over 0.13-0.80.
- SCRE-Pareto: beats both defaults over 0.08-0.13, 0.15-0.80.

## HELOC Scorecard Check

- Scorecard beats both default strategies over: 0.07-0.08, 0.11-0.80.