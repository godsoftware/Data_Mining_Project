# Decision Curve Summary: TAIWAN

Decision curves are evaluated on the held-out test split. No model training or threshold optimization is performed here.

## Net Benefit Winner Ranges

| Threshold Range | Best Model | Better Than Treat-All? | Better Than Treat-None? | Max Net Benefit | Comment |
|---|---|---|---|---:|---|
| 0.01 | SCRE-Optimized | YES | YES | 0.2133 | Useful decision-support range |
| 0.02 | Best XGBoost | YES | YES | 0.2053 | Useful decision-support range |
| 0.03 | Best CatBoost | YES | YES | 0.1974 | Useful decision-support range |
| 0.04 | SCRE-Optimized | YES | YES | 0.1891 | Useful decision-support range |
| 0.05 | Best CatBoost | YES | YES | 0.1813 | Useful decision-support range |
| 0.06 | Best Scorecard | YES | YES | 0.1734 | Useful decision-support range |
| 0.07 | Best CatBoost | YES | YES | 0.1658 | Useful decision-support range |
| 0.08-0.10 | Best XGBoost | YES | YES | 0.1588 | Useful decision-support range |
| 0.11 | Best CatBoost | YES | YES | 0.1420 | Useful decision-support range |
| 0.12 | SCRE-Pareto | YES | YES | 0.1371 | Useful decision-support range |
| 0.13-0.14 | SCRE-Optimized | YES | YES | 0.1317 | Useful decision-support range |
| 0.15-0.18 | Best CatBoost | YES | YES | 0.1207 | Useful decision-support range |
| 0.19-0.22 | Best XGBoost | YES | YES | 0.1025 | Useful decision-support range |
| 0.23 | SCRE-Pareto | YES | YES | 0.0938 | Useful decision-support range |
| 0.24-0.26 | Best CatBoost | YES | YES | 0.0923 | Useful decision-support range |
| 0.27 | SCRE-Pareto | YES | YES | 0.0845 | Useful decision-support range |
| 0.28 | Best CatBoost | YES | YES | 0.0831 | Useful decision-support range |
| 0.29-0.30 | SCRE-Optimized | YES | YES | 0.0808 | Useful decision-support range |
| 0.31 | Best CatBoost | YES | YES | 0.0763 | Useful decision-support range |
| 0.32-0.33 | SCRE-Optimized | YES | YES | 0.0725 | Useful decision-support range |
| 0.34-0.35 | Best CatBoost | YES | YES | 0.0690 | Useful decision-support range |
| 0.36 | SCRE-Pareto | YES | YES | 0.0644 | Useful decision-support range |
| 0.37-0.40 | SCRE-Optimized | YES | YES | 0.0618 | Useful decision-support range |
| 0.41-0.45 | Best CatBoost | YES | YES | 0.0551 | Useful decision-support range |
| 0.46 | SCRE-Pareto | YES | YES | 0.0467 | Useful decision-support range |
| 0.47-0.48 | Best CatBoost | YES | YES | 0.0453 | Useful decision-support range |
| 0.49 | Best Scorecard | YES | YES | 0.0420 | Useful decision-support range |
| 0.50 | SCRE-Pareto | YES | YES | 0.0410 | Useful decision-support range |
| 0.51 | Best CatBoost | YES | YES | 0.0394 | Useful decision-support range |
| 0.52-0.53 | SCRE-Optimized | YES | YES | 0.0378 | Useful decision-support range |
| 0.54 | Best CatBoost | YES | YES | 0.0346 | Useful decision-support range |
| 0.55 | SCRE-Optimized | YES | YES | 0.0336 | Useful decision-support range |
| 0.56-0.59 | SCRE-Pareto | YES | YES | 0.0317 | Useful decision-support range |
| 0.60-0.63 | SCRE-Optimized | YES | YES | 0.0269 | Useful decision-support range |
| 0.64-0.65 | SCRE-Pareto | YES | YES | 0.0203 | Useful decision-support range |
| 0.66-0.67 | Best CatBoost | YES | YES | 0.0189 | Useful decision-support range |
| 0.68-0.69 | SCRE-Pareto | YES | YES | 0.0165 | Useful decision-support range |
| 0.70-0.74 | SCRE-Optimized | YES | YES | 0.0143 | Useful decision-support range |
| 0.75 | Best XGBoost | YES | YES | 0.0098 | Useful decision-support range |
| 0.76 | SCRE-Optimized | YES | YES | 0.0088 | Useful decision-support range |
| 0.77 | Best XGBoost | YES | YES | 0.0080 | Useful decision-support range |
| 0.78 | Best CatBoost | YES | YES | 0.0076 | Useful decision-support range |
| 0.79-0.80 | Best XGBoost | YES | YES | 0.0063 | Useful decision-support range |

## Peak Net Benefit By Model

| Model | Best Threshold Probability | Max Net Benefit | Better Than Treat-All? | Better Than Treat-None? |
|---|---:|---:|---|---|
| Best CatBoost | 0.0100 | 0.2131 | NO | YES |
| Best Scorecard | 0.0100 | 0.2133 | NO | YES |
| Best XGBoost | 0.0100 | 0.2132 | NO | YES |
| SCRE-Optimized | 0.0100 | 0.2133 | YES | YES |
| SCRE-Pareto | 0.0100 | 0.2133 | NO | YES |

## Useful Ranges By Model

- Best CatBoost: beats both defaults over 0.03-0.80.
- Best XGBoost: beats both defaults over 0.02, 0.04, 0.06-0.80.
- Best Scorecard: beats both defaults over 0.03-0.80.
- SCRE-Optimized: beats both defaults over 0.01, 0.03-0.04, 0.06-0.80.
- SCRE-Pareto: beats both defaults over 0.04-0.80.

## Taiwan CatBoost 0.15 Check

- Net benefit at pt=0.15: 0.1207.
- Better than treat-all: YES.
- Better than treat-none: YES.
- Rank at pt=0.15: 1.

## SCRE-Optimized vs CatBoost

- SCRE-Optimized has higher net benefit than Best CatBoost over: 0.01, 0.04, 0.06, 0.12-0.14, 0.23, 0.29-0.30, 0.32-0.33, 0.37-0.40, 0.46, 0.49-0.50, 0.52-0.53, 0.55-0.56, 0.58-0.63, 0.69-0.74, 0.76, 0.80.