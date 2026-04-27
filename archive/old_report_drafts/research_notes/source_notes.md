# Source Notes

These sources justify the restructured project scope.

## Primary Dataset

UCI Default of Credit Card Clients:

- Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Reason used: confirms 30,000 instances, 23 features, binary default target, and the credit-card default probability framing.

Original Taiwan default paper:

- Source: https://doi.org/10.1016/j.eswa.2007.12.020
- Reason used: frames the task as probability-of-default prediction, not only hard classification.

## External Robustness Dataset

FICO HELOC via AIX360 documentation:

- Source: https://aix360.readthedocs.io/en/latest/datasets.html
- Reason used: describes HELOC as anonymized home-equity line-of-credit applications and defines `RiskPerformance`.

OpenML FICO-HELOC-cleaned:

- Source: https://www.openml.org/api/v1/json/data/45554
- Reason used: practical downloadable mirror used for `data/external/heloc/`.

Interpretable AI HELOC example:

- Source: https://docs.interpretable.ai/stable/examples/fico/
- Reason used: explains the HELOC task as predicting whether applicants repay within the risk-performance horizon and motivates explainability in lending.

## Methodology Guardrails

Imbalanced-learn common pitfalls:

- Source: https://imbalanced-learn.org/dev/common_pitfalls.html
- Reason used: supports keeping resampling inside train folds only.

Scikit-learn calibration:

- Source: https://scikit-learn.org/stable/modules/calibration.html
- Reason used: supports calibration curves, Brier score, sigmoid calibration, and isotonic calibration.

