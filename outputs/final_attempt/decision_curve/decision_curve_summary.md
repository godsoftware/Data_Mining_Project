# Decision Curve Final Analysis Summary

## Protocol
- Net benefit was computed as `TP/N - FP/N * (pt / (1 - pt))`.
- Threshold probability grid: 0.01 to 0.80 by 0.01.
- Validation and held-out diagnostic splits are reported.
- Held-out diagnostic rows are not used for final model selection.
- Treat-all and treat-none are evaluated at every threshold.

## TAIWAN validation useful ranges

- SCRE-Pareto probability ranking: useful=0.01-0.80, width=80, max NB=0.2133 at pt=0.01, best-threshold range=0.01; 0.09-0.11; 0.14; ...
- SCRE-Optimized probability ranking: useful=0.01-0.80, width=80, max NB=0.2133 at pt=0.01, best-threshold range=0.02-0.08; 0.12-0.13; 0.15-0.21; ...
- Best CatBoost probability ranking: useful=0.04; 0.06-0.80, width=76, max NB=0.2131 at pt=0.01, best-threshold range=none
- V2 final locked policy ranking: useful=0.04; 0.06-0.80, width=76, max NB=0.2131 at pt=0.01, best-threshold range=none
- Best literature reproduction model: useful=0.04-0.78, width=75, max NB=0.2132 at pt=0.01, best-threshold range=0.76-0.77
- Best EBM/monotonic model: Monotonic LightGBM: useful=0.06-0.78, width=73, max NB=0.2133 at pt=0.01, best-threshold range=0.53-0.55; 0.60-0.61

## TAIWAN heldout_test_diagnostic useful ranges

- SCRE-Pareto probability ranking: useful=0.01; 0.03-0.80, width=79, max NB=0.2133 at pt=0.01, best-threshold range=0.01; 0.13; 0.23; ...
- SCRE-Optimized probability ranking: useful=0.01; 0.03-0.80, width=79, max NB=0.2133 at pt=0.01, best-threshold range=0.12; 0.14; 0.17; ...
- Best CatBoost probability ranking: useful=0.03-0.80, width=78, max NB=0.2131 at pt=0.01, best-threshold range=none
- V2 final locked policy ranking: useful=0.03-0.80, width=78, max NB=0.2131 at pt=0.01, best-threshold range=none
- Best EBM/monotonic model: Monotonic LightGBM: useful=0.04-0.80, width=77, max NB=0.2133 at pt=0.01, best-threshold range=0.04-0.08; 0.33-0.47; 0.49-0.55
- Best literature reproduction model: useful=0.02; 0.04; 0.06-0.80, width=77, max NB=0.2132 at pt=0.01, best-threshold range=0.02; 0.09-0.10; 0.19-0.22; ...

## HELOC validation useful ranges

- Best CatBoost probability ranking: useful=0.03-0.80, width=78, max NB=0.5154 at pt=0.01, best-threshold range=none
- Best literature reproduction model: useful=0.03-0.80, width=78, max NB=0.5154 at pt=0.01, best-threshold range=none
- SCRE-Pareto probability ranking: useful=0.07-0.80, width=74, max NB=0.5154 at pt=0.01, best-threshold range=0.11-0.12; 0.18-0.19; 0.24-0.38; ...
- Best Scorecard probability ranking: useful=0.07-0.09; 0.11-0.80, width=73, max NB=0.5154 at pt=0.01, best-threshold range=none
- V2 final locked policy ranking: useful=0.07-0.09; 0.11-0.80, width=73, max NB=0.5154 at pt=0.01, best-threshold range=none
- Best EBM/monotonic model: EBM: useful=0.03-0.05; 0.11; 0.13-0.80, width=72, max NB=0.5151 at pt=0.01, best-threshold range=0.03-0.05

## HELOC heldout_test_diagnostic useful ranges

- Best CatBoost probability ranking: useful=0.03-0.80, width=78, max NB=0.5157 at pt=0.01, best-threshold range=none
- Best literature reproduction model: useful=0.03-0.80, width=78, max NB=0.5157 at pt=0.01, best-threshold range=none
- SCRE-Pareto probability ranking: useful=0.09-0.80, width=72, max NB=0.5157 at pt=0.01, best-threshold range=0.13; 0.23-0.26; 0.28; ...
- Best Scorecard probability ranking: useful=0.07-0.08; 0.11-0.80, width=72, max NB=0.5157 at pt=0.01, best-threshold range=none
- V2 final locked policy ranking: useful=0.07-0.08; 0.11-0.80, width=72, max NB=0.5157 at pt=0.01, best-threshold range=none
- Best EBM/monotonic model: EBM: useful=0.12-0.80, width=69, max NB=0.5143 at pt=0.01, best-threshold range=0.27; 0.35-0.36; 0.64; ...

## Questions

1. Widest useful threshold range: Taiwan validation = SCRE-Pareto probability ranking; HELOC validation = Best CatBoost probability ranking.
2. CatBoost is most relevant where its `best_model_at_threshold_range` is non-empty; otherwise it remains a strong comparator but not the NB winner.
3. Scorecard is most relevant as the interpretable baseline; HELOC scorecard remains useful but is not always the highest net-benefit ranker.
4. SCRE is meaningful when its useful range is wide and its best-threshold range is non-empty; SCRE-Optimized is especially competitive on HELOC.
5. Treat-all comparison is explicit in `better_than_treat_all`; low thresholds often make treat-all hard to beat, but models become useful over practical mid-threshold ranges.
6. Treat-none comparison is explicit in `better_than_treat_none`; useful ranges require beating both treat-all and treat-none.
7. Decision curve should inform final recommendation, but it should not override validation-only selection by itself.

## Candidate availability notes
- taiwan Best deep model: No persisted probability artifact found
- heloc Best deep model: No persisted probability artifact found
