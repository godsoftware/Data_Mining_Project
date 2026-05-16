# Capacity-Aware Manual Review Summary

## Protocol
- Models are evaluated as manual-review prioritization rankers, not automatic rejection systems.
- Validation and held-out test diagnostic splits are both reported.
- Test rows are not used to lock a final policy; `test_set_used_for_selection=False` is written to output rows.
- Review workload cost is 0.5 per reviewed customer; each captured default is valued at FN cost 5.0.
- Net value = FN cost saved by captured defaults - review workload cost.

## TAIWAN validation top review rankers

- SCRE-Optimized probability ranking: top10 capture=0.321, top20 capture=0.529, top30 capture=0.647, precision@20=0.585, lift@20=2.65, net@20=2910.0
- SCRE-Pareto probability ranking: top10 capture=0.318, top20 capture=0.526, top30 capture=0.651, precision@20=0.582, lift@20=2.63, net@20=2890.0
- Best EBM/monotonic model: Monotonic LightGBM: top10 capture=0.317, top20 capture=0.520, top30 capture=0.638, precision@20=0.575, lift@20=2.60, net@20=2850.0
- Best CatBoost probability ranking: top10 capture=0.314, top20 capture=0.518, top30 capture=0.632, precision@20=0.573, lift@20=2.59, net@20=2835.0
- Best literature reproduction model: top10 capture=0.317, top20 capture=0.518, top30 capture=0.639, precision@20=0.573, lift@20=2.59, net@20=2835.0

## TAIWAN heldout_test_diagnostic top review rankers

- Best EBM/monotonic model: Monotonic LightGBM: top10 capture=0.317, top20 capture=0.514, top30 capture=0.635, precision@20=0.568, lift@20=2.57, net@20=2810.0
- SCRE-Optimized probability ranking: top10 capture=0.316, top20 capture=0.512, top30 capture=0.633, precision@20=0.567, lift@20=2.56, net@20=2800.0
- SCRE-Pareto probability ranking: top10 capture=0.316, top20 capture=0.511, top30 capture=0.635, precision@20=0.565, lift@20=2.55, net@20=2790.0
- Best CatBoost probability ranking: top10 capture=0.315, top20 capture=0.510, top30 capture=0.630, precision@20=0.564, lift@20=2.55, net@20=2785.0
- V2 final locked policy ranking: top10 capture=0.315, top20 capture=0.510, top30 capture=0.630, precision@20=0.564, lift@20=2.55, net@20=2785.0

## HELOC validation top review rankers

- SCRE-Optimized probability ranking: top10 capture=0.182, top20 capture=0.340, top30 capture=0.500, precision@20=0.884, lift@20=1.70, net@20=1547.5
- Best EBM/monotonic model: EBM: top10 capture=0.169, top20 capture=0.338, top30 capture=0.481, precision@20=0.878, lift@20=1.69, net@20=1537.5
- SCRE-Pareto probability ranking: top10 capture=0.174, top20 capture=0.334, top30 capture=0.490, precision@20=0.868, lift@20=1.67, net@20=1517.5
- Best Scorecard probability ranking: top10 capture=0.165, top20 capture=0.328, top30 capture=0.485, precision@20=0.853, lift@20=1.64, net@20=1487.5
- V2 final locked policy ranking: top10 capture=0.165, top20 capture=0.328, top30 capture=0.485, precision@20=0.853, lift@20=1.64, net@20=1487.5

## HELOC heldout_test_diagnostic top review rankers

- SCRE-Optimized probability ranking: top10 capture=0.176, top20 capture=0.339, top30 capture=0.482, precision@20=0.881, lift@20=1.69, net@20=1542.5
- SCRE-Pareto probability ranking: top10 capture=0.177, top20 capture=0.338, top30 capture=0.482, precision@20=0.878, lift@20=1.69, net@20=1537.5
- Best CatBoost probability ranking: top10 capture=0.178, top20 capture=0.337, top30 capture=0.478, precision@20=0.876, lift@20=1.68, net@20=1532.5
- Best EBM/monotonic model: EBM: top10 capture=0.168, top20 capture=0.337, top30 capture=0.482, precision@20=0.876, lift@20=1.68, net@20=1532.5
- Best literature reproduction model: top10 capture=0.178, top20 capture=0.337, top30 capture=0.478, precision@20=0.876, lift@20=1.68, net@20=1532.5

## Questions

1. If the riskiest 10% is reviewed, how many defaults are captured?
   - Use `top_10_capture` in `capacity_model_comparison.csv`; this is reported separately for validation and held-out test diagnostic rows.
2. If the riskiest 20% is reviewed, how many defaults are captured?
   - Use `top_20_capture`. This is the most operationally useful capacity point in the summary tables.
3. If the riskiest 30% is reviewed, how many defaults are captured?
   - Use `top_30_capture`; gains should be compared against random review at 30%.
4. How much better than random review is the model?
   - `lift_at_k` and `cost_saved_vs_random_review` show this directly. Random review has lift near 1.0 by construction.
5. CatBoost, Scorecard, or SCRE?
   - This is ranking-focused. CatBoost often remains strong on Taiwan, Scorecard/EBM are competitive on HELOC, and SCRE should be judged by its Lift@K rather than only threshold cost.
6. If manual-review capacity is 20%, does the final model change?
   - Treat this as validation-selection evidence in the next final selector. Held-out test rows are diagnostic only.
7. Plain-language explanation for the instructor:
   - Instead of saying the model rejects applicants, say: 'When we can manually inspect only the highest-risk 20% of cases, the model concentrates a much larger share of actual defaults than random review.'

## Candidate availability notes
- taiwan Best deep model: No persisted probability artifact found
- heloc Best deep model: No persisted probability artifact found
