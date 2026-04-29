# Final Revision Audit

## Required Artifact Check

| Artifact | Exists | Non-empty | Shape |
|---|---:|---:|---|
| `outputs\tables\result_inventory_audit.csv` | YES | YES | 252 rows x 23 cols |
| `outputs\tables\final_leaderboard_taiwan.csv` | YES | YES | 10 rows x 37 cols |
| `outputs\tables\final_leaderboard_heloc.csv` | YES | YES | 8 rows x 37 cols |
| `outputs\tables\pareto_front_taiwan.csv` | YES | YES | 11 rows x 30 cols |
| `outputs\tables\pareto_front_heloc.csv` | YES | YES | 11 rows x 30 cols |
| `outputs\tables\scre_pareto_results_taiwan.csv` | YES | YES | 2 rows x 30 cols |
| `outputs\tables\scre_optimized_results_taiwan.csv` | YES | YES | 5 rows x 34 cols |
| `outputs\tables\scre_revision_global_comparison.csv` | YES | YES | 16 rows x 23 cols |
| `outputs\tables\kendalls_w_stability_taiwan.csv` | YES | YES | 20 rows x 19 cols |
| `outputs\tables\statistical_tests_cleaned.csv` | YES | YES | 84 rows x 25 cols |
| `outputs\tables\manual_review_band_revision_taiwan.csv` | YES | YES | 10 rows x 40 cols |
| `outputs\tables\final_decision_matrix.csv` | YES | YES | 11 rows x 9 cols |
| `outputs\tables\claim_control_matrix.csv` | YES | YES | 12 rows x 7 cols |
| `outputs\final_positioning_statement.md` | YES | YES |  |

## Final Decisions

| Check | Decision | Evidence |
|---|---:|---|
| SCRE revised successfully | YES | SCRE-Pareto and SCRE-Optimized outputs exist and are included in global comparison. |
| SCRE beats CatBoost on Taiwan operational cost | NO | CatBoost cost=3247; SCRE-Pareto cost=3307; SCRE-Optimized cost=3312. |
| SCRE beats CatBoost on Taiwan PR-AUC | YES | CatBoost PR-AUC=0.5639; SCRE-Pareto PR-AUC=0.5661; SCRE-Optimized PR-AUC=0.5669. |
| SCRE beats Scorecard on HELOC cost | NO | Scorecard cost=892; Old SCRE cost=895; SCRE-Pareto cost=904; SCRE-Optimized cost=900. |
| Kendall's W added | YES | Kendall's W top-5 rows are present in the Taiwan stability table. |
| Statistical tests cleaned | YES | Clean statistical test table has 84 rows and required winner/direction columns. |
| Manual review band revised | YES | Manual-review bands use validation-selected thresholds and include remaining-cost fields. |
| Final model recommendation clear | YES | Final decision matrix includes an objective-specific deployment recommendation. |
| Safe claims ready | YES | Claim-control matrix marks the three unsupported SCRE dominance claims as DO NOT CLAIM. |
| Proceed to final report | YES | Proceed only with scoped, non-overclaiming language. |

## Bottom Line

SCRE-Credit was revised successfully as SCRE-Pareto and SCRE-Optimized, and the revised framework is now positioned more honestly. It should not be claimed as the best operational model on every dataset.

The final operational recommendation is CatBoost for Taiwan under the FN=5, FP=1 expected-cost objective, and Scorecard for HELOC expected cost. SCRE-Optimized should be presented as the proposed reliability-aware framework and PR-AUC-oriented SCRE variant, not as a universal cost winner.

Proceed to final report: YES
