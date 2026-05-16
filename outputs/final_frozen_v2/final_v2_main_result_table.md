# Final V2 Main Result Table

These are the only final operational policies to use as main report evidence.

| dataset | final_policy | precision | recall | specificity | fp | fn | cost | manual_review_rate | safe_category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | V2 CatBoost manual-review | 0.529 | 0.575 | 0.854 | 680 | 564 | 2705.5 | 0.295 | FINAL MAIN EVIDENCE |
| heloc | V2 Scorecard manual-review | 0.683 | 0.846 | 0.574 | 403 | 158 | 764.5 | 0.27 | FINAL MAIN EVIDENCE |

Interpretation:

- Taiwan final operational evidence: V2 CatBoost manual-review.
- HELOC final external-validation evidence: V2 Scorecard manual-review.
- Both are audit-ready and selected under the validation-only / locked-test protocol.
- These policies support screening and manual-review prioritization, not automatic rejection.
