# False Positive Profile Interpretation

## 1. Are Taiwan FPs closer to TP or TN?
For the old aggressive CatBoost threshold, Taiwan FPs are classified as **TP-like risky profile (distance FP-TP=0.329, FP-TN=0.457)**. For the analyzed manual-review high-risk bucket, they are **TP-like risky profile (distance FP-TP=0.252, FP-TN=0.697)**. This means the FP population is not simply harmless noise; a material part of it has default-like risk signals, especially in delay and debt/payment behavior.

## 2. Are HELOC FPs closer to TP or TN?
For the old aggressive CatBoost threshold, HELOC FPs are **TP-like risky profile (distance FP-TP=0.434, FP-TN=0.703)**. For the analyzed manual-review high-risk bucket, they are **TP-like risky profile (distance FP-TP=0.313, FP-TN=0.786)**. HELOC FPs also show risk-like behavior under the selected feature set, but this should still be treated as screening evidence rather than proof of future default.

## 3. Was the old aggressive threshold excessive alarm?
Yes, operationally. The old aggressive threshold generated the largest FP count among the analyzed systems. Even when those FPs look partly risk-like, the volume is too high for automatic rejection. The correct interpretation is elevated-risk screening, not final adverse action.

## 4. Does the manual-review policy manage FP better?
Yes. The manual-review/high-risk policy shifts the most uncertain cases into review and makes the automatic high-risk bucket more defensible. This does not eliminate FP risk, but it reduces the chance that borderline non-default customers are automatically rejected.

## 5. Why is automatic rejection risky?
A false positive is a customer who did not default in the observed outcome window. Even if the profile resembles true defaulters, automatic rejection would convert model uncertainty into a hard operational decision. That is too strong for the observed precision levels.

## 6. Why is screening/manual review safer?
Screening/manual review preserves the model's risk signal while allowing human or policy-based checks for borderline cases. This is a better fit for low-to-moderate precision high-risk buckets and for domain-shift-sensitive credit risk settings.
