# Final V2 Model Roles

These roles are locked for final report writing. They should not be changed unless a new audit-clean result package is produced.

## Taiwan

- **Operational screening policy:** V2 CatBoost manual-review
- **Interpretable benchmark:** Scorecard / WOE, plus monotonic model as supporting evidence if needed
- **Research framework:** SCRE-Credit
- **Appendix robustness:** Final Attempt

Interpretation:

Taiwan's final operational evidence is the V2 CatBoost manual-review policy. It should be presented as a screening/manual-review support policy, not as an automatic rejection model. Scorecard / WOE remains the interpretable benchmark. Monotonic models may be used as additional supporting interpretability evidence, but they are not the final operational winner. SCRE-Credit should be discussed as the research framework.

## HELOC

- **Operational screening policy:** V2 Scorecard manual-review
- **Interpretable benchmark:** Scorecard / WOE
- **Research framework:** SCRE-Credit
- **Appendix robustness:** Final Attempt

Interpretation:

HELOC's final operational evidence is the V2 Scorecard manual-review policy. It is also the strongest interpretable benchmark for the external-validation side. This should be described as external-validation evidence for the framework and policy logic, not as proof that one Taiwan-trained model directly transfers to HELOC.

## SCRE-Credit

- **Not a universally superior classifier**
- **Reliability-aware framework**
- **Review-prioritization / capacity-analysis support where applicable**
- **Not final automatic decision model**

Interpretation:

SCRE-Credit should be positioned as a reliability-aware framework that organizes performance, calibration, cost, stability, faithfulness, and external-validation evidence. It should not be claimed as the best classifier in every setting. Where the capacity-aware review results support it, SCRE-Credit can be discussed as a review-prioritization or ranking-support component.

## Final Attempt

- **Role:** Appendix / robustness / stress-test evidence
- **Use:** Literature reproduction, advanced imbalance, DNN/BP NN, EBM/monotonic, capacity-review, and decision-curve appendix evidence
- **Do not use as:** Final operational evidence while the Final Attempt audit remains BLOCKED
