# Final Decision Explained

## Net Karar

- Proceed to conclusion: **YES**
- Use V2 as final: **YES**
- Use V3 as final: **NO**
- Use Final Attempt as appendix: **YES**
- Use SCRE as final classifier: **NO**
- Use SCRE as framework/prioritization: **YES**
- Automatic rejection: **NO**
- Screening/manual-review support: **YES**

## Neden V2?

V2, test-set ranking problemini düzelten, validation-only selection kullanan ve locked held-out test evidence üreten audit-ready aşamadır. `outputs/decision_revision_v2/final_audit_v2.md` dosyasında critical/high issue yoktur. `outputs/final_frozen_v2/final_readiness_check.md` dosyasında V2 final operational evidence olarak kilitlenmiştir.

Taiwan için final policy V2 CatBoost manual-review policy'dir: precision 0.529, recall 0.575, specificity 0.854, FP 680, FN 564, review-adjusted cost 2705.5, manual-review rate 0.295. HELOC için final policy V2 Scorecard manual-review policy'dir: precision 0.683, recall 0.846, specificity 0.574, FP 403, FN 158, review-adjusted cost 764.5, manual-review rate 0.270.

## Neden V3 Değil?

Weakness-closing paketi V3 adaylarını üretti ancak final replacement kriterlerini geçirmedi. Taiwan recall candidate recall'ı 0.633'e kadar çıkardı fakat precision/specificity düştü ve FP arttı. HELOC specificity candidate specificity'yi 0.676'ya çıkardı fakat cost yükseldi ve recall düştü. Final weakness-closing audit READY olsa da final selector V2'yi korudu.

## Final Attempt Neden Appendix?

Final Attempt geniş bir robustness/stress-test paketidir ama audit verdict BLOCKED idi. Ana problem manual-review locked-test satırlarında binary confusion metrics ile 3x2 manual-review bucket metrics'in karışmasıydı. Daha sonra metric repair yapıldı; ancak bu repair Final Attempt'i final evidence haline getirmez. Final Attempt literature reproduction, DNN, EBM/monotonic, decision curve ve capacity analysis için appendix olarak değerlidir.

## SCRE-Credit'in Rolü

SCRE-Credit dominant classifier değildir. Taiwan'da operational final CatBoost, HELOC'ta Scorecard'dır. SCRE'nin güvenli rolü reliability-aware framework, model comparison structure ve review-prioritization/ranking support'tur. Capacity-aware review analizinde SCRE-Optimized top-20% review için Taiwan'da capture@20=0.529 ve lift@20=2.645 ile faydalı ranking sinyali vermiştir.

## Son Cümle

“Bu proje en yüksek skor alan default modeli olarak değil; leakage-aware, validation-selected, manual-review destekli ve reliability-aware kredi riski karar framework'ü olarak sunulmalıdır.”
