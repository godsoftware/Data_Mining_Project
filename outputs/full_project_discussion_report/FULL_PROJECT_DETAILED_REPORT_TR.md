# FULL PROJECT DETAILED REPORT TR

Bu rapor mevcut proje çıktılarının teknik tartışma envanteridir. Bu aşamada yeni model eğitilmemiş, yeni feature üretilmemiş, yeni threshold seçilmemiş ve test set üzerinden yeni karar verilmemiştir. Sayısal sonuçlar mevcut dosyalardan okunmuştur; bulunamayan dosyalar envanterde not edilmiştir.

## Ana final karar

- Final operational version: **V2**
- Taiwan final policy: **V2 CatBoost manual-review**
- HELOC final policy: **V2 Scorecard manual-review**
- V3 final mi: **Hayır**
- Final Attempt final mi: **Hayır, appendix/stress-test**
- SCRE-Credit final classifier mı: **Hayır; reliability-aware framework ve review-prioritization support**
- Sistem automatic rejection mı: **Hayır; screening/manual-review decision-support**

## V2 ana sonuç tablosu

| dataset | final_policy | precision | recall | specificity | fp | fn | cost | manual_review_rate | audit_status | safe_category |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | V2 CatBoost manual-review | 0.529 | 0.575 | 0.854 | 680 | 564 | 2705.5 | 0.295 | READY | FINAL MAIN EVIDENCE |
| heloc | V2 Scorecard manual-review | 0.683 | 0.846 | 0.574 | 403 | 158 | 764.5 | 0.27 | READY | FINAL MAIN EVIDENCE |

## Weakness-closing özetinden kilit noktalar

# WEAKNESS CLOSING RESULTS FOR CHATGPT

## 1. Audit verdict
- Verdict: READY
- Critical issues: 0
- High issues: 1 (`validation_all` files include diagnostic test columns; not used for selection)
- Ready for conclusion: YES
- Test leakage: NO
- Metric consistency: PASS
- Manual-review consistency: PASS

## 2. V2 baseline
| Dataset | Policy | Precision | Recall | Specificity | FP | FN | Cost | MR Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | V2 CatBoost manual-review | 0.529 | 0.575 | 0.854 | 680 | 564 | 2705.500 | 0.295 |
| heloc | V2 Scorecard manual-review | 0.683 | 0.846 | 0.574 | 403 | 158 | 764.500 | 0.270 |

## 3. Manual-review metric repair
- Final Attempt repaired? YES, repaired into explicit 3x2 manual-review bucket tables.
- Critical failures remaining? NO for repaired 3x2 identities.
- Can Final Attempt be final? NO. It remains appendix/robustness only unless a future full audit promotes it.
- Use as appendix? YES.

## 4. Taiwan recall improvement
| Candidate | Recall | Precision | Specificity | FP | FN | Cost | MR Rate | Delta vs V2 | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan_recall_cand_003289 | 0.633 | 0.463 | 0.791 | 976 | 487 | 1993.200 | 0.295 | recall +0.058, FP +296, FN -77 | V3 audit candidate; recall improves but precision/specificity drop. |
| taiwan_recall_cand_003394 | 0.622 | 0.478 | 0.807 | 901 | 502 | 2034.300 | 0.289 | recall +0.047, FP +221, FN -62 | V3 audit candidate; recall improves but precision/specificity drop. |
| taiwan_recall_cand_003399 | 0.619 | 0.480 | 0.809 | 891 | 505 | 2025.600 | 0.291 | recall +0.044, FP +211, FN -59 | V3 audit candidate; recall improves but precision/specificity drop. |

## 5. Capacity-aware review
| Dataset | Model | Capacity | Capture Rate | Precision@K | Lift@K | Low-risk Default Rate | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | SCRE-Optimized probability ranking | 0.200 | 0.529 | 0.585 | 2.645 | 0.130 | Capacity/review-prioritization evidence, not final automatic decision. |
| taiwan | SCRE-Pareto probability ranking | 0.200 | 0.526 | 0.582 | 2.630 | 0.131 | Capacity/review-prioritization evidence, not final automatic decision. |
| taiwan | Best EBM/monotonic model: Monotonic LightGBM | 0.200 | 0.520 | 0.575 | 2.600 | 0.133 | Capacity/review-prioritization evidence, not final automatic decision. |
| heloc | SCRE-Optimized probability ranking | 0.200 | 0.340 | 0.884 | 1.698 | 0.429 | Capacity/review-prioritization evidence, not final automatic decision. |
| heloc | Best EBM/monotonic model: EBM | 0.200 | 0.338 | 0.878 | 1.689 | 0.431 | Capacity/review-prioritization evidence, not final automatic decision. |
| heloc | SCRE-Pareto probability ranking | 0.200 | 0.334 | 0.868 | 1.669 | 0.433 | Capacity/review-prioritization evidence, not final automatic decision. |

## 6. HELOC specificity improvement
| Candidate | Specificity | Recall | Precision | FP | FN | Cost | MR Rate | Delta vs V2 | Comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heloc_spec_cand_000065 | 0.676 | 0.782 | 0.724 | 307 | 224 | 1427.000 | 0.000 | specificity +0.101, FP -96, cost +662.5 | Specificity improves; cost increases; appendix/V3 candidate. |
| heloc_spec_cand_040371 | 0.631 | 0.811 | 0.705 | 349 | 194 | 1319.000 | 0.000 | specificity +0.057, FP -54, cost +554.5 | Specificity improves; cost increases; appendix/V3 candidate. |
| heloc_spec_cand_040365 | 0.621 | 0.815 | 0.700 | 359 | 190 | 1309.000 | 0.000 | specificity +0.046, FP -44, cost +544.5 | Specificity improves; cost increases; appendix/V3 candidate. |

## 7. Cost sensitivity and decision curve
| Dataset | Best Stable System | Cost Robust? | Useful Threshold Range | Comment |
| --- | --- | --- | --- | --- |
| Taiwan | V2 CatBoost manual-review | YES under core FN-dominant / moderate review-cost assumptions | Decision curves: SCRE ranking useful in broad ranges; not final selector | Cost uncertainty does not replace V2; identifies V3 candidates. |
| HELOC | V2 Scorecard manual-review | YES; SCRE threshold also competitive but not final | Decision curves: SCRE/capacity ranking useful for review prioritization | V2 remains safest operational evidence. |

## 8. Instance-dependent cost
| Dataset | Did final decision change? | Best System | Comment |
| --- | --- | --- | --- |
| Taiwan | PARTIALLY / not final | Taiwan recall candidate under proxy-cost definitions | Recall candidate lowers proxy instance-dependent cost, but remains V3 audit candidate. |
| HELOC | NO | V2 Scorecard manual-review | V2 remains best under HELOC proxy-cost definitions. |

## 9. SCRE strengthening
| Dataset | SCRE Role | Capture@20 | Lift@20 | Safe Claim |
| --- | --- | --- | --- | --- |
| taiwan | Review prioritization / reliability-aware framework | 0.529 | 2.645 | SCRE supports Top-K review prioritization; not dominant classifier. |
| heloc | Review prioritization / reliability-aware framework | 0.340 | 1.698 | SCRE supports Top-K review prioritization; not dominant classifier. |

## 10. Segment-aware threshold
| Dataset | Candidate | Better than V2? | Risk | Use |
| --- | --- | --- | --- | --- |
| Taiwan | limit/utilization segment policies | NO clear replacement | Overfitting; recall not improved beyond V2 | Appendix only |
| HELOC | revolving_burden_terciles | Specificity improves, cost worsens | Not audit-ready; trade-off heavy | Appendix / V3 candidate |

## 11. Temporal robustness
- Real temporal validation possible? NO.
- Result: No verified timestamp/application-date field exists. Only pseudo/order-based and random-partition diagnostics were produced.
- Use as: LIMITATION ONLY / APPENDIX DIAGNOSTIC.

## 12. V2 vs V3 final decision
| Dataset | Final Version | Reason |
| --- | --- | --- |
| Taiwan | V2 | Only audit-ready final operational evidence; recall candidate not audit-ready. |
| HELOC | V2 | Scorecard manual-review remains safest; specificity candidates increase cost. |

## 13. What improved?
- Taiwan recall: improved in candidate policy from V2 0.575 to up to 0.633, but candidate is not final.
- Manual review workload: segment/capacity candidates can reduce workload, but trade off recall/cost.
- HELOC specificity: improved in candidate policy from V2 0.574 to 0.676, but cost rises.
- Cost robustness: V2 remains stable under core assumptions; instance-dependent proxy analysis clarifies exposure-sensitive trade-offs.
- SCRE role: stronger as review-prioritization/reliability-aware framework, not classifier.
- Audit status: final weakness-closing audit READY with 0 critical issues.

## 14. What did not improve?
- No V3 candidate became audit-ready final replacement.
- Taiwan recall gains reduce precision/specificity and increase FP.
- HELOC specificity gains increase cost and reduce recall.
- Real temporal deployment validation remains impossible due to missing timestamps.
- Real bank loss/cost data remains unavailable; only sensitivity/proxy-cost analysis is possible.
- S


# 1. Executive Summary

## Ne yapıldı?
Proje kredi default tahmini ile başladı; zamanla salt model skoru üretmekten çıkıp validation-selected, calibration-aware, cost-sensitive ve manual-review destekli karar framework'üne dönüştü. Ana sonuç V2'dir.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_frozen_v2/final_readiness_check.md`
- `outputs/decision_revision_v2/final_audit_v2.md`

## Sonuç ne çıktı?
V2 READY; Final Attempt appendix; weakness-closing READY ama V3 final değil.

## İşe yaradı mı?
Evet, ana final karar için işe yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 final kaldı.

## Hocaya nasıl anlatılır?
Bu projede en önemli başarı en yüksek skoru kovalamak değil, metodolojik olarak savunulabilir final policy üretmektir.

# 2. Projenin Amacı

## Ne yapıldı?
Amaç kredi riski/default olasılığını tahmin etmek ve bu tahminleri gerçekçi karar destek politikasına dönüştürmekti.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `PROJECT_SCOPE.md`
- `outputs/final_positioning_statement.md`

## Sonuç ne çıktı?
Primary dataset Taiwan, external validation HELOC; SCRE framework olarak konumlandı.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Proje automatic rejection yerine screening/manual-review olarak sunuldu.

## Hocaya nasıl anlatılır?
Amaç bankanın tek tuşla ret sistemi değil, riskli başvuruları önceliklendiren bir karar destek çerçevesi.

# 3. Problem Tanımı

## Ne yapıldı?
Problem binary credit default classification olarak başladı; ancak imbalanced data, false positives, calibration, cost ve manual-review boyutları nedeniyle karar problemi haline geldi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision/current_problem_summary.md`
- `outputs/decision_revision_v2/SELECTION_PROTOCOL_V2.md`

## Sonuç ne çıktı?
Cost-minimization tek başına fazla agresif davranabildi.

## İşe yaradı mı?
Evet, problem doğru genişletildi.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 manual-review policy doğdu.

## Hocaya nasıl anlatılır?
Sadece 'default tahmin ettik' değil, hangi müşterinin insana gönderileceğini de tartıştık.

# 4. Dataset Açıklaması

## Ne yapıldı?
Taiwan ana dataset, HELOC external validation dataset olarak kullanıldı. German Credit aktif kapsamdan çıkarıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `data/processed/taiwan_model_ready.csv`
- `data/processed/heloc_model_ready.csv`
- `PROJECT_SCOPE.md`

## Sonuç ne çıktı?
Taiwan model-ready 41 kolon; HELOC model-ready 32 kolon olarak doğrulandı.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Taiwan ve HELOC rolleri ayrıldı.

## Hocaya nasıl anlatılır?
Taiwan ana deney, HELOC dış doğrulama/robustness gibi düşünülmeli.

# 5. Dataset Kolonları ve Kullanım Durumu

## Ne yapıldı?
Her kolon için anlam, modelde kullanım, cost/explainability/segment analizinde kullanım sözlüğe yazıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/DATASET_COLUMN_DICTIONARY.csv`

## Sonuç ne çıktı?
Kolon sözlüğü üretildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Raporun dataset appendix'ine girecek.

## Hocaya nasıl anlatılır?
ID ve target feature değildir; ödeme gecikme ve borç/ödeme kolonları temel risk sinyalleridir.

# 6. Data Cleaning ve Anomali İşleme

## Ne yapıldı?
Taiwan EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3; HELOC special codes NaN olarak temizlendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `src/data_preprocessing.py`
- `src/heloc_preprocessing.py`
- `outputs/full_project_discussion_report/data_cleaning_audit.csv`

## Sonuç ne çıktı?
Cleaning target-free ve leakage yaratmıyor.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Güvenli preprocessing olarak ana raporda anlatılacak.

## Hocaya nasıl anlatılır?
Anomali temizliği hedefe bakmadan yapıldı; bu yüzden leakage değil.

# 7. Missing / Duplicate / Target Kontrolleri

## Ne yapıldı?
Missing, duplicate, target distribution ve negative bill count kontrolleri toplandı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/data_cleaning_audit.csv`
- `outputs/tables/taiwan_data_audit.csv`
- `outputs/tables/heloc_data_audit.csv`

## Sonuç ne çıktı?
Taiwan target imbalanced; HELOC missing special-code kaynaklı olabilir.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Data audit kanıtı oluşturdu.

## Hocaya nasıl anlatılır?
Sınıf dengesizliği accuracy'nin neden yeterli olmadığını açıklıyor.

# 8. EDA ve İlk Bulgular

## Ne yapıldı?
Target dağılımı, sınıf dengesizliği, gecikme davranışları ve borç/ödeme değişkenleri incelendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/figures/taiwan_target_distribution.png`
- `outputs/figures/heloc_target_distribution.png`
- `outputs/tables/taiwan_data_audit.csv`

## Sonuç ne çıktı?
Ödeme gecikmeleri güçlü risk sinyali; class imbalance belirgin.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Metric ve threshold seçim gerekçesi güçlendi.

## Hocaya nasıl anlatılır?
EDA bize modelden önce problemin dengesiz ve cost-sensitive olduğunu gösterdi.

# 9. Feature Engineering

## Ne yapıldı?
Row-wise, target-free finansal davranış feature'ları üretildi: delay_count, severe_delay_count, utilization_proxy, payment_to_bill_ratio vb.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `src/feature_engineering.py`
- `src/features/heloc_features.py`
- `outputs/full_project_discussion_report/USED_UNUSED_FEATURES.csv`

## Sonuç ne çıktı?
Feature'lar target kullanmadı ve dataset-level fitted statistic kullanmadı.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Model/explainability/cost analizine destek verdi.

## Hocaya nasıl anlatılır?
Feature engineering hedef değişkenden türetilmedi; bu kritik güvenlik noktası.

# 10. Preprocessing Pipeline

## Ne yapıldı?
Imputation, scaling/encoding ve model pipeline'larının split sonrası fit edilmesi prensibi korundu.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `src/models/model_factory.py`
- `src/data/split_leakage.py`
- `outputs/tables/leakage_checklist.csv`

## Sonuç ne çıktı?
Preprocessing leakage riskleri audit edildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Leakage-free claim desteklendi.

## Hocaya nasıl anlatılır?
Scaler/encoder testten öğrenirse sonuç şişer; biz bu riski audit ettik.

# 11. Train / Validation / Test Ayrımı

## Ne yapıldı?
Train model fitting, validation policy seçimi, test locked evaluation için kullanıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision_v2/SELECTION_PROTOCOL_V2.md`
- `outputs/decision_revision_v2/final_table_methodology_note.md`

## Sonuç ne çıktı?
V2 test-ranking sorununu düzeltti.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 final seçimi güvenli hale geldi.

## Hocaya nasıl anlatılır?
Test seti karar seçmek için değil, seçilmiş kararı sınamak için kullanıldı.

# 12. Baseline Modeller

## Ne yapıldı?
Logistic Regression, Random Forest ve temel modeller benchmark olarak denendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/model_results_taiwan.csv`
- `outputs/tables/model_results_heloc.csv`
- `src/models/baselines.py`

## Sonuç ne çıktı?
Baseline'lar boosting/scorecard/SCRE için referans verdi.

## İşe yaradı mı?
Evet ama final olmadı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Appendix/baseline evidence.

## Hocaya nasıl anlatılır?
Baselines final değil ama güçlü modellerin gerçekten değer kattığını görmemizi sağlar.

# 13. Boosting Modelleri

## Ne yapıldı?
XGBoost, LightGBM, CatBoost ve monotonic varyantlar denendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/boosting_results_taiwan.csv`
- `outputs/tables/boosting_results_heloc.csv`
- `outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv`

## Sonuç ne çıktı?
Taiwan final operational model CatBoost manual-review oldu.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Taiwan final policy seçimini belirledi.

## Hocaya nasıl anlatılır?
Taiwan'da en dengeli audit-ready operasyonel sonuç CatBoost manual-review ile geldi.

# 14. Scorecard / WOE Logistic Regression

## Ne yapıldı?
WOE/Scorecard finansal yorumlanabilir baseline olarak kuruldu.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `src/models/scorecard.py`
- `outputs/tables/scorecard_results.csv`
- `outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv`

## Sonuç ne çıktı?
HELOC final operational policy Scorecard manual-review oldu.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
HELOC final model rolünü belirledi.

## Hocaya nasıl anlatılır?
HELOC'ta açıklanabilir scorecard final kalması governance açısından güçlü.

# 15. Imbalance Handling Denemeleri

## Ne yapıldı?
SMOTE/SMOTENC, class weights, scale_pos_weight, advanced imbalance ve literature reproduction denendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision/imbalance_training_variants_taiwan.csv`
- `outputs/final_attempt/imbalance_boosting/advanced_imbalance_summary.md`

## Sonuç ne çıktı?
Bazı trade-offlar iyileşti ama V2 yerine audit-ready final çıkmadı.

## İşe yaradı mı?
Appendix olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final kararı değiştirmedi.

## Hocaya nasıl anlatılır?
Imbalance çözümü sadece recall artırmak değil; FP, calibration ve cost da korunmalı.

# 16. Hyperparameter Tuning

## Ne yapıldı?
Optuna ile XGBoost/LightGBM/CatBoost tuning yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/optuna_trials.csv`
- `outputs/tables/best_params.csv`
- `outputs/tables/optuna_run_status.csv`

## Sonuç ne çıktı?
Taiwan 100, HELOC 50 trial gibi kapsamlı tuning kayıtları üretildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Güçlü candidate modeller üretildi.

## Hocaya nasıl anlatılır?
Tuning testle değil validation/CV mantığıyla yapılmalı; bu proje bunu ayırmaya çalıştı.

# 17. Calibration

## Ne yapıldı?
Sigmoid/isotonic calibration ve Brier/ECE ölçümleri yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/calibration_results_taiwan.csv`
- `outputs/tables/calibration_results_heloc.csv`
- `outputs/decision_revision_v2/calibration_threshold_caveat.md`

## Sonuç ne çıktı?
Probability reliability ayrı değerlendirildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Manual-review ve threshold politikaları için daha güvenilir olasılıklar sağladı.

## Hocaya nasıl anlatılır?
Calibration AUC artırmak için değil, olasılığı güvenilir yapmak için kullanıldı.

# 18. Threshold Optimization

## Ne yapıldı?
0.50, cost-optimal, precision/specificity constrained threshold politikaları denendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision/precision_constrained_test_results_taiwan.csv`
- `outputs/decision_revision_v2/validation_candidate_registry_all.csv`

## Sonuç ne çıktı?
Eski agresif threshold FP problemini gösterdi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 manual-review yaklaşımına geçişi motive etti.

## Hocaya nasıl anlatılır?
Threshold seçimi validation'da yapılmazsa test leakage olur; bu özellikle düzeltildi.

# 19. Cost-Sensitive Decision Layer

## Ne yapıldı?
FN/FP cost ve cost matrix sensitivity analizleri yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision/cost_matrix_sensitivity_taiwan.csv`
- `outputs/final_weakness_closing/cost_sensitivity/cost_decision_summary.md`

## Sonuç ne çıktı?
Cost varsayımlarına duyarlılık açıkça raporlandı.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Gerçek cost yokluğu limitation olarak yönetildi.

## Hocaya nasıl anlatılır?
Cost gerçek banka zararı değil; varsayım ve sensitivity analizidir.

# 20. Manual-Review Band Sistemi

## Ne yapıldı?
Binary karar üçlü bucket'a çevrildi: low risk, manual review, high risk.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_frozen_v2/manual_review_metric_definitions.md`
- `outputs/decision_revision_v2/manual_review_cost_model_all.csv`

## Sonuç ne çıktı?
V2 final policy manual-review destekli oldu.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Automatic rejection claim kaldırıldı.

## Hocaya nasıl anlatılır?
Belirsiz müşteriyi otomatik reddetmek yerine insana gönderiyoruz.

# 21. SCRE-Credit Framework

## Ne yapıldı?
SCRE-Credit performance, calibration, cost, stability ve faithfulness kanıtlarını ağırlıklandıran framework olarak geliştirildi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `src/models/scre_credit.py`
- `outputs/tables/scre_pareto_results_taiwan.csv`
- `outputs/final_weakness_closing/scre_prioritization/scre_role_summary.md`

## Sonuç ne çıktı?
SCRE dominant classifier değil; review-prioritization/framework rolü güçlü.

## İşe yaradı mı?
Evet, framework olarak.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final classifier olmadı ama proje katkısı olarak kaldı.

## Hocaya nasıl anlatılır?
SCRE'yi 'her şeyi yenen model' değil, güvenilirlik-aware framework diye anlatmalıyız.

# 22. Explainability: SHAP ve LIME

## Ne yapıldı?
Global/local SHAP, LIME ve agreement tabloları üretildi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/shap_top_features.csv`
- `outputs/tables/lime_local_features.csv`
- `outputs/tables/shap_lime_agreement.csv`

## Sonuç ne çıktı?
Model behavior explanation appendix'i oluştu.

## İşe yaradı mı?
Evet, supporting.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final seçim değil, yorumlama desteği.

## Hocaya nasıl anlatılır?
Açıklamalar causal proof değil; model davranışı sanity check.

# 23. SHAP Stability / Kendall’s W

## Ne yapıldı?
Seed bazlı SHAP stability, Kendall's W ve literature comparison üretildi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/kendalls_w_stability_taiwan.csv`
- `outputs/tables/kendalls_w_stability_heloc.csv`
- `outputs/tables/shap_stability_literature_comparison.csv`

## Sonuç ne çıktı?
Explanation reliability tartışması güçlendi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
SCRE/reliability framework anlatımını destekledi.

## Hocaya nasıl anlatılır?
Açıklama sadece tek koşuda değil, seed'ler arasında ne kadar stabil diye de kontrol edildi.

# 24. Faithfulness Test

## Ne yapıldı?
Feature deletion/insertion/perturbation analizleri ile açıklamaların model davranışına uyumu incelendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/faithfulness_results.csv`
- `outputs/tables/faithfulness_group_results.csv`

## Sonuç ne çıktı?
Causality değil, model behavior sanity check.

## İşe yaradı mı?
Appendix olarak evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final policy seçmedi.

## Hocaya nasıl anlatılır?
Top feature'lar silinince performans ne oluyor diye baktık; bu nedensellik iddiası değil.

# 25. Statistical Tests

## Ne yapıldı?
Bootstrap CI, paired tests, McNemar ve cleaned statistical tests üretildi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/tables/statistical_tests_cleaned.csv`
- `outputs/tables/statistical_tests_summary.csv`

## Sonuç ne çıktı?
p-value yön/winner ile temizlendi.

## İşe yaradı mı?
Supporting.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Overclaim riskini azalttı.

## Hocaya nasıl anlatılır?
Significant demek bizim model kazandı demek değildir; yönüne bakmak gerekir.

# 26. İlk Final Sonuçlar ve Problemler

## Ne yapıldı?
İlk final tablolar test metric ranking ambiguity içerdi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision/final_decision_revision_audit.md`

## Sonuç ne çıktı?
NOT READY kararı verildi.

## İşe yaradı mı?
Evet, hata yakalandı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 protokolüne geçildi.

## Hocaya nasıl anlatılır?
Bu aşama projeyi güçlendirdi çünkü hatayı saklamadık.

# 27. Decision Revision V1

## Ne yapıldı?
FP reduction, precision constraints, cost matrix, DCA, manual review, imbalance, eligibility ve FP analysis yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision/`

## Sonuç ne çıktı?
Retrospective evidence güvenli ama final selection olarak güvenli değildi.

## İşe yaradı mı?
Diagnostic olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 reset ihtiyacını doğurdu.

## Hocaya nasıl anlatılır?
V1 sonuçları öğretici ama final seçim kanıtı değil.

# 28. V2 Revision ve Final Operational Policy

## Ne yapıldı?
Validation-only candidate registry, manual-review cost correction, locked policy ve held-out evidence üretildi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision_v2/final_audit_v2.md`
- `outputs/final_frozen_v2/`

## Sonuç ne çıktı?
V2 READY ve final operational version oldu.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final karar V2.

## Hocaya nasıl anlatılır?
V2 projenin metodolojik olarak temiz final noktasıdır.

# 29. Final Attempt Denemeleri

## Ne yapıldı?
Literature reproduction, imbalance, DNN, EBM/monotonic, capacity, decision curve, final selector denendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_attempt/report_to_user/FINAL_ATTEMPT_RESULTS_FOR_CHATGPT.md`

## Sonuç ne çıktı?
Geniş stress-test yapıldı.

## İşe yaradı mı?
Appendix olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2'yi değiştirmedi.

## Hocaya nasıl anlatılır?
Son büyük deneme negatif sonuçlar dahil dürüstçe raporlandı.

# 30. Final Attempt Neden Final Olmadı?

## Ne yapıldı?
Audit BLOCKED: manual-review locked-test metric consistency failures.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_attempt/audit/final_attempt_audit.md`
- `outputs/final_weakness_closing/metric_repair/final_attempt_metric_repair_summary.md`

## Sonuç ne çıktı?
3x2 repair sonrası bile appendix only.

## İşe yaradı mı?
Final olarak hayır.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 korunur.

## Hocaya nasıl anlatılır?
Metric tutarsızlığı varsa sonuç iyi görünse bile final yapılmaz.

# 31. Weakness-Closing Denemeleri

## Ne yapıldı?
V2 zayıflıklarını kapatmak için recall, capacity, specificity, cost, instance cost, SCRE, segment, temporal analizleri yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md`

## Sonuç ne çıktı?
Audit READY, critical 0; ancak V3 final replacement olmadı.

## İşe yaradı mı?
Appendix/supporting olarak çok yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
V2 final kaldı.

## Hocaya nasıl anlatılır?
Zayıflıklar denendi ama daha iyi final denge çıkmadı.

# 32. Taiwan Recall İyileştirme Denemeleri

## Ne yapıldı?
Taiwan recall 0.575'ten 0.633'e çıkaran candidate bulundu.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv`

## Sonuç ne çıktı?
Precision 0.463'e düştü, FP 976'ya çıktı.

## İşe yaradı mı?
Trade-off kanıtı olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final olmadı.

## Hocaya nasıl anlatılır?
Daha çok default yakalamak daha fazla yanlış alarm doğurdu.

# 33. Capacity-Aware Review ve Lift@K

## Ne yapıldı?
Top-K review prioritization analizi yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv`

## Sonuç ne çıktı?
SCRE-Optimized Taiwan top20 capture 0.529, lift 2.645; HELOC capture 0.340, lift 1.698.

## İşe yaradı mı?
Evet, appendix güçlü.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
SCRE framework rolünü güçlendirdi.

## Hocaya nasıl anlatılır?
Banka sadece %20 inceleyebiliyorsa model sıralama aracı olarak işe yarıyor.

# 34. HELOC Specificity İyileştirme Denemeleri

## Ne yapıldı?
HELOC specificity candidate specificity'yi 0.676'ya çıkardı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv`

## Sonuç ne çıktı?
Recall 0.782'ye düştü, cost 1427'ye yükseldi.

## İşe yaradı mı?
Appendix olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final olmadı.

## Hocaya nasıl anlatılır?
İyi müşterileri daha iyi ayırmak maliyeti artırdı.

# 35. Cost Sensitivity ve Decision Curve

## Ne yapıldı?
FN/FP/review cost ve net benefit analizleri yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/cost_sensitivity/cost_decision_summary.md`

## Sonuç ne çıktı?
V2 core assumptions altında stabil kaldı; cost uncertainty final kararı değiştirmedi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Gerçek cost limitation'ı yönetildi.

## Hocaya nasıl anlatılır?
Gerçek maliyet yoksa en doğru savunma sensitivity analysis'tir.

# 36. Instance-Dependent Cost Analysis

## Ne yapıldı?
LIMIT/BILL/utilization ve HELOC risk proxy'leriyle örnek-bağımlı cost sensitivity yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/instance_cost/instance_cost_summary.md`

## Sonuç ne çıktı?
Taiwan recall candidate bazı proxy costlarda iyi; HELOC V2 best kaldı.

## İşe yaradı mı?
Appendix olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final karar değişmedi.

## Hocaya nasıl anlatılır?
Bu gerçek banka maliyeti değil, proxy-based sensitivity.

# 37. Segment-Aware Threshold Denemeleri

## Ne yapıldı?
Limit/utilization/recent delay ve HELOC risk burden segmentlerinde policy denendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/segment_thresholds/segment_policy_summary.md`

## Sonuç ne çıktı?
Taiwan'da net replacement yok; HELOC specificity trade-off var.

## İşe yaradı mı?
Appendix/diagnostic.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final olmadı.

## Hocaya nasıl anlatılır?
Segment policy overfitting riski taşır; dikkatli appendix.

# 38. Temporal Robustness / Deployment Simülasyonu

## Ne yapıldı?
Gerçek timestamp arandı; bulunamadı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/temporal_robustness/temporal_data_availability_check.md`

## Sonuç ne çıktı?
Gerçek temporal validation mümkün değil; limitation only.

## İşe yaradı mı?
Limitation olarak yaradı.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Deployment claim kaldırıldı.

## Hocaya nasıl anlatılır?
Tarih yoksa temporal validation iddiası kuramayız.

# 39. Final Audit ve Metric Consistency

## Ne yapıldı?
Final weakness-closing audit yapıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/audit/final_weakness_closing_audit.md`

## Sonuç ne çıktı?
READY, critical 0, test leakage NO, metric consistency PASS; high caveat diagnostic test columns.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Sonuca geçilebilir.

## Hocaya nasıl anlatılır?
Audit READY ama diagnostic columns selection evidence değildir.

# 40. Final Karar: Neden V2?

## Ne yapıldı?
V2 tek audit-ready final operational evidence olarak kaldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_frozen_v2/final_readiness_check.md`
- `outputs/final_weakness_closing/final_selection/final_selection_summary.md`

## Sonuç ne çıktı?
Use V2 as final YES; use V3 final NO.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Final rapor V2 üzerine kurulmalı.

## Hocaya nasıl anlatılır?
V2 sıkıcı ama güvenli; bilimsel olarak güvenli olan final budur.

# 41. Ne İşe Yaradı?

## Ne yapıldı?
V2 final policies, SCRE prioritization, capacity, cost sensitivity, XAI/stability ve claim control işe yaradı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/WHAT_WORKED_WHAT_FAILED.csv`

## Sonuç ne çıktı?
Worked/failed tablosu üretildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Rapor planını netleştirdi.

## Hocaya nasıl anlatılır?
İşe yarayanları final/appendix diye ayırdım.

# 42. Ne İşe Yaramadı?

## Ne yapıldı?
Final Attempt, DNN, KMeansSMOTE, advanced imbalance ve V3 candidates V2'yi değiştirmedi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/WHAT_WORKED_WHAT_FAILED.csv`

## Sonuç ne çıktı?
Negatif sonuçlar saklanmadı.

## İşe yaradı mı?
Evet, dürüst rapor için.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Overclaim engellendi.

## Hocaya nasıl anlatılır?
Başarısız deney de bilimsel kanıttır.

# 43. Neler Appendix’e Girecek?

## Ne yapıldı?
Final Attempt, weakness-closing candidates, capacity/lift, decision curve, XAI/stability, instance cost, segment diagnostics appendix'e girebilir.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_project_inventory/report_usage_plan.md`
- `outputs/full_project_discussion_report/TABLE_FIGURE_PLACEMENT_PLAN.csv`

## Sonuç ne çıktı?
Appendix planı oluşturuldu.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Ana hikâye şişmeyecek.

## Hocaya nasıl anlatılır?
Ana sonuç V2; appendix kapsamlı robustness.

# 44. Neler Ana Sonuç Olarak Kullanılacak?

## Ne yapıldı?
V2 frozen policies, V2 audit READY, final claim control, manual-review metric definitions ana sonuçtur.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_frozen_v2/`

## Sonuç ne çıktı?
Ana evidence net.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Rapor ana gövdesi V2'ye dayanır.

## Hocaya nasıl anlatılır?
Ana sonuç sayısı az ama temiz.

# 45. Neler Kesinlikle Claim Edilmeyecek?

## Ne yapıldı?
Automatic rejection, SCRE beats all, V3 final, Final Attempt replaces V2, gerçek deployment, gerçek bank cost iddiaları yasak.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/FINAL_CLAIM_CONTROL_TABLE.csv`

## Sonuç ne çıktı?
Claim control üretildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Güvenli akademik dil sağlandı.

## Hocaya nasıl anlatılır?
Ne söylemeyeceğimizi bilmek en az sonuç kadar önemli.

# 46. Literatürle Karşılaştırma

## Ne yapıldı?
Imbalanced learning, DNN, boosting, scorecard, XAI, cost, manual-review ve reliability framework temalarıyla karşılaştırıldı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/literature_positioning_table.csv`

## Sonuç ne çıktı?
Proje SOTA score değil audit-ready decision framework olarak konumlandı.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Literature discussion güvenli hale geldi.

## Hocaya nasıl anlatılır?
Düşük görünen skorlar daha dürüst protokolün sonucu olabilir.

# 47. Projenin Güçlü Yanları

## Ne yapıldı?
Leakage-aware protokol, V2 audit READY, manual-review policy, external HELOC, SCRE framework, claim control.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/decision_revision_v2/final_audit_v2.md`
- `outputs/final_frozen_v2/final_readiness_check.md`

## Sonuç ne çıktı?
Güçlü yanlar net.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Rapor conclusion desteklendi.

## Hocaya nasıl anlatılır?
Projenin gücü sadece model değil, validation disiplini.

# 48. Projenin Zayıf Yanları

## Ne yapıldı?
Taiwan recall orta, MR rate yüksek, HELOC specificity sınırlı, gerçek temporal/cost/deployment yok, Final Attempt blocked.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md`

## Sonuç ne çıktı?
Zayıflıklar saklanmadı.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Limitations güvenli yazılacak.

## Hocaya nasıl anlatılır?
Zayıflıkları gizlemedim; test ettim ve sınırlılık olarak yazdım.

# 49. Kalan Limitations

## Ne yapıldı?
Gerçek banka deployment, timestamp, richer financial data, actual loss/cost, fairness/legal proof yok.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/final_weakness_closing/temporal_robustness/temporal_robustness_summary.md`
- `outputs/final_frozen_v2/final_v2_do_not_claim.md`

## Sonuç ne çıktı?
Limitations net.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Claim sınırı belirlendi.

## Hocaya nasıl anlatılır?
Bu proje deployment değil, akademik validation framework.

# 50. Hocanın Sorabileceği Sorular ve Cevaplar

## Ne yapıldı?
60+ Q&A hazırlandı.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/TEACHER_QA_PREPARATION.md`

## Sonuç ne çıktı?
Hocaya tartışma hazırlığı üretildi.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Sunum savunması kolaylaştı.

## Hocaya nasıl anlatılır?
Zor sorulara doğrudan, iddiasız ve kanıtlı cevap vereceğiz.

# 51. Final Sonuç

## Ne yapıldı?
V2 final, V3 hayır, Final Attempt appendix, SCRE framework, automatic rejection hayır.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/FINAL_DECISION_EXPLAINED.md`

## Sonuç ne çıktı?
Proceed to conclusion YES.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Sonraki aşama final conclusion/report writing olabilir.

## Hocaya nasıl anlatılır?
Bu proje güvenli karar destek framework'ü olarak sunulmalı.

# 52. Ekler / Appendix Planı

## Ne yapıldı?
Appendix'e girecek tablo/figür ve kaynaklar belirlendi.

## Neden yapıldı?
Projenin kredi riski tahmini, karar politikası, açıklanabilirlik veya audit güvenilirliği açısından ilgili ihtiyacı karşılamak için yapıldı. Bu bölümde amaç sadece model skoru üretmek değil; leakage, calibration, threshold, cost, manual-review ve claim güvenliği açısından savunulabilir kanıt üretmekti.

## Nasıl yapıldı?
Mevcut scriptler ve çıktı dosyaları üzerinden çalışıldı. Bu rapor aşamasında yeni model, yeni feature, yeni threshold veya yeni deney üretilmedi; yalnızca var olan sonuçlar sınıflandırıldı ve yorumlandı.

## Hangi dosyalara dayanıyor?
- `outputs/full_project_discussion_report/TABLE_FIGURE_PLACEMENT_PLAN.csv`

## Sonuç ne çıktı?
Ana vs appendix ayrımı netleşti.

## İşe yaradı mı?
Evet.

## İşe yaramadıysa neden?
İşe yaramayan veya final olmayan deneylerde ana sebepler genellikle precision/recall/specificity/cost trade-off'u, metric consistency caveat'i, audit BLOCKED sonucu, validation-only final replacement kriterlerinin geçilememesi veya automatic rejection için yeterli güvenlik oluşmamasıdır.

## Final karara etkisi ne oldu?
Rapor şişmeden detay korunur.

## Hocaya nasıl anlatılır?
Ana gövde temiz; tüm ekstra deneyler appendix'te kanıt olarak durur.

# Dataset Kolon Sözlüğü Özeti

| dataset | column_name | type | meaning | raw_used | cleaned_used | engineered_from_it | used_in_model | used_in_cost_analysis | used_in_explainability | used_in_segment_analysis | excluded_reason | notes | used_in_scorecard | used_in_specificity_experiment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | ID | int64 | Müşteri satır kimliği. Model feature olarak kullanılmamalıdır. | True | True | False | False | False | False | False | ID excluded from model | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | LIMIT_BAL | int64 | Verilen kredi limiti / kredi kartı limiti. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | SEX | int64 | Cinsiyet kodu. Demografik değişkendir; fairness açısından dikkatli yorumlanır. | True | True | False | True | False | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | EDUCATION | int64 | Eğitim seviyesi. 0/5/6 anomalileri 4=Other altında birleştirilmiştir. | True | True | False | True | False | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | MARRIAGE | int64 | Medeni durum. 0 anomalisi 3=Other altında birleştirilmiştir. | True | True | False | True | False | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | AGE | int64 | Yaş. | True | True | False | True | False | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_0 | int64 | En güncel ödeme durumu / gecikme kodu. | True | True | True | True | False | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_2 | int64 | İki dönem önceki ödeme durumu / gecikme kodu. | True | True | True | True | False | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_3 | int64 | Üç dönem önceki ödeme durumu / gecikme kodu. | True | True | True | True | False | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_4 | int64 | Dört dönem önceki ödeme durumu / gecikme kodu. | True | True | True | True | False | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_5 | int64 | Beş dönem önceki ödeme durumu / gecikme kodu. | True | True | True | True | False | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_6 | int64 | Altı dönem önceki ödeme durumu / gecikme kodu. | True | True | True | True | False | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | BILL_AMT1 | int64 | En güncel fatura/borç tutarı. | True | True | True | True | True | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | BILL_AMT2 | int64 | İki dönem önceki fatura/borç tutarı. | True | True | True | True | True | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | BILL_AMT3 | int64 | Üç dönem önceki fatura/borç tutarı. | True | True | True | True | True | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | BILL_AMT4 | int64 | Dört dönem önceki fatura/borç tutarı. | True | True | True | True | True | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | BILL_AMT5 | int64 | Beş dönem önceki fatura/borç tutarı. | True | True | True | True | True | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | BILL_AMT6 | int64 | Altı dönem önceki fatura/borç tutarı. | True | True | True | True | True | True | False |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_AMT1 | int64 | En güncel ödeme tutarı. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_AMT2 | int64 | İki dönem önceki ödeme tutarı. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_AMT3 | int64 | Üç dönem önceki ödeme tutarı. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_AMT4 | int64 | Dört dönem önceki ödeme tutarı. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_AMT5 | int64 | Beş dönem önceki ödeme tutarı. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | PAY_AMT6 | int64 | Altı dönem önceki ödeme tutarı. | True | True | True | True | True | True | True |  | Target-free source column; cleaning/feature use documented. | NA | NA |
| taiwan | default_next_month | int64 | Temizlenmiş hedef değişken; 1=gelecek ay default, 0=non-default. | True | True | False | False | False | False | False | target; never used as feature | Only target; not a feature. | NA | NA |
| heloc | ExternalRiskEstimate | float64 | Dış kredi bürosu risk skoru. | True | True | False | True | True | True | True |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | MSinceOldestTradeOpen | float64 | En eski hesabın açılışından bu yana geçen ay. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | MSinceMostRecentTradeOpen | float64 | En yeni hesabın açılışından bu yana geçen ay. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | AverageMInFile | float64 | Dosyadaki hesapların ortalama ay yaşı. | True | True | True | True | True | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumSatisfactoryTrades | float64 | Tatmin edici hesap sayısı. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumTrades60Ever2DerogPubRec | float64 | 60+ gün gecikmeli veya derogatory public record sayısı. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumTrades90Ever2DerogPubRec | float64 | 90+ gün gecikmeli veya derogatory public record sayısı. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | PercentTradesNeverDelq | float64 | Hiç gecikmemiş hesap yüzdesi. | True | True | True | True | True | True | True |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | MSinceMostRecentDelq | float64 | Son gecikmeden bu yana geçen ay. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | MaxDelq2PublicRecLast12M | int64 | Son 12 aydaki maksimum gecikme/public record şiddeti. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | MaxDelqEver | int64 | Tüm tarihteki maksimum gecikme şiddeti. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumTotalTrades | float64 | Toplam hesap/trade sayısı. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumTradesOpeninLast12M | float64 | Son 12 ayda açılan hesap sayısı. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | PercentInstallTrades | float64 | Taksitli hesapların yüzdesi. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | MSinceMostRecentInqexcl7days | float64 | Son sorgudan bu yana ay; son 7 gün hariç. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumInqLast6M | float64 | Son 6 ay sorgu sayısı. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumInqLast6Mexcl7days | float64 | Son 6 ay sorgu sayısı; son 7 gün hariç. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NetFractionRevolvingBurden | float64 | Revolving kredi yükü oranı. | True | True | True | True | True | True | True |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NetFractionInstallBurden | float64 | Taksitli kredi yükü oranı. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumRevolvingTradesWBalance | float64 | Bakiyesi olan revolving hesap sayısı. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumInstallTradesWBalance | float64 | Bakiyesi olan taksitli hesap sayısı. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | NumBank2NatlTradesWHighUtilization | float64 | Yüksek kullanım oranına sahip banka/national hesap sayısı. | True | True | True | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | PercentTradesWBalance | float64 | Bakiyesi olan hesapların yüzdesi. | True | True | False | True | False | True | False |  | Special missing codes converted to NaN and imputed inside model pipelines. | True | True |
| heloc | bad_flag | int64 | Temizlenmiş hedef; 1=Bad, 0=Good. | True | True | False | False | False | False | False | target; never used as feature | Positive class = Bad. | False | False |

# Feature Engineering Özeti

| dataset | feature_name | source_columns | formula | reason | used_in_model | used_in_error_analysis | used_in_cost_analysis | used_in_explainability | risk_of_redundancy | kept_or_removed | uses_target | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | delay_count | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | count(PAY_0..PAY_6 > 0 after clipping negative pay-status codes to 0) | payment_delay_frequency | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | severe_delay_count | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | count(PAY_0..PAY_6 >= 2 after clipping negative pay-status codes to 0) | severe_payment_delay_frequency | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | max_delay | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | row-wise max(PAY_0..PAY_6 after clipping negative pay-status codes to 0) | maximum_delay_severity | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | avg_delay | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | row-wise mean(PAY_0..PAY_6 after clipping negative pay-status codes to 0) | average_delay_severity | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | recent_delay | PAY_0, PAY_2 | 1 if PAY_0 > 0 or PAY_2 > 0 after clipping negative pay-status codes to 0 else 0 | recent_payment_delay | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | total_bill_amt | BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6 | sum(BILL_AMT1..BILL_AMT6) | six_month_total_billed_amount | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | avg_bill_amt | BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6 | mean(BILL_AMT1..BILL_AMT6) | average_bill_amount | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | max_bill_amt | BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6 | max(BILL_AMT1..BILL_AMT6) | maximum_bill_amount | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | bill_trend | BILL_AMT1, BILL_AMT6 | BILL_AMT1 - BILL_AMT6 | recent_change_in_bill_balance | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | total_pay_amt | PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6 | sum(PAY_AMT1..PAY_AMT6) | six_month_total_payment_amount | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | avg_pay_amt | PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6 | mean(PAY_AMT1..PAY_AMT6) | average_payment_amount | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | max_pay_amt | PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6 | max(PAY_AMT1..PAY_AMT6) | maximum_payment_amount | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | payment_to_bill_ratio | total_pay_amt, total_bill_amt | total_pay_amt / (abs(total_bill_amt) + 1) | repayment_capacity_relative_to_balance | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | utilization_proxy | avg_bill_amt, LIMIT_BAL | avg_bill_amt / (LIMIT_BAL + 1) | credit_limit_utilization_proxy | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | recent_payment_intensity | PAY_AMT1, PAY_AMT2, BILL_AMT1, BILL_AMT2 | (PAY_AMT1 + PAY_AMT2) / (abs(BILL_AMT1) + abs(BILL_AMT2) + 1) | recent_repayment_intensity | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | bill_volatility | BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6 | row-wise population std(BILL_AMT1..BILL_AMT6) | bill_balance_variability | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | payment_volatility | PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6 | row-wise population std(PAY_AMT1..PAY_AMT6) | payment_amount_variability | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | delinquency_intensity | NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq | NumTrades60Ever2DerogPubRec + 2*NumTrades90Ever2DerogPubRec + max(0, 100 - PercentTradesNeverDelq) / 100 | delinquency_frequency_and_severity | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | trade_activity_ratio | NumTradesOpeninLast12M, NumTotalTrades | NumTradesOpeninLast12M / (NumTotalTrades + 1) | recent_trade_activity_relative_to_total_history | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | recent_inquiry_pressure | NumInqLast6Mexcl7days, MSinceMostRecentInqexcl7days | NumInqLast6Mexcl7days / (MSinceMostRecentInqexcl7days + 1) | recent_credit_inquiry_pressure | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | revolving_burden_proxy | NetFractionRevolvingBurden | NetFractionRevolvingBurden | revolving_credit_burden | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | installment_burden_proxy | NetFractionInstallBurden | NetFractionInstallBurden | installment_credit_burden | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | credit_history_length_proxy | MSinceOldestTradeOpen, AverageMInFile | row-wise max(MSinceOldestTradeOpen, AverageMInFile) | credit_history_length | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | negative_trade_signal | NumTrades60Ever2DerogPubRec, NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq | 1 if NumTrades60Ever2DerogPubRec > 0 or NumTrades90Ever2DerogPubRec > 0 or PercentTradesNeverDelq < 100 else 0 | negative_trade_history_indicator | True | True | False | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| heloc | high_utilization_signal | NetFractionRevolvingBurden, NumBank2NatlTradesWHighUtilization | 1 if NetFractionRevolvingBurden >= 80 or NumBank2NatlTradesWHighUtilization > 0 else 0 | high_credit_utilization_indicator | True | True | True | True | MEDIUM; several engineered features summarize overlapping payment/bill behavior. | kept | False | Row-wise, target-free feature engineering; allowed before split because no fitted statistics or target are used. |
| taiwan | default_next_month | default_next_month | target | Prediction label only. | False | True | False | False | NA | target_not_feature | True | Never used as model feature. |
| heloc | bad_flag | bad_flag | target | Prediction label only. | False | True | False | False | NA | target_not_feature | True | Never used as model feature. |

# Metric Sözlüğü

| metric_name | formula | ne_anlama_gelir | bu_projede_neden_kullanildi | hangi_durumda_yaniltici_olabilir | hangi_final_karar_icin_kullanildi |
| --- | --- | --- | --- | --- | --- |
| Accuracy | (TP+TN)/(TP+FP+FN+TN) | Genel doğru sınıflama oranı. | İlk bakış için var; imbalance altında ana karar değildir. | Default azsa yüksek accuracy yanıltabilir. | Final seçimde tek başına kullanılmadı. |
| Precision | TP/(TP+FP) | Riskli denilenlerin gerçekten default olma oranı. | False positive baskısını ölçmek için kritik. | Recall çok düşerse tek başına iyi görünür. | Operational/manual-review trade-off. |
| Recall | TP/(TP+FN) | Defaultların ne kadarını yakaladığımız. | Kredi riskinde kaçırılan default maliyetlidir. | Çok düşük threshold recall artırıp FP patlatabilir. | Taiwan weakness-closing ana hedeflerinden biri. |
| Specificity | TN/(TN+FP) | Non-defaultları doğru ayırma oranı. | İyi müşterileri yanlış alarmdan korur. | Recall ile trade-off yapabilir. | HELOC weakness-closing hedefi. |
| F1 | 2PR/(P+R) | Precision ve recall harmonik ortalaması. | Denge metriği. | Cost veya calibration'ı yansıtmaz. | Secondary metric. |
| ROC-AUC | TPR-FPR eğrisi alanı | Threshold bağımsız ayrım gücü. | Genel discrimination için. | Imbalanced datada PR-AUC kadar hassas olmayabilir. | Raw classifier track. |
| PR-AUC | Precision-Recall eğrisi alanı | Minority/default sınıf performansı. | Imbalanced default problemi için değerli. | Threshold/policy costunu direkt göstermez. | Raw classifier comparison. |
| Brier | mean((p-y)^2) | Probability reliability hata ölçüsü. | Calibration için. | AUC iyi olsa bile Brier kötü olabilir. | Calibration track. |
| ECE | bin bazlı beklenen calibration error | Tahmin olasılıklarının güvenilirliği. | Calibrated risk scores için. | Bin seçimine duyarlı. | Calibration and audit. |
| Cost | FN_cost*FN + FP_cost*FP | Yanlış karar maliyeti. | Cost-sensitive policy için. | Cost varsayımına bağımlı. | Threshold/cost analysis. |
| Review-adjusted cost | auto_FP*FP_cost + auto_FN*FN_cost + MR_count*review_cost | Manual-review iş yükünü içeren maliyet. | Manual-review policy için binary costtan daha adil. | Review cost varsayımsal. | V2 policy evidence. |
| Manual review rate | MR_count/N | İnsana giden oran. | Operasyonel kapasiteyi ölçer. | Düşük MR default kaçırabilir. | V2 and capacity. |
| High-risk precision | high_default/high_risk_count | High-risk bucket kalitesi. | Manual-review 3x2 metrik. | Binary precision ile karıştırılmamalı. | Manual-review analysis. |
| High-risk recall | high_default/total_defaults | High-risk bucket default yakalama. | Manual-review 3x2 metrik. | MR bucket defaultları dahil edilmeyebilir. | Manual-review analysis. |
| Low-risk default rate | low_default/low_risk_count | Low-risk bucket leakage. | Otomatik düşük risk karar güvenliği. | Low-risk çok küçükse oynak olur. | Manual-review analysis. |
| Capture@K | reviewed_defaults/total_defaults | Top-K review'da yakalanan default oranı. | Capacity-aware review için. | Direct classification değil. | SCRE/top-k appendix. |
| Precision@K | reviewed_defaults/review_count | İncelenen K%'de default yoğunluğu. | Banka review kapasitesi için. | Capacity seçimine bağlı. | Capacity analysis. |
| Lift@K | precision@K/base_default_rate | Random review'a göre kazanç. | Top-K ranking faydası. | Base rate'e bağlı. | Capacity analysis. |
| Net benefit | TP/N - FP/N*pt/(1-pt) | Treat-all/none karşısında karar faydası. | Decision curve için. | Threshold probability varsayımına bağlı. | Supporting decision evidence. |
| Kendall's W | rank agreement coefficient | SHAP ranking stability. | Explanation reliability için. | Model/seed sayısına bağlı. | XAI stability appendix. |
| Faithfulness score | Perturb/delete/insert behavior consistency | Explanationların model davranışıyla uyumu. | XAI sanity check. | Causal proof değildir. | Appendix. |

# Split ve Leakage Kontrol Tablosu

| process_step | train_used | validation_used | test_used | allowed_or_not | leakage_risk | final_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_data_loading | True | False | False | allowed | LOW | PASS | Raw files are read before splitting; no target-derived fitting. |
| cleaning_category_mapping | True | False | False | allowed | LOW | PASS | Taiwan EDUCATION/MARRIAGE mappings and HELOC special-code handling are target-free. |
| row_wise_feature_engineering | True | False | False | allowed | LOW | PASS | Features are row-wise and do not use target or fitted dataset statistics. |
| model_training | True | False | False | allowed | LOW | PASS | Models are fitted on train data. |
| preprocessing_fit | True | False | False | allowed only on train/pipeline folds | MEDIUM if done globally | PASS/guarded | Imputation/scaling/encoding should be fitted inside train-fold pipelines. |
| calibration_fit | False | True | False | allowed on validation/calibration split | MEDIUM | PASS with caveat | V2 documents calibration-policy caveat and no test calibration fit. |
| threshold_selection | False | True | False | allowed on validation only | CRITICAL if test used | PASS | V2 selection is validation-only. |
| manual_review_band_selection | False | True | False | allowed on validation only | CRITICAL if test used | PASS | Locked policy selected before held-out test. |
| final_locked_test_evaluation | False | False | True | allowed after locking | LOW | PASS | Test table contains no model-selection winner/rank. |
| all_candidate_test_diagnostics | False | False | True | diagnostic only | HIGH if used for selection | CAVEAT | Weakness-closing audit notes diagnostic test columns in validation_all files; not selection evidence. |
| resampling | True | False | False | train-fold only | CRITICAL if split-before or test-resampled | PASS by protocol | Strict literature reproduction forbids split-before SMOTE and test resampling. |
| final_claim_generation | False | False | False | audit-gated only | HIGH | PASS | Claim control forbids automatic rejection and SCRE-dominance claims. |

# Model ve Policy Sonuç Özeti

| dataset | stage | model_or_policy | policy_type | audit_status | safe_category | precision | recall | specificity | f1 | pr_auc | roc_auc | brier | ece | fp | fn | tp | tn | cost | manual_review_rate | capture_at_20 | lift_at_20 | main_strength | main_weakness | use_in_report_as | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taiwan | V2 Frozen | V2 CatBoost manual-review | manual_review_band | READY | FINAL MAIN EVIDENCE | 0.529 | 0.575 | 0.854 |  |  |  |  |  | 680 | 564 |  |  | 2705.5 | 0.295 |  |  | Audit-ready final operational evidence. | Manual-review screening only; not automatic rejection. | FINAL MAIN EVIDENCE | Final V2 Taiwan operational policy. Use as main report evidence. This is a manual-review screening policy, not automatic rejection. |
| heloc | V2 Frozen | V2 Scorecard manual-review | manual_review_band | READY | FINAL MAIN EVIDENCE | 0.683 | 0.846 | 0.574 |  |  |  |  |  | 403 | 158 |  |  | 764.5 | 0.27 |  |  | Audit-ready final operational evidence. | Manual-review screening only; not automatic rejection. | FINAL MAIN EVIDENCE | Final V2 HELOC operational policy. Use as main external-validation evidence. This is a manual-review screening policy, not automatic rejection. |
| taiwan | Weakness Closing | taiwan_recall_cand_003399 | manual_review_band | READY package but not final replacement | SUPPORTING / APPENDIX EVIDENCE | 0.4798598949211909 | 0.6194423511680482 | 0.8093301947357158 |  |  |  |  |  | 891 | 505 |  |  | 2025.6 | 0.291 |  |  | Recall increased versus V2. | Precision/specificity dropped and FP increased. | Appendix / weakness-closing discussion | Recall improves with bounded FP and manual-review workload; candidate worth audit review. |
| taiwan | Weakness Closing | taiwan_recall_cand_003394 | manual_review_band | READY package but not final replacement | SUPPORTING / APPENDIX EVIDENCE | 0.4779837775202781 | 0.6217030896759608 | 0.8071902418146801 |  |  |  |  |  | 901 | 502 |  |  | 2034.3 | 0.2888333333333333 |  |  | Recall increased versus V2. | Precision/specificity dropped and FP increased. | Appendix / weakness-closing discussion | Recall improves with bounded FP and manual-review workload; candidate worth audit review. |
| taiwan | Weakness Closing | taiwan_recall_cand_003289 | manual_review_band | READY package but not final replacement | SUPPORTING / APPENDIX EVIDENCE | 0.4625550660792951 | 0.6330067822155238 | 0.791140594906912 |  |  |  |  |  | 976 | 487 |  |  | 1993.2 | 0.2953333333333333 |  |  | Recall increased versus V2. | Precision/specificity dropped and FP increased. | Appendix / weakness-closing discussion | Recall improves with bounded FP and manual-review workload; candidate worth audit review. |
| heloc | Weakness Closing | heloc_spec_cand_040365 | threshold_only | READY package but not final replacement | SUPPORTING / APPENDIX EVIDENCE | 0.7000835421888053 | 0.8151750972762646 | 0.6209081309398099 | 0.7532584269662921 | 0.8038770280230678 | 0.801480407101681 | 0.1816523806088388 | 0.014207949481563 | 359 | 190 |  |  | 1309.0 | 0.0 |  |  | Specificity and FP improved. | Cost increased and recall declined. | Appendix / weakness-closing discussion | Specificity gain is limited. |
| heloc | Weakness Closing | heloc_spec_cand_000065 | threshold_only | READY package but not final replacement | SUPPORTING / APPENDIX EVIDENCE | 0.7236723672367237 | 0.7821011673151751 | 0.675818373812038 | 0.7517531556802244 | 0.8051123452356737 | 0.8027911200226807 | 0.1812034550674869 | 0.0190903471944552 | 307 | 224 |  |  | 1427.0 | 0.0 |  |  | Specificity and FP improved. | Cost increased and recall declined. | Appendix / weakness-closing discussion | Specificity improves while recall remains reasonably high; V3 audit candidate. |
| heloc | Weakness Closing | heloc_spec_cand_040371 | threshold_only | READY package but not final replacement | SUPPORTING / APPENDIX EVIDENCE | 0.7049873203719358 | 0.811284046692607 | 0.6314677930306231 | 0.7544097693351425 | 0.8038770280230678 | 0.801480407101681 | 0.1816523806088388 | 0.014207949481563 | 349 | 194 |  |  | 1319.0 | 0.0 |  |  | Specificity and FP improved. | Cost increased and recall declined. | Appendix / weakness-closing discussion | Specificity gain is limited. |
| heloc | Weakness Closing Capacity | SCRE-Optimized probability ranking | top-k review prioritization | Appendix evidence | SUPPORTING / APPENDIX EVIDENCE |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.2 | 0.3398247322297955 | 1.6982633453711808 | Review prioritization evidence. | Ranking support, not final classifier. | Appendix / SCRE role | Top-20% review prioritization result. |
| heloc | Weakness Closing Capacity | SCRE-Pareto probability ranking | top-k review prioritization | Appendix evidence | SUPPORTING / APPENDIX EVIDENCE |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.2 | 0.3339824732229795 | 1.669066840866232 | Review prioritization evidence. | Ranking support, not final classifier. | Appendix / SCRE role | Top-20% review prioritization result. |
| heloc | Weakness Closing Capacity | V2 final locked policy ranking | top-k review prioritization | Appendix evidence | SUPPORTING / APPENDIX EVIDENCE |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.2 | 0.3281402142161636 | 1.6398703363612834 | Review prioritization evidence. | Ranking support, not final classifier. | Appendix / SCRE role | Top-20% review prioritization result. |
| taiwan | Weakness Closing Capacity | SCRE-Optimized probability ranking | top-k review prioritization | Appendix evidence | SUPPORTING / APPENDIX EVIDENCE |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.2 | 0.5290128108515448 | 2.645064054257724 | Review prioritization evidence. | Ranking support, not final classifier. | Appendix / SCRE role | Top-20% review prioritization result. |
| taiwan | Weakness Closing Capacity | SCRE-Pareto probability ranking | top-k review prioritization | Appendix evidence | SUPPORTING / APPENDIX EVIDENCE |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.2 | 0.5259984928409948 | 2.6299924642049737 | Review prioritization evidence. | Ranking support, not final classifier. | Appendix / SCRE role | Top-20% review prioritization result. |
| taiwan | Weakness Closing Capacity | V2 final locked policy ranking | top-k review prioritization | Appendix evidence | SUPPORTING / APPENDIX EVIDENCE |  |  |  |  |  |  |  |  |  |  |  |  |  | 0.2 | 0.517709118311982 | 2.5885455915599094 | Review prioritization evidence. | Ranking support, not final classifier. | Appendix / SCRE role | Top-20% review prioritization result. |

# Worked / Failed Tablosu

| experiment | worked_or_failed | why | evidence_file | final_role | notes |
| --- | --- | --- | --- | --- | --- |
| V2 CatBoost manual-review Taiwan | WORKED AS FINAL | Audit-ready, validation-selected, locked test evidence; balanced FP control vs old aggressive policy. | outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv | FINAL MAIN EVIDENCE | Screening/manual-review only. |
| V2 Scorecard manual-review HELOC | WORKED AS FINAL | External dataset tarafında interpretable ve audit-ready operational policy. | outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv | FINAL MAIN EVIDENCE | Direct transfer claim değil. |
| SCRE top-K review prioritization | WORKED AS APPENDIX | SCRE-Optimized top-20 capture/lift iyi; framework rolünü güçlendirdi. | outputs/final_weakness_closing/scre_prioritization/scre_topk_review_taiwan.csv | APPENDIX / FRAMEWORK | Dominant classifier claim yok. |
| Taiwan recall candidate | WORKED AS APPENDIX | Recall 0.575'ten 0.633'e çıktı; FP arttı ve precision/specificity düştü. | outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv | APPENDIX / V3 candidate | V2'yi değiştirmedi. |
| HELOC specificity candidate | WORKED AS APPENDIX | Specificity 0.574'ten 0.676'ya çıktı; cost arttı ve recall düştü. | outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv | APPENDIX / V3 candidate | V2'yi değiştirmedi. |
| Capacity-aware review | WORKED AS APPENDIX | Top-K review prioritization hocaya anlatılabilir güçlü operasyonel analiz verdi. | outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv | APPENDIX / SUPPORTING | Automatic decision değil. |
| Instance-dependent cost | WORKED AS APPENDIX | Gerçek cost yokluğunu proxy sensitivity ile tartışılabilir hale getirdi. | outputs/final_weakness_closing/instance_cost/instance_cost_summary.md | APPENDIX | Gerçek bank loss değil. |
| Final Attempt | DID NOT REPLACE V2 | Metric consistency audit BLOCKED; repaired only for appendix. | outputs/final_attempt/audit/final_attempt_audit.md | APPENDIX ONLY | Final evidence olarak kullanılmayacak. |
| DNN/BP NN | DID NOT REPLACE V2 | Leakage-free reproduction V2'yi temiz şekilde geçmedi / final audit blocked package içinde kaldı. | outputs/final_attempt/deep_tabular/deep_tabular_summary.md | APPENDIX / NEGATIVE RESULT | Literature high-score overclaim yok. |
| KMeansSMOTE / literature reproduction | DID NOT REPLACE V2 | Doğal test dağılımında audit-ready operational replacement olmadı. | outputs/final_attempt/literature_reproduction/literature_reproduction_summary.md | APPENDIX / ROBUSTNESS | Split-before-SMOTE kuralı korunmalı. |
| Temporal robustness | LIMITATION | Gerçek timestamp yok; pseudo/order diagnostic limitation only. | outputs/final_weakness_closing/temporal_robustness/temporal_robustness_summary.md | LIMITATION | Gerçek deployment validation değil. |
| Real bank cost | LIMITATION | Gerçek zarar/fayda maliyeti yok; sensitivity/proxy cost kullanıldı. | outputs/final_weakness_closing/cost_sensitivity/cost_decision_summary.md | LIMITATION | Claim sınırı gerekli. |

# Claim Control Tablosu

| claim | safe_or_unsafe | why | evidence | correct_wording | wrong_wording |
| --- | --- | --- | --- | --- | --- |
| V2 is the final operational version. | SAFE | Final Attempt remains appendix/stress-test evidence only. | outputs/final_frozen_v2/final_v2_audit_summary.md; V2 audit READY. | V2 is used as the final operational version in this report. | All later experiments supersede V2. |
| Taiwan final policy is CatBoost manual-review. | SAFE | Use screening/manual-review wording. | outputs/final_frozen_v2/final_v2_taiwan_operational_policy.csv | For Taiwan, the final operational screening policy is V2 CatBoost manual-review. | CatBoost should automatically reject Taiwan applicants. |
| HELOC final policy is Scorecard manual-review. | SAFE | HELOC supports external-validation framework evidence, not direct model transfer. | outputs/final_frozen_v2/final_v2_heloc_operational_policy.csv | For HELOC, the final operational screening policy is V2 Scorecard manual-review. | The Taiwan model directly transfers to HELOC. |
| Final Attempt replaced V2. | UNSAFE | Final Attempt locked-test manual-review rows are not audit-clean. | outputs/final_frozen_v2/final_attempt_usage_status.md; Final Attempt audit BLOCKED. | Final Attempt is used as appendix robustness and stress-test evidence. | Final Attempt replaced V2 as the final operational version. |
| SCRE-Credit beats all individual models. | UNSAFE | CatBoost and Scorecard remain final operational winners for Taiwan and HELOC. | outputs/final_frozen_v2/final_v2_model_roles.md; outputs/final_project_inventory/scre_role_assessment.md | SCRE-Credit is evaluated as a reliability-aware framework and review-prioritization support tool. | SCRE-Credit outperforms all individual models. |
| SCRE-Credit is a reliability-aware framework. | SAFE | Framework claim is safe; universal dominance is not. | outputs/final_frozen_v2/final_v2_model_roles.md | SCRE-Credit is positioned as a reliability-aware framework that organizes performance, calibration, cost, stability, faithfulness, and external-validation evidence. | SCRE-Credit is a new fundamental machine-learning algorithm. |
| The model is suitable for automatic credit rejection. | UNSAFE | Precision/FP behavior and governance requirements require manual-review framing. | outputs/final_frozen_v2/final_v2_do_not_claim.md | The model is suitable for screening and manual-review support, not automatic rejection. | The model can automatically reject credit applicants. |
| The model is suitable for screening/manual-review prioritization. | SAFE | This is the safest operational framing. | outputs/final_frozen_v2/final_v2_main_result_table.md; outputs/final_frozen_v2/manual_review_metric_definitions.md | The final system should be presented as screening/manual-review prioritization support. | The final system is a fully automated lending decision engine. |
| Final Attempt can be used as appendix robustness evidence. | UNSAFE | Always mention BLOCKED audit if discussing final locked-test rows. | outputs/final_frozen_v2/final_attempt_usage_status.md | Final Attempt is included as appendix robustness evidence, clearly labeled as not final operational evidence. | Final Attempt results are the main final results. |
| Final Attempt locked-test rows are final evidence. | UNSAFE | Audit reported 9 critical metric-consistency failures. | outputs/final_attempt/audit/final_attempt_audit.md | Final Attempt locked-test rows are diagnostic only until metric-consistency issues are repaired. | Final Attempt locked-test rows are final audit-clean evidence. |
| Literature-inspired resampling/DNN gave a clean operational replacement. | UNSAFE | Use as negative/robustness finding. | outputs/final_attempt/report_to_user/FINAL_ATTEMPT_RESULTS_FOR_CHATGPT.md; outputs/final_project_inventory/literature_positioning_summary.md | Literature-inspired resampling and DNN experiments were useful robustness checks but did not cleanly replace V2. | KMeansSMOTE or DNN reproduced the high literature scores and replaced V2. |
| Manual-review cost is identical to binary FN/FP cost. | UNSAFE | Manual-review policies have low-risk, manual-review, and high-risk buckets. | outputs/final_frozen_v2/manual_review_metric_definitions.md | Manual-review cost is separate from binary FN/FP cost and includes workflow/review assumptions. | Manual-review cost can be interpreted exactly like binary expected cost. |
| V2 is the final audit-ready operational version. | SAFE | V2 audit READY and frozen. | outputs/final_frozen_v2/final_readiness_check.md | V2 is the final operational evidence. | Final Attempt replaced V2. |
| This is an automatic credit rejection system. | UNSAFE | Manual-review policy, precision and FP caveats remain. | outputs/final_frozen_v2/final_v2_do_not_claim.md | The system supports screening/manual-review prioritization. | The model can automatically reject applicants. |
| SCRE beats all models. | UNSAFE | Claim control says no; CatBoost/Scorecard win some operational roles. | outputs/tables/claim_control_matrix.csv | SCRE is a reliability-aware framework and review-prioritization support. | SCRE-Credit outperforms all individual models. |
| V3 is final. | UNSAFE | Weakness-closing selector kept V2. | outputs/final_weakness_closing/final_selection/final_selection_summary.md | V3 candidates are appendix/diagnostic candidates. | V3 replaces V2. |
| Real temporal validation was performed. | UNSAFE | No verified timestamp exists. | outputs/final_weakness_closing/temporal_robustness/temporal_data_availability_check.md | Only pseudo/order robustness diagnostics were possible. | The model was temporally validated for deployment. |

# Literature Positioning Tablosu

| literature_theme | typical_method | reported_strength | project_equivalent | project_result | stronger_or_weaker | safe_interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Imbalanced learning | SMOTE/ADASYN/weighted losses ile minority recall artırılır. | Promptlarda resampling ve weighting leakage-free denendi. | Recall/capacity trade-off açık incelendi. | Raw high-score hedefi değil; V2 replacement çıkmadı. | Farklı split/protokol nedeniyle doğrudan skor kıyası sınırlı. | Negative but useful robustness finding. |
| KMeansSMOTE | Bazı çalışmalarda büyük skor sıçraması iddia edilir. | Strict literature reproduction içinde denendi. | Split-before-resampling kuralı korunarak daha dürüst protokol. | V2 yerine audit-ready final replacement olmadı. | Literature protokolleri farklı olabilir. | Yüksek literatür skorları leakage-free protokolle kontrol edilmelidir. |
| BP Neural Network / DNN | Nonlinear tabular model ile yüksek AUC/accuracy iddiaları. | Final Attempt deep tabular reproduction. | Overclaim yapılmadı. | Audit-ready final model olmadı. | Hiperparametre/split farkları etkiler. | DNN negatif sonucu raporda dürüstçe appendix olarak verilmeli. |
| Boosting models | XGBoost/LightGBM/CatBoost credit scoring'de güçlüdür. | Taiwan final CatBoost; HELOC scorecard. | CatBoost Taiwan operational policy güçlü. | Tek model her dataset için kazanmadı. | Dataset schemas differ. | Model seçimi hedefe/dataset'e bağlıdır. |
| Scorecard | Regüle kredi riskinde yorumlanabilir baseline. | WOE/scorecard HELOC final. | HELOC'ta interpretable final policy. | Taiwan'da CatBoost kadar operasyonel olmadı. | Scorecard simplicity-performance trade-off. | Scorecard güçlü benchmark olarak kalır. |
| XAI / SHAP / LIME | Model açıklanabilirliği ve yerel/global açıklamalar. | SHAP/LIME, agreement, local cases üretildi. | Interpretability evidence zengin. | Causality claim yok. | Feature correlation affects explanations. | XAI model behavior sanity check'tir. |
| SHAP stability | Seed/ranking stability önemli. | Kendall's W ve top-k stability eklendi. | Reliability discussion güçlendi. | Lin & Wang gibi çalışmalarla birebir aynı değil. | Seed/model/feature differences. | Stability appendix/supporting evidence. |
| Cost-sensitive threshold | FN/FP maliyetleri threshold politikasını değiştirir. | FN/FP sensitivity, review adjusted cost, instance proxy cost. | Accuracy dışına çıkan karar mantığı. | Gerçek banka cost yok. | Proxy costs are assumptions. | Sensitivity analysis olarak sunulmalı. |
| Manual review / reject option | Belirsiz vakaları insan incelemesine gönderme. | V2 final manual-review policy. | Automatic rejection yerine güvenli decision support. | MR workload yüksek olabilir. | Operational capacity unknown. | Projenin ana katkı yönlerinden biri. |
| Decision curve / net benefit | Treat-all/treat-none karşısında karar faydası. | Decision curve analysis eklendi. | Decision usefulness tartışmasını güçlendirdi. | Final selector değil, supporting. | Threshold utility assumptions. | Supporting decision evidence. |
| Reliability framework | Performance+calibration+cost+stability+faithfulness entegrasyonu. | SCRE-Credit framework. | Structured comparison and prioritization. | Dominant classifier değil. | Framework claims differ from classifier claims. | SCRE reliability-aware framework olarak konumlanmalı. |

# Table/Figure Placement Planı

| table_or_figure | source_file | section_to_place | main_or_appendix | why | caption_suggestion |
| --- | --- | --- | --- | --- | --- |
| V2 final result table | outputs/final_frozen_v2/final_v2_*_operational_policy.csv | Executive Summary / V2 Ana Sonuçlar | main | Final audit-ready operational evidence. | Final V2 manual-review policies. |
| V2 vs V3 decision table | outputs/final_weakness_closing/final_selection/v2_vs_v3_selection_*.csv | Final Karar: Neden V2? | main | Shows why V3 did not replace V2. | V2 vs V3 final selection decision. |
| Taiwan recall candidate table | outputs/final_weakness_closing/taiwan_recall/taiwan_recall_policy_locked_test.csv | Weakness-Closing / Taiwan Recall | appendix | Recall improved but trade-offs remain. | Taiwan recall-improving V3 candidates. |
| Capacity-aware review table | outputs/final_weakness_closing/capacity_review/capacity_model_comparison.csv | Capacity-Aware Review | appendix/supporting | Shows top-K review value. | Default capture and lift at review capacity. |
| HELOC specificity candidate table | outputs/final_weakness_closing/heloc_specificity/heloc_specificity_locked_test.csv | HELOC Specificity | appendix | Specificity improved but cost increased. | HELOC specificity V3 candidates. |
| Cost sensitivity table | outputs/final_weakness_closing/cost_sensitivity/cost_sensitivity_all.csv | Cost Sensitivity | appendix/supporting | Cost assumption robustness. | Cost sensitivity across FN/FP and review costs. |
| Decision curve figure | outputs/final_weakness_closing/cost_sensitivity/decision_curve_taiwan.png | Decision Curve | appendix/supporting | Net benefit visualization. | Decision curve on Taiwan. |
| Lift@K figure | outputs/final_weakness_closing/capacity_review/lift_curve_taiwan.png | Capacity-Aware Review | appendix/supporting | Review prioritization curve. | Lift curve for top-K review. |
| Final audit summary | outputs/final_weakness_closing/audit/final_weakness_closing_audit.md | Final Audit | main | Audit status. | Final weakness-closing audit verdict. |
| Dataset column dictionary | outputs/full_project_discussion_report/DATASET_COLUMN_DICTIONARY.csv | Dataset Kolonları | main/appendix | Column meaning and usage. | Dataset variable dictionary. |
| Feature engineering inventory | outputs/full_project_discussion_report/USED_UNUSED_FEATURES.csv | Feature Engineering | main/appendix | Feature formulas. | Engineered feature inventory. |
| Claim control matrix | outputs/full_project_discussion_report/FINAL_CLAIM_CONTROL_TABLE.csv | Final Claim Control | main | Prevents overclaiming. | Safe and unsafe claims. |
| SCRE role table | outputs/final_weakness_closing/scre_prioritization/scre_topk_review_taiwan.csv | SCRE-Credit Framework | appendix/supporting | SCRE as ranking/framework. | SCRE top-K prioritization. |
| Weakness-closing summary | outputs/final_weakness_closing/send_to_chatgpt/WEAKNESS_CLOSING_RESULTS_FOR_CHATGPT.md | Weakness Closing | main/supporting | Single package summary. | Weakness-closing results summary. |

# Audit excerpts

## V2 audit excerpt

# Final Audit V2

Generated: 2026-05-05 15:19:59

## 1. Executive verdict

Final verdict: **READY**.

The previous NOT READY issue was test-set ranking/model-selection ambiguity. V2 fixes this: candidate registration and final selection are validation-only, final policies are locked before held-out test evaluation, and held-out test evidence contains no actual rank or winner column.

## 2. Critical issues

- None.

## 3. High issues

- None.

## 4. Medium issues

- None.

## 5. Leakage status

- Test set used for model/policy selection: **NO**.
- Test set used for threshold selection: **NO**.
- Test set used for manual-review band selection: **NO**.
- Test set used for calibration fitting: **NO**.
- Validation candidate and selection tables contain no `test_` metric columns.

## 6. Test-ranking status

- `final_heldout_test_evidence_table.csv` contains no actual model-selection rank/winner columns.
- `rank_column_allowed = NO` and `winner_column_allowed = NO` are audit guard columns, not ranking evidence.
- Rank/winner fields are restricted to validation-only selection artifacts.

Previous NOT READY issue fixed: **YES**.

## 7. Manual-review cost status

- Workload cost is explicitly modeled.
- Cost definitions are separated: automation-only residual, workload-adjusted cost, and imperfect-review adjusted cost.
- Review-cost sensitivity exists across review costs `0.1`, `0.25`, `0.5`, `1.0`, and `2.0` and multiple FN/FP scenarios.
- Manual-review cost is not presented as identical to binary expected cost.

Manual-review cost corrected: **YES**.

## 8. Calibration caveat status

- Same-validation calibration/threshold caveat is explicitly documented.
- It is correctly described as not test leakage.
- A separated calibration/policy validation check was added for final non-SCRE policies.
- SCRE full-stack separation remains documented as future work.

Calibration caveat acceptable: **YES**.

## 9. Metric consistency status

- Locked binary diagnostic metrics match confusion-matrix formulas.
- Locked manual-review metrics recompute from frozen test probabilities and locked bands.
- Review-adjusted cost and imperfect-review adjusted cost are consistent with the stated assumptions.

Metric consistency: **PASS**.

## 10. Final recommendation consistency

- Final recommendation is validation-selection first and held-out evidence second.
- It does not use test metrics to rank candidates.
- It rejects automatic rejection claims.
- It rejects SCRE dominance claims.
- It rejects causal feature/XAI claims.

Final recommendation consistency: **PASS**.

## 11. Ready for report?

**YES**.

## 12. Must fix before report

- None. V2 is ready for report writing.

## 13. Safe final claim

Pure cost minimization produced an overly aggressive screening model with high recall but excessive false positives. The revised validation-selected policies reduce false-positive pressure and improve operational interpretability through constrained thresholding, manual-review bands

## Final Attempt audit/report excerpt

# FINAL ATTEMPT RESULTS

This markdown is prepared to send to ChatGPT for strict external review. Please evaluate these results strictly. Do not overpraise. Tell me whether the final attempt truly improves the project or whether V2 should remain the final version.

## 1. Audit verdict

- Final audit verdict: **BLOCKED**
- Critical issue count: **9**
- High issue count: **4**
- Ready for report: **NO**
- Test leakage: **NO detected**
- Test ranking: **NO detected**
- Resampling leakage: **NO detected**
- Main blocker: **9 metric-consistency failures** in final locked-test manual-review rows. Displayed recall/specificity/F1 are not always recomputable from displayed `test_tn/test_fp/test_fn/test_tp`.

## 2. What was tried?

- strict literature reproduction
- advanced imbalance boosting
- deep tabular / BP NN
- EBM / monotonic interpretable models
- capacity-aware manual review
- decision curve
- final multi-objective selector
- locked test evaluation

## 3. V2 baseline reference

| Dataset | V2 final model/policy | Precision | Recall | Specificity | FP | FN | Cost | MR Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| taiwan | Best CatBoost / manual_review_band | 0.529 | 0.575 | 0.854 | 680 | 564 | 2705.5 | 0.295 |
| heloc | Best Scorecard / manual_review_band | 0.683 | 0.846 | 0.574 | 403 | 158 | 764.5 | 0.270 |

## 4. Best result from strict literature reproduction

| Dataset | Best Config | Precision | Recall | Specificity | F1 | PR-AUC | ROC-AUC | Cost | Beats V2? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| taiwan | XGBoost + no resampling | 0.375 | 0.775 | 0.633 | 0.551 | 0.564 | 0.789 | 3209.0 | Validation PR-AUC winner; not confirmed vs V2 on locked test |
| heloc | CatBoost Balanced | 0.574 | 0.975 | 0.215 | 0.758 | 0.799 | 0.803 | 873.0 | Validation PR-AUC winner; not confirmed vs V2 on locked test |

## 5. Best result from advanced imbalance boosting

| Dataset | Best Model | Variant | Calibration | Policy | Precision | Recall | Specificity | PR-AUC | Cost | Beats V2? |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| taiwan | XGBoost | xgboost_scale_pos_weight_5 | sigmoid | manual_review_band_mr_le_0_30 | 0.514 | 0.588 | 0.842 | 0.562 | 2648.5 | Validation: marginally yes; locked test: NO |
| heloc | CatBoost | catboost_baseline | isotonic | manual_review_band_mr_le_0_30 | 0.672 | 0.870 | 0.540 | 0.783 | 747.0 | Validation: NO vs V2 Scorecard |

## 6. Best result from deep tabular / BP NN

| Dataset | Best Model | Resampling/Loss | Precision | Recall | Specificity | PR-AUC | ROC-AUC | Cost | Beats V2? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| taiwan | MLP/BP Neural Network mlp_256-128-64_relu_base | none/binary_crossentropy | 0.687 | 0.354 | 0.954 | 0.552 | 0.780 | 4499.0 | NO |
| heloc | MLP/BP Neural Network mlp_128m64_l2_0.001 | none/binary_crossentropy | 0.730 | 0.762 | 0.694 | 0.811 | 0.804 | 1510.0 | NO operationally; raw PR-AUC only |

## 7. Best interpretable middle model

| Dataset | Best Model | Policy | Precision | Recall | Specificity | PR-AUC | Cost | Interpretability Comment | Beats Scorecard? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| taiwan | Monotonic LightGBM conservative | manual_review_mr_le_0_30 | 0.508 | 0.585 | 0.839 | 0.562 | 2668.0 | Governance-friendly middle/interpretable candidate | YES vs scorecard on Taiwan validation balance |
| heloc | WOE scorecard unweighted | m

## Weakness-closing audit excerpt

# Final Weakness Closing Audit

## 1. Executive verdict
**Verdict: READY**

READY criteria check:
- Critical issue count: 0
- High issue count: 1
- Medium issue count: 1
- Test leakage: NO
- Test ranking used for final selection: NO
- Metric/formula consistency failures: 0
- Manual-review metric consistency: PASS
- Final claim consistency: PASS

## 2. Critical issues
- None.

## 3. High issues
- all_candidate_test_columns_present: validation_all files include test metric columns: [('outputs\\final_weakness_closing\\taiwan_recall\\taiwan_recall_policy_validation_all.csv', 16290, 9), ('outputs\\final_weakness_closing\\heloc_specificity\\heloc_specificity_validation_all.csv', 122482, 14)]. Flags show not used for selection, but these must be treated as diagnostic-only. (outputs\final_weakness_closing\taiwan_recall)

## 4. Medium issues
- diagnostic_invalid_segment_rows: 3 validation-failed segment locked-test rows are present but summary labels them diagnostic-only. (outputs\final_weakness_closing\segment_thresholds)

## 5. Leakage status
- Explicit `test_set_used_*` flags are False or absent across audited CSVs.
- Explicit selection split columns are validation-only where present.
- Final selection keeps V2 as final and marks V3 candidates as non-final.
- HIGH caveat: `validation_all` files include test metric columns for all candidates. The flags and final selector show they were not used for selection, but these columns must be treated as diagnostic-only and not cited as selection evidence.

## 6. Metric consistency
- Binary precision/recall/specificity/F1 checks: PASS.
- Taiwan recall candidate inferred-TP consistency: PASS.
- Manual-review 3x2 formula checks: PASS.
- Cost formula checks: PASS.
- Capacity precision@K/capture@K/lift@K checks: PASS.
- Decision curve net benefit formula checks: PASS.

## 7. Manual-review status
Final Attempt manual-review rows are repaired into explicit 3x2 tables. Original binary denominator mismatch is documented. Final Attempt is not final.

## 8. V2 vs V3 status
- V3 did **not** become final.
- Taiwan recall candidate improves recall but is not audit-ready as replacement.
- HELOC specificity candidates improve specificity but increase cost and are not audit-ready as replacement.
- SCRE is strengthened as review-prioritization/framework evidence, not final classifier.

## 9. Cost and deployment limitations
- Cost sensitivity and instance-dependent cost are properly labeled as sensitivity/proxy analyses, not real bank loss.
- Temporal robustness is limitation-only because no verified timestamp exists.

## 10. Safe final claim
The weakness-closing package supports keeping V2 as the final operational version while documenting V3 candidates and appendix analyses. The project should be presented as a screening/manual-review support system, not an automatic rejection system. SCRE should be presented as a reliability-aware review-prioritization framework, not a universally superior classifier.

## 11. Ready for conclusion phase?
**YES**


# Hocaya 5 dakikada anlatım

1. Problem kredi default tahminiydi.
2. İlk modeller ve cost-threshold yaklaşımları fazla agresif davranıp FP sorununu büyütebildi.
3. Sadece accuracy/AUC yeterli olmadığı için precision, recall, specificity, cost, calibration ve manual-review metrikleri birlikte ele alındı.
4. V2 ile validation-only seçilen manual-review destekli karar sistemi kuruldu.
5. Taiwan'da final operational policy CatBoost manual-review, HELOC'ta Scorecard manual-review oldu.
6. SCRE-Credit final classifier değil, reliability-aware framework ve review-prioritization destek aracı olarak kaldı.
7. Final Attempt ve weakness-closing deneyleri ek robustluk sağladı ama V2'yi temiz şekilde geçmedi.
8. Son sistem automatic rejection değil, screening/manual-review support sistemidir.
9. Audit sonucu final V2 için READY; test leakage yok.
10. Sonuçlar abartılmadan, limitations açık yazılarak rapora geçilebilir.


# Maymuna anlatır gibi versiyon

Bankaya 100 kişi geliyor. Model bu kişileri risk sırasına koyuyor. Çok riskli görünenleri işaretliyor, emin olamadıklarını insana gönderiyor. İlk denemelerde daha çok default yakalamak için eşik düşürülünce yanlış alarm çok arttı. Sonra daha dengeli bir sistem kuruldu: Taiwan için CatBoost, HELOC için Scorecard. Daha fazla default yakalamak tekrar denendi ama bu kez yanlış alarm veya maliyet arttı. Bu yüzden en güvenli ve denetlenebilir final V2 oldu.
