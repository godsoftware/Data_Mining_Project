# Teacher Q&A Preparation

## 1. [Dataset] Neden Taiwan dataset seçildi?

**Kısa cevap:** Taiwan daha büyük, kredi kartı default problemine doğrudan bağlı ve akademik literatürde yaygın bir benchmarktır.

**Teknik açıklama:** 30.000 satırlık kredi kartı default datası model, calibration, threshold ve manual-review deneyleri için German Credit'e göre daha güçlü bir deney alanı sağlar.

**Hocaya güvenli cümle:** Hocam, Taiwan ana veri olduğu için sonuçlar daha istatistiksel ve literatürle daha kıyaslanabilir.

## 2. [Dataset] HELOC neden kullanıldı?

**Kısa cevap:** HELOC external validation / robustness dataset olarak kullanıldı.

**Teknik açıklama:** Feature şeması Taiwan'dan farklıdır; bu yüzden direkt model transferi değil, framework mantığının başka kredi datasında davranışını görmek için değerlidir.

**Hocaya güvenli cümle:** HELOC'u aynı modeli kopyalamak için değil, yöntemin dış veri tarafında nasıl davrandığını görmek için kullandım.

## 3. [Dataset] German Credit neden ana dataset değil?

**Kısa cevap:** German Credit aktif kapsamdan çıkarıldı.

**Teknik açıklama:** Daha küçük ve proje kapsamını dağıtan bir veri olduğu için archive altında bırakıldı; ana kapsam Taiwan + HELOC olarak temizlendi.

**Hocaya güvenli cümle:** Kapsamı şişirmemek için German Credit'i final kapsamdan çıkardım.

## 4. [Dataset] Target nedir?

**Kısa cevap:** Taiwan'da `default_next_month`, HELOC'ta `bad_flag`.

**Teknik açıklama:** Her ikisinde de positive class riskli/default/bad performanstır.

**Hocaya güvenli cümle:** Pozitif sınıf kredi riski gerçekleşen müşteri anlamına geliyor.

## 5. [Dataset] ID neden çıkarıldı?

**Kısa cevap:** ID satır kimliğidir, tahmin edici finansal bilgi değildir.

**Teknik açıklama:** Model feature olarak kullanılması anlamsız ezber riski yaratabilir; bu yüzden model-ready datadan çıkarıldı.

**Hocaya güvenli cümle:** ID modelin müşteriyi tanımasını değil, risk örüntüsünü öğrenmesini istiyoruz.

## 6. [Dataset] EDUCATION anomalileri neydi?

**Kısa cevap:** Taiwan'da 0, 5, 6 gibi belirsiz kategoriler vardı.

**Teknik açıklama:** Bunlar literatürde yaygın şekilde 4=Other altında birleştirilir.

**Hocaya güvenli cümle:** Bu temizlik target kullanmadan yapıldığı için leakage değildir.

## 7. [Dataset] MARRIAGE anomalisi neydi?

**Kısa cevap:** Taiwan'da MARRIAGE=0 anomalisi vardı.

**Teknik açıklama:** Bu değer 3=Other altında birleştirildi.

**Hocaya güvenli cümle:** Bu kategori temizliği veri sözlüğüne uygun bir normalizasyon adımıdır.

## 8. [Dataset] PAY_0 neyi ifade eder?

**Kısa cevap:** En güncel ödeme gecikme durumunu ifade eder.

**Teknik açıklama:** Gecikme değişkenleri default riskinin en güçlü sinyallerindendir.

**Hocaya güvenli cümle:** Ödeme gecikmesi arttıkça default riski beklenen şekilde artıyor.

## 9. [Dataset] BILL_AMT negatif olabilir mi?

**Kısa cevap:** Evet, kredi bakiyesi/ters bakiye gibi domain durumları nedeniyle olabilir.

**Teknik açıklama:** Projede bu değerler silinmedi; sadece audit edildi.

**Hocaya güvenli cümle:** Negatif fatura tutarlarını körlemesine outlier diye atmadım.

## 10. [Dataset] HELOC special missing codes ne oldu?

**Kısa cevap:** -9/-8/-7 gibi özel kodlar NaN'e çevrildi.

**Teknik açıklama:** Model pipeline içinde train-fold imputation ile ele alındı.

**Hocaya güvenli cümle:** Missingness'i testten öğrenmedik; imputation pipeline içinde yapıldı.

## 11. [Methodology] Train/validation/test neden ayrıldı?

**Kısa cevap:** Model fitting, policy seçimi ve final değerlendirmeyi ayırmak için.

**Teknik açıklama:** Train model eğitir; validation threshold/policy seçer; test sadece locked final evidence verir.

**Hocaya güvenli cümle:** Test seti karar vermek için değil, kilitlenmiş kararı kontrol etmek için kullandım.

## 12. [Methodology] Test set ne zaman kullanıldı?

**Kısa cevap:** Final policy kilitlendikten sonra.

**Teknik açıklama:** V2'de held-out test evidence tablosu rank/winner içermez.

**Hocaya güvenli cümle:** Test seti sonucu seçmek için kullanmadım.

## 13. [Methodology] Leakage nasıl engellendi?

**Kısa cevap:** ID/target feature dışı bırakıldı, resampling train içinde tutuldu, threshold/calibration testte fit edilmedi.

**Teknik açıklama:** Audit dosyaları test leakage olmadığını raporladı.

**Hocaya güvenli cümle:** Son auditlerde test leakage NO olarak geçti.

## 14. [Methodology] Calibration nedir?

**Kısa cevap:** Model olasılıklarını gerçek default oranlarına daha uyumlu hale getirme işlemidir.

**Teknik açıklama:** Sigmoid/isotonic gibi yöntemler olasılık reliability için kullanılır.

**Hocaya güvenli cümle:** AUC yükseltmek değil, risk olasılığını daha güvenilir yapmak için yaptım.

## 15. [Methodology] Threshold tuning nedir?

**Kısa cevap:** 0.50 sabit eşik yerine validation üzerinde amaç/kısıta göre eşik seçmektir.

**Teknik açıklama:** Cost, precision, specificity ve recall trade-offları threshold ile değişir.

**Hocaya güvenli cümle:** Eşiği testte seçmedim; validation'da seçtim.

## 16. [Methodology] Manual-review band nedir?

**Kısa cevap:** Düşük risk, manual review ve yüksek risk olarak üç karar bucket'ı üretmektir.

**Teknik açıklama:** Belirsiz müşteriler otomatik karar yerine insana gönderilir.

**Hocaya güvenli cümle:** Bu sistem kredi reddi değil, inceleme önceliklendirme sistemidir.

## 17. [Methodology] Manual-review cost neden farklı?

**Kısa cevap:** Manual-review kararı binary high/low kararla aynı değildir.

**Teknik açıklama:** Review workload cost ayrıca eklenmelidir; 3x2 bucket mantığı kullanılır.

**Hocaya güvenli cümle:** Manual review maliyetini binary FN/FP cost ile karıştırmadım.

## 18. [Methodology] Validation-only selection ne demek?

**Kısa cevap:** Final model/policy sadece validation kanıtıyla seçildi.

**Teknik açıklama:** Test metrikleri ranking veya winner seçimi için kullanılmadı.

**Hocaya güvenli cümle:** Bu V2'nin NOT READY sorununu düzelten ana metodolojik nokta.

## 19. [Methodology] Locked test evaluation ne demek?

**Kısa cevap:** Model/policy/eşik sabitlendikten sonra testte sadece performans ölçmektir.

**Teknik açıklama:** Test sonucu kötü çıksa bile policy değiştirilmez.

**Hocaya güvenli cümle:** Bu yüzden test sonucunu seçim değil kanıt olarak sundum.

## 20. [Methodology] Resampling split öncesi yapıldı mı?

**Kısa cevap:** Final güvenli protokolde hayır.

**Teknik açıklama:** SMOTE/KMeansSMOTE gibi yöntemler train fold içinde tutulmalıdır.

**Hocaya güvenli cümle:** Split öncesi resampling olsaydı critical leakage olurdu; bunu yasakladım.

## 21. [Metrics] Accuracy neden ana metric değil?

**Kısa cevap:** Default sınıfı dengesiz olduğu için accuracy çoğunluk sınıfını iyi tahmin ederek yüksek görünebilir.

**Teknik açıklama:** Kredi riskinde default yakalama ve yanlış alarm ayrı ölçülmelidir.

**Hocaya güvenli cümle:** Accuracy tek başına bankanın karar kalitesini anlatmaz.

## 22. [Metrics] Precision nedir?

**Kısa cevap:** Riskli dediğimiz müşterilerin ne kadarının gerçekten default olduğu.

**Teknik açıklama:** False positive baskısını gösterir.

**Hocaya güvenli cümle:** Precision düşükse otomatik red tehlikelidir.

## 23. [Metrics] Recall nedir?

**Kısa cevap:** Gerçek defaultların ne kadarını yakaladığımız.

**Teknik açıklama:** FN maliyeti kredi riskinde önemlidir.

**Hocaya güvenli cümle:** Recall yükselirken FP artabilir; bu yüzden tek başına yeterli değildir.

## 24. [Metrics] Specificity nedir?

**Kısa cevap:** Non-default müşterileri doğru düşük risk ayırma oranı.

**Teknik açıklama:** İyi müşteriye yanlış alarm vermemek için önemlidir.

**Hocaya güvenli cümle:** HELOC weakness-closing specificity üzerine kuruldu.

## 25. [Metrics] PR-AUC neden önemli?

**Kısa cevap:** Imbalanced datada minority/default sınıf performansını ROC-AUC'den daha görünür yapar.

**Teknik açıklama:** Default prediction probleminde precision-recall trade-off doğrudan önemlidir.

**Hocaya güvenli cümle:** Raw classifier kıyaslarında PR-AUC kullandım.

## 26. [Metrics] ROC-AUC neden kullanıldı?

**Kısa cevap:** Threshold bağımsız genel ayrım gücünü gösterir.

**Teknik açıklama:** Ama operational threshold seçimi için tek başına yetmez.

**Hocaya güvenli cümle:** ROC-AUC modelin sıralama gücünü anlatır, karar politikasını değil.

## 27. [Metrics] Brier ve ECE ne işe yaradı?

**Kısa cevap:** Olasılıkların kalibrasyon kalitesini ölçtü.

**Teknik açıklama:** Bankacılıkta risk skoru olasılık gibi yorumlanacaksa reliability gerekir.

**Hocaya güvenli cümle:** Sadece iyi sıralama değil, güvenilir olasılık istedim.

## 28. [Metrics] Cost nasıl hesaplandı?

**Kısa cevap:** Binary cost FN_cost*FN + FP_cost*FP, manual-review cost ise workload ile ayrı hesaplandı.

**Teknik açıklama:** FN=5 FP=1 temel senaryo ve sensitivity analizleri kullanıldı.

**Hocaya güvenli cümle:** Cost varsayımdır; gerçek banka zararı değildir.

## 29. [Metrics] Capture@K nedir?

**Kısa cevap:** En riskli K% incelenirse defaultların kaçının yakalandığıdır.

**Teknik açıklama:** Manual-review capacity için en anlaşılır metriklerden biridir.

**Hocaya güvenli cümle:** Bankanın sadece %20 kişiyi inceleyebildiği senaryo gibi düşünülebilir.

## 30. [Metrics] Decision curve nedir?

**Kısa cevap:** Model kararının treat-all ve treat-none stratejilerine göre net faydasını ölçer.

**Teknik açıklama:** AUC/accuracy dışına çıkan karar faydası analizi sağlar.

**Hocaya güvenli cümle:** Model gerçekten karar destek faydası sağlıyor mu sorusuna bakar.

## 31. [Models] Neden CatBoost?

**Kısa cevap:** Categorical/numeric tabular kredi datasında güçlü boosting modelidir.

**Teknik açıklama:** Taiwan V2 final operational policy CatBoost manual-review oldu.

**Hocaya güvenli cümle:** Taiwan'da en audit-ready operational denge CatBoost ile geldi.

## 32. [Models] Neden Scorecard?

**Kısa cevap:** Kredi riskinde regülasyon ve yorumlanabilirlik açısından klasik benchmarktır.

**Teknik açıklama:** HELOC'ta V2 final policy Scorecard manual-review oldu.

**Hocaya güvenli cümle:** HELOC tarafında interpretable ve güçlü bir final kanıt verdi.

## 33. [Models] Neden XGBoost final değil?

**Kısa cevap:** Güçlü model olmasına rağmen final audit-ready operational role V2'de CatBoost/Scorecard'a geçti.

**Teknik açıklama:** Bazı appendix deneylerde iyiydi ama final replacement olmadı.

**Hocaya güvenli cümle:** XGBoost güçlü benchmark, fakat final policy winner değil.

## 34. [Models] Neden LightGBM final değil?

**Kısa cevap:** Weighted/monotonic/interpretable denemelerde kullanıldı, ama final V2'yi değiştirmedi.

**Teknik açıklama:** Probability calibration ve FP/recall trade-offları final rolünü sınırladı.

**Hocaya güvenli cümle:** LightGBM destekleyici model olarak kaldı.

## 35. [Models] Neden DNN final değil?

**Kısa cevap:** Leakage-free reproduction audit-ready final replacement üretmedi.

**Teknik açıklama:** Tabular credit scoring'de DNN her zaman boosting/scorecard'ı geçmeyebilir.

**Hocaya güvenli cümle:** DNN negatif sonucu saklamadım; appendix'e koydum.

## 36. [Models] Neden KMeansSMOTE final değil?

**Kısa cevap:** Doğal test dağılımında V2 yerine temiz operasyonel replacement olmadı.

**Teknik açıklama:** Resampling recall artırabilir ama calibration/precision/cost trade-off yaratabilir.

**Hocaya güvenli cümle:** Literatürdeki yüksek skorları leakage-free tekrar kontrol ettim.

## 37. [Models] SCRE neden final classifier değil?

**Kısa cevap:** SCRE tüm modelleri her metrikte geçmedi.

**Teknik açıklama:** CatBoost/Scorecard bazı operational rollerde daha güçlü kaldı.

**Hocaya güvenli cümle:** SCRE'yi classifier değil reliability-aware framework olarak konumlandırdım.

## 38. [Models] SCRE'nin katkısı ne?

**Kısa cevap:** Performance, calibration, cost, stability, faithfulness ve review prioritization kanıtlarını bir araya getiren çerçevedir.

**Teknik açıklama:** Top-K review prioritization tarafında faydalı destek sundu.

**Hocaya güvenli cümle:** SCRE proje katkısıdır ama dominant model iddiası değildir.

## 39. [Models] Scorecard neden HELOC'ta final?

**Kısa cevap:** HELOC'ta interpretable ve audit-ready manual-review policy en güvenli final kanıt oldu.

**Teknik açıklama:** External dataset tarafında açıklanabilir benchmark olması da avantajdır.

**Hocaya güvenli cümle:** HELOC'ta karmaşık model yerine scorecard daha savunulabilir kaldı.

## 40. [Models] Monotonic model ne işe yaradı?

**Kısa cevap:** Risk yönü belli feature'larda daha governance-friendly alternatif sundu.

**Teknik açıklama:** Fakat final operational policy olarak V2'yi geçmedi.

**Hocaya güvenli cümle:** Monotonic model appendix/interpretability kanıtıdır.

## 41. [Models] EBM ne işe yaradı?

**Kısa cevap:** Scorecard ile boosting arasında yorumlanabilir orta model ihtimalini test etti.

**Teknik açıklama:** Final Attempt BLOCKED olduğu için final kanıt değil; appendix olabilir.

**Hocaya güvenli cümle:** EBM, performans-yorumlanabilirlik tartışmasına destek verir.

## 42. [Results] Taiwan sonucu iyi mi kötü mü?

**Kısa cevap:** Operational olarak dengeli ama mükemmel değil.

**Teknik açıklama:** Precision 0.529, recall 0.575, specificity 0.854; manual-review destekli screening sistemi için audit-ready.

**Hocaya güvenli cümle:** Taiwan'da güçlü yan FP kontrolü; zayıf yan recall'ın orta seviyede kalması.

## 43. [Results] HELOC sonucu iyi mi kötü mü?

**Kısa cevap:** External validation tarafında güçlü recall ve yorumlanabilirlik var.

**Teknik açıklama:** Precision 0.683, recall 0.846, specificity 0.574; specificity zayıflığı kabul edildi.

**Hocaya güvenli cümle:** HELOC sonucu scorecard'ın dış veri tarafında hâlâ değerli olduğunu gösterdi.

## 44. [Results] Neden V2 final?

**Kısa cevap:** V2 audit READY ve final selection validation-only.

**Teknik açıklama:** Final Attempt BLOCKED, V3 adayları ise final replacement kriterlerini geçmedi.

**Hocaya güvenli cümle:** En güvenli final kanıt V2 olduğu için final V2 kalmalı.

## 45. [Results] Neden V3 değil?

**Kısa cevap:** V3 adayları bazı metrikleri iyileştirdi ama trade-off yarattı ve final replacement olmadı.

**Teknik açıklama:** Taiwan recall artsa da FP arttı; HELOC specificity artsa da cost yükseldi.

**Hocaya güvenli cümle:** Zayıflıkları kapatma denendi ama final dengesi V2'den daha güvenli olmadı.

## 46. [Results] Final Attempt neden appendix?

**Kısa cevap:** Audit BLOCKED: manual-review locked-test metric consistency hataları vardı.

**Teknik açıklama:** Sonra 3x2 repair yapıldı ama Final Attempt yine final evidence olmadı.

**Hocaya güvenli cümle:** Final Attempt'i robustluk/negatif sonuç appendix'i olarak kullanacağım.

## 47. [Results] Recall candidate neden final olmadı?

**Kısa cevap:** Recall 0.633'e çıktı ama precision 0.463'e düştü ve FP 976'ya çıktı.

**Teknik açıklama:** Yani daha çok default yakaladı ama daha fazla yanlış alarm üretti.

**Hocaya güvenli cümle:** Bu bir trade-off gösterimi; final replacement değil.

## 48. [Results] HELOC specificity candidate neden final olmadı?

**Kısa cevap:** Specificity 0.676'ya çıktı ama cost 1427'ye yükseldi ve recall 0.782'ye düştü.

**Teknik açıklama:** İyi müşterileri daha iyi ayırdı ama default kaçırma/maliyet arttı.

**Hocaya güvenli cümle:** Appendix'te trade-off olarak anlatılmalı.

## 49. [Results] Capacity-aware review ne gösterdi?

**Kısa cevap:** SCRE-Optimized top-20% review'da Taiwan default capture 0.529 ve lift 2.645 verdi.

**Teknik açıklama:** Bu SCRE'nin ranking/review prioritization rolünü güçlendirdi.

**Hocaya güvenli cümle:** Banka %20 inceleme kapasitesine sahipse model sıralama aracı olabilir.

## 50. [Results] Instance-dependent cost sonucu neydi?

**Kısa cevap:** Taiwan recall candidate proxy-cost açısından bazı tanımlarda iyi göründü; HELOC'ta V2 kaldı.

**Teknik açıklama:** Bu gerçek banka maliyeti değil, proxy sensitivity analizidir.

**Hocaya güvenli cümle:** Cost belirsizliğini dürüstçe tartışmak için kullanılır.

## 51. [Results] Temporal robustness sonucu neydi?

**Kısa cevap:** Gerçek timestamp yok, gerçek temporal validation yapılamadı.

**Teknik açıklama:** Sadece limitation/pseudo diagnostic olarak raporlanabilir.

**Hocaya güvenli cümle:** Deployment claim'i üretmiyorum.

## 52. [Weakness] Taiwan recall düşük değil mi?

**Kısa cevap:** Evet, orta seviyede bir zayıflık.

**Teknik açıklama:** Recall artırma denemeleri yapıldı ama FP ve precision trade-off'u doğdu.

**Hocaya güvenli cümle:** Bu zayıflığı saklamıyorum; final sistem screening desteği olarak sunuluyor.

## 53. [Weakness] Manual review rate yüksek değil mi?

**Kısa cevap:** Yaklaşık %29.5 Taiwan ve %27 HELOC, operasyonel olarak dikkate değer.

**Teknik açıklama:** Capacity-aware analizle %20/%25 review senaryoları incelendi.

**Hocaya güvenli cümle:** MR workload bir limitation ve operasyonel tasarım konusu.

## 54. [Weakness] HELOC specificity düşük değil mi?

**Kısa cevap:** Evet, 0.574 sınırlı.

**Teknik açıklama:** Specificity artırma denendi; FP düştü ama cost arttı ve recall düştü.

**Hocaya güvenli cümle:** Bu yüzden V2 değişmedi.

## 55. [Weakness] Gerçek deployment yoksa çalışma nasıl değerli?

**Kısa cevap:** Bu bir deployment değil, leakage-aware model validation ve decision-support framework çalışmasıdır.

**Teknik açıklama:** Temporal validation limitation olarak yazıldı.

**Hocaya güvenli cümle:** Gerçek deployment için zamanlı veri ve banka costu gerekir.

## 56. [Weakness] Gerçek cost yoksa cost analizi nasıl savunulur?

**Kısa cevap:** Cost sensitivity ve proxy instance-dependent cost olarak savunulur.

**Teknik açıklama:** Gerçek bank loss iddiası yapılmaz.

**Hocaya güvenli cümle:** Ben maliyeti varsayım olarak kullandım ve duyarlılık analizi yaptım.

## 57. [Literature] Literatürde daha yüksek skorlar var, neden sizin sonuçlar daha düşük?

**Kısa cevap:** Çünkü bazı literatür skorları farklı split, resampling veya leakage-riskli protokollerle raporlanmış olabilir.

**Teknik açıklama:** Biz doğal test distribution ve validation-only selection kullandık.

**Hocaya güvenli cümle:** Daha düşük ama daha dürüst/audit-ready sonuç ürettim.

## 58. [Literature] KMeansSMOTE neden final olmadı?

**Kısa cevap:** Resampling doğal testte operasyonel replacement üretmedi.

**Teknik açıklama:** SMOTE türleri minority recall'ı artırabilir ama precision/calibration/cost trade-off yaratabilir.

**Hocaya güvenli cümle:** Bu negatif sonuç literatüre karşı dürüst bir kontrol.

## 59. [Literature] DNN neden final olmadı?

**Kısa cevap:** Tabular credit data'da DNN her zaman boosting/scorecard'dan iyi değildir.

**Teknik açıklama:** Leakage-free protokolde final audit-ready replacement olmadı.

**Hocaya güvenli cümle:** DNN denendi ama sonuç saklanmadı.

## 60. [Literature] Bu çalışma neyi farklı yapıyor?

**Kısa cevap:** Sadece AUC değil; calibration, cost, threshold, manual review, audit, explainability ve framework rolünü birlikte ele alıyor.

**Teknik açıklama:** Final amaç SOTA skor değil audit-ready decision-support.

**Hocaya güvenli cümle:** Katkı bu bütünleşik validasyon mantığıdır.

## 61. [Claims] Bu sistem kredi reddi yapabilir mi?

**Kısa cevap:** Hayır, bu şekilde claim edilmemeli.

**Teknik açıklama:** Precision ve manual-review gerekliliği nedeniyle automatic rejection uygun değil.

**Hocaya güvenli cümle:** Screening/manual-review prioritization için kullanılabilir.

## 62. [Claims] Banka bunu kullanabilir mi?

**Kısa cevap:** Doğrudan deployment için değil; karar destek prototipi olarak değerlendirilebilir.

**Teknik açıklama:** Gerçek deployment için zamanlı validation, gerçek maliyet ve governance gerekir.

**Hocaya güvenli cümle:** Akademik proje olarak audit-aware framework sunuyor.

## 63. [Claims] SCRE'nin katkısı ne?

**Kısa cevap:** Reliability-aware framework ve review-prioritization desteği.

**Teknik açıklama:** SCRE performance, calibration, cost, stability, faithfulness ve capacity evidence'ı yapılandırır.

**Hocaya güvenli cümle:** SCRE'yi her şeyi yenen model diye değil, framework diye anlatmalıyım.

## 64. [Claims] Makale/bildiri çıkar mı?

**Kısa cevap:** Potansiyel olarak evet, ama claim doğru kurulmalı.

**Teknik açıklama:** Hikâye: leakage-aware, validation-selected, manual-review destekli credit-risk decision framework.

**Hocaya güvenli cümle:** SCRE'yi yeni temel algoritma değil framework olarak sunmak gerekir.

## 65. [Claims] Ana sonuç cümlesi ne olmalı?

**Kısa cevap:** V2 final audit-ready manual-review support system oldu.

**Teknik açıklama:** Taiwan'da CatBoost, HELOC'ta Scorecard; SCRE framework/ranking desteği.

**Hocaya güvenli cümle:** Proje en yüksek skor değil, güvenilir karar destek framework'ü olarak anlatılmalı.
