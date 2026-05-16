# Hocaya Anlatım

Bu projede başlangıçta amaç yalnızca beklenen maliyeti düşürmekti. Fakat Taiwan tarafında saf cost-minimization modeli çok agresif davrandı: recall yükseldi ama precision ve specificity düştü, false positive sayısı çok arttı. Bu yüzden projeyi otomatik kredi reddi sistemi gibi değil, manuel inceleme destekli bir screening sistemi gibi yeniden konumlandırdım.

V2 aşamasında final seçim protokolünü düzelttim. Model ve policy seçimleri validation set üzerinden yapıldı; test set yalnızca kilitlenmiş policy için held-out evidence olarak kullanıldı. V2 audit sonucu READY olduğu için ana raporda en güvenli final sonuç V2 olmalı. Taiwan için CatBoost manual-review policy, HELOC için Scorecard manual-review policy ana sonuç olarak kullanılabilir.

Sonrasında Final Attempt ile daha güçlü yöntemleri de denedim: SMOTE/KMeansSMOTE literatür reprodüksiyonu, class-weight/imbalance boosting, BP/DNN, EBM/monotonic modeller, capacity-aware review ve decision curve. Bunlar faydalı ek analizler üretti, ama Final Attempt V2'yi temiz şekilde geçmedi. Ayrıca final locked-test manual-review tablolarında metric-consistency audit problemi çıktığı için Final Attempt ana final sonuç olarak kullanılmamalı.

SCRE-Credit'i de abartmadan sunmak gerekiyor. SCRE her modeli yenen bir classifier değil; daha doğru rolü reliability-aware framework ve bazı capacity-review analizlerinde review-prioritization aracı olmasıdır. Sonuç olarak proje otomatik karar sistemi değil, kredi riski için screening/manual-review support sistemi olarak anlatılmalı.
