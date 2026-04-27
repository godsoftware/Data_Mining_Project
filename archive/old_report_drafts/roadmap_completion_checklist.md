# Roadmap Completion Checklist

Bu dosya, ilk verilen proje yol haritasındaki maddelerin mevcut proje klasöründe ne kadar tamamlandığını kontrol etmek için hazırlanmıştır.

Durum etiketleri:

- DONE: Tamamlandı ve çıktı üretildi.
- PARTIAL: Temel kısmı yapıldı, ama rapor/proje kalitesi için genişletilebilir.
- TODO: Henüz yapılmadı.
- OPTIONAL: İlk yol haritasında öneri olarak vardı; final proje için şart değil.

## Genel Durum

Proje teknik olarak çalışır durumda. Dataset indirildi, temizlendi, feature engineering yapıldı, modeller eğitildi, final model seçildi, threshold/calibration/SHAP/LIME/stability/faithfulness çıktıları üretildi.

Final model güncel kontrolden sonra XGBoost olarak seçildi.

- Final model: XGBoost, no resampling
- Test ROC-AUC: 0.7804
- Test PR-AUC: 0.5637
- Default threshold 0.50 recall: 0.3670
- Cost-sensitive threshold: 0.15
- Cost-sensitive recall: 0.7724
- SHAP stability mean pairwise Spearman rho: about 0.913

## 1. Projenin Nihai Amacı

Status: DONE

Yapılanlar:

- Default tahmini yapıldı.
- SHAP ve LIME ile açıklanabilirlik çıktıları üretildi.
- Calibration, threshold optimization, SHAP stability ve faithfulness testleri eklendi.

Kanıt dosyaları:

- `outputs/tables/final_test_metrics.csv`
- `outputs/tables/threshold_comparison.csv`
- `outputs/tables/calibration_results.csv`
- `outputs/tables/shap_top_features.csv`
- `outputs/tables/shap_stability_spearman.csv`
- `outputs/tables/faithfulness_test_results.csv`

## 2. Dataset Seçimi ve Gerekçelendirme

Status: DONE

Yapılanlar:

- Default of Credit Card Clients / Taiwan dataset seçildi.
- UCI kaynağından indirildi.
- 30,000 kayıt ve 23 predictor bilgisi doğrulandı.
- Target dağılımı doğrulandı: 23,364 non-default, 6,636 default.

Kanıt dosyaları:

- `data/raw/default_credit_card_clients.xls`
- `data/raw/default_credit_card_clients.csv`
- `outputs/tables/data_quality_report.json`
- `outputs/tables/eda_summary.json`

## 3. Proje Klasör Yapısı

Status: PARTIAL

Yapılanlar:

- `data/raw`, `data/interim`, `data/processed` oluşturuldu.
- `notebooks`, `src`, `outputs`, `report` klasörleri oluşturuldu.
- `requirements.txt` ve `README.md` yazıldı.

Eksik/kısmi:

- `report/final_report.docx` henüz yazılmadı.
- Presentation slides henüz hazırlanmadı.

Kanıt dosyaları:

- `README.md`
- `requirements.txt`
- `report/final_report_outline.md`

## 4. Değişkenleri Doğru Anlama

Status: DONE

Yapılanlar:

- Kolon isimleri anlamlı hale getirildi.
- Target `default_next_month` olarak yeniden adlandırıldı.
- Demografik, ödeme gecikmesi, fatura tutarı ve ödeme tutarı değişkenleri kodda ayrıldı.

Kanıt dosyaları:

- `src/data_preprocessing.py`
- `src/feature_engineering.py`
- `data/interim/cleaned_data.csv`

## 5. Veri Temizleme Planı

Status: DONE

Yapılanlar:

- Shape kontrolü yapıldı.
- Column isimleri kontrol edildi ve düzeltildi.
- Data type kontrolü yapıldı.
- Missing value kontrolü yapıldı.
- Duplicate kontrolü yapıldı.
- Target distribution çıkarıldı.
- EDUCATION anomalileri düzeltildi: 0, 5, 6 -> 4.
- MARRIAGE anomalisi düzeltildi: 0 -> 3.
- Negatif BILL_AMT değerleri raporlandı ve korunarak bırakıldı.

Sonuçlar:

- Missing value yok.
- Duplicate row yok.
- EDUCATION/MARRIAGE anomalileri temizlik sonrası yok.
- Negatif bill değerleri var ama domain açısından korunmuş durumda.

Kanıt dosyaları:

- `outputs/tables/data_quality_report.json`
- `data/interim/cleaned_data.csv`

## 6. Exploratory Data Analysis

Status: PARTIAL

Yapılanlar:

- Target class distribution grafiği üretildi.
- AGE distribution by default status üretildi.
- LIMIT_BAL distribution by default status üretildi.
- PAY_0 default rate grafiği üretildi.
- Correlation heatmap üretildi.

Eksik/kısmi:

- Default rate by EDUCATION/MARRIAGE/SEX ayrıca üretilmedi.
- BILL_AMT1-BILL_AMT6 dağılımları ayrı ayrı üretilmedi.
- PAY_AMT1-PAY_AMT6 dağılımları ayrı ayrı üretilmedi.
- Default rate by delay_count grafiği ayrıca üretilmedi.

Kanıt dosyaları:

- `outputs/figures/target_class_distribution.png`
- `outputs/figures/age_distribution_by_default.png`
- `outputs/figures/limit_bal_distribution_by_default.png`
- `outputs/figures/pay0_default_rate.png`
- `outputs/figures/correlation_heatmap.png`

## 7. Feature Engineering Planı

Status: DONE

Yapılanlar:

- `delay_count`
- `max_delay`
- `avg_delay`
- `recent_delay`
- `severe_delay_count`
- `total_bill_amt`
- `avg_bill_amt`
- `max_bill_amt`
- `bill_trend`
- `total_pay_amt`
- `avg_pay_amt`
- `max_pay_amt`
- `payment_to_bill_ratio`
- `utilization_proxy`

Kanıt dosyaları:

- `src/feature_engineering.py`
- `data/processed/model_ready_data.csv`

## 8. Preprocessing Planı

Status: DONE

Yapılanlar:

- Stratified train-test split kullanıldı.
- Test setine SMOTE uygulanmadı.
- Preprocessing pipeline içinde yapıldı.
- Numeric features StandardScaler ile işlendi.
- SEX, EDUCATION, MARRIAGE OneHotEncoder ile encode edildi.
- SMOTENC denendi.

Not:

- SMOTE yerine kategorik kolonlar olduğu için SMOTENC kullanıldı; bu teknik olarak daha doğru bir seçim.

Kanıt dosyaları:

- `src/model_training.py`
- `src/run_enhanced_modeling.py`
- `outputs/tables/enhanced_model_comparison.csv`

## 9. Modelleme Planı

Status: PARTIAL

Yapılanlar:

- Logistic Regression baseline eğitildi.
- Balanced Logistic Regression eğitildi.
- Decision Tree eğitildi.
- Random Forest eğitildi.
- HistGradientBoosting eğitildi.
- XGBoost eğitildi.
- LightGBM eğitildi.

Eksik/kısmi:

- Neural network uygulanmadı.
- Stacking ensemble uygulanmadı.

Not:

- Neural network ve stacking ilk planda opsiyonel olarak geçiyordu. Final XGBoost + SHAP/LIME + stability ekseni için şart değil.

Kanıt dosyaları:

- `outputs/tables/baseline_model_comparison.csv`
- `outputs/tables/enhanced_model_comparison.csv`
- `outputs/tables/cv_oof_model_threshold_comparison.csv`
- `outputs/models/final_model.joblib`

## 10. Hyperparameter Tuning Planı

Status: PARTIAL

Yapılanlar:

- Model karşılaştırması yapıldı.
- 5-fold out-of-fold model/threshold karşılaştırması eklendi.
- XGBoost final model olarak yeniden seçildi.

Eksik/kısmi:

- RandomizedSearchCV tam olarak çalıştırılmadı.
- Optuna uygulanmadı.
- Hyperparameter search result tablosu henüz yok.

Kanıt dosyaları:

- `src/model_training.py` içinde search space fonksiyonları hazır.
- `outputs/tables/cv_oof_model_threshold_comparison.csv`
- `outputs/tables/validation_threshold_test_results.csv`

## 11. Evaluation Planı

Status: DONE

Yapılanlar:

- Accuracy hesaplandı.
- Precision hesaplandı.
- Recall hesaplandı.
- F1-score hesaplandı.
- ROC-AUC hesaplandı.
- PR-AUC hesaplandı.
- Specificity hesaplandı.
- Confusion matrix grafikleri üretildi.
- Calibration curve üretildi.
- Brier score hesaplandı.

Eksik/kısmi:

- G-mean ayrıca tabloya eklenmedi.

Kanıt dosyaları:

- `outputs/tables/final_test_metrics.csv`
- `outputs/tables/calibration_results.csv`
- `outputs/figures/confusion_matrix_threshold_05.png`
- `outputs/figures/confusion_matrix_threshold_035.png`
- `outputs/figures/roc_curves.png`
- `outputs/figures/precision_recall_curves.png`
- `outputs/figures/calibration_curve.png`

## 12. Threshold Optimization

Status: DONE

Yapılanlar:

- 0.10-0.90 threshold aralığı test edildi.
- Precision, recall, F1, FP, FN, expected cost hesaplandı.
- FN cost = 5, FP cost = 1 varsayımı kullanıldı.
- Final XGBoost için en düşük test cost threshold değeri 0.15 çıktı.
- Ek olarak validation/CV threshold kontrolü yapıldı.

Kanıt dosyaları:

- `outputs/tables/threshold_comparison.csv`
- `outputs/tables/model_threshold_cost_comparison.csv`
- `outputs/tables/validation_threshold_test_results.csv`
- `outputs/tables/cv_oof_model_threshold_comparison.csv`

## 13. Calibration Planı

Status: DONE

Yapılanlar:

- Calibration curve üretildi.
- Brier score hesaplandı.
- CalibratedClassifierCV kullanıldı.
- Sigmoid calibration denendi.
- Isotonic calibration denendi.

Kanıt dosyaları:

- `outputs/tables/calibration_results.csv`
- `outputs/figures/calibration_curve.png`
- `outputs/models/final_model_calibrated_sigmoid.joblib`
- `outputs/models/final_model_calibrated_isotonic.joblib`

## 14. Explainability - SHAP Planı

Status: PARTIAL

Yapılanlar:

- SHAP global feature importance üretildi.
- SHAP beeswarm plot üretildi.
- SHAP bar plot üretildi.
- SHAP waterfall plot high-risk customer için üretildi.
- SHAP top features tablosu üretildi.

Eksik/kısmi:

- 3 farklı müşteri için local SHAP ayrı ayrı üretilmedi.
- Decision plot üretilmedi.
- Force plot üretilmedi.

Kanıt dosyaları:

- `outputs/tables/shap_top_features.csv`
- `outputs/shap_results/shap_beeswarm.png`
- `outputs/shap_results/shap_bar.png`
- `outputs/shap_results/shap_waterfall_high_risk_customer.png`

## 15. LIME Planı

Status: PARTIAL

Yapılanlar:

- Borderline customer için LIME explanation üretildi.
- LIME HTML ve PNG çıktısı üretildi.
- LIME feature weight tablosu üretildi.

Eksik/kısmi:

- Yanlış sınıflandırılmış müşteri için LIME yok.
- Doğru default / doğru non-default karşılaştırmalı LIME yok.

Kanıt dosyaları:

- `outputs/shap_results/lime_borderline_customer.html`
- `outputs/figures/lime_borderline_customer.png`
- `outputs/tables/lime_borderline_customer.csv`

## 16. SHAP Stability Analysis

Status: DONE

Yapılanlar:

- Final XGBoost modeli 10 farklı seed ile tekrar eğitildi.
- Her seed için top 10 SHAP feature listesi çıkarıldı.
- Spearman rank correlation hesaplandı.
- Top-5 frequency tablosu üretildi.

Sonuç:

- Mean pairwise Spearman rho yaklaşık 0.913.
- PAY_0, max_delay, recent_delay ve utilization_proxy tüm 10 seed'de top-5 içinde kaldı.

Eksik/kısmi:

- 20 veya 100 seed yapılmadı.
- Kendall's W hesaplanmadı.

Kanıt dosyaları:

- `outputs/tables/shap_stability_rankings.csv`
- `outputs/tables/shap_stability_spearman.csv`
- `outputs/tables/shap_top5_frequency.csv`
- `outputs/figures/shap_stability_top5_frequency.png`

## 17. Faithfulness Analysis

Status: DONE

Yapılanlar:

- SHAP top feature'ları permute edildi.
- AUC drop hesaplandı.
- Mean probability shift hesaplandı.
- Probability shift grafiği üretildi.

Kanıt dosyaları:

- `outputs/tables/faithfulness_test_results.csv`
- `outputs/figures/probability_shift_after_perturbation.png`

## 18. Model Seçim Stratejisi

Status: DONE

Yapılanlar:

- Tek accuracy ile model seçilmedi.
- ROC-AUC, PR-AUC, recall, F1, calibration, threshold cost ve stability dikkate alındı.
- Ek kontrolden sonra XGBoost final model seçildi.

Kanıt dosyaları:

- `outputs/tables/final_model_metadata.json`
- `outputs/tables/cv_oof_model_threshold_comparison.csv`
- `outputs/tables/enhanced_model_comparison.csv`
- `report/result_quality_check.md`

## 19. Final Rapor Yapısı

Status: PARTIAL

Yapılanlar:

- Final report outline hazırlandı.
- References dosyası başlatıldı.
- Result quality check yazıldı.
- Bu roadmap completion checklist yazıldı.

Eksik:

- Final report DOCX/PDF henüz yazılmadı.

Kanıt dosyaları:

- `report/final_report_outline.md`
- `report/references.bib`
- `report/result_quality_check.md`
- `report/roadmap_completion_checklist.md`

## 20. Research Gap

Status: PARTIAL

Yapılanlar:

- Research gap fikri README ve report outline içinde temsil edildi.
- Calibration + threshold + SHAP/LIME + stability + faithfulness birleşik çerçevesi teknik olarak uygulandı.

Eksik:

- Research gap final rapor metnine akademik paragraf olarak henüz yazılmadı.

Kanıt dosyaları:

- `README.md`
- `report/final_report_outline.md`
- `report/result_quality_check.md`

## 21. Mutlaka Üretilecek Tablolar

Status: PARTIAL

Üretilenler:

- Data cleaning decisions / quality report
- Baseline model comparison
- Enhanced model comparison
- Cross-validation / out-of-fold comparison
- Final test metrics
- Threshold comparison
- Calibration results
- SHAP top features
- SHAP stability results
- Faithfulness results
- LIME explanation table

Eksik/kısmi:

- Dataset variable description table ayrıca CSV olarak yok.
- Reviewed papers summary table yok.
- Feature engineering list table ayrıca CSV olarak yok.
- Hyperparameter search space/results table yok.

Kanıt dosyaları:

- `outputs/tables/*.csv`
- `outputs/tables/data_quality_report.json`

## 22. Mutlaka Üretilecek Grafikler

Status: PARTIAL

Üretilenler:

- Target class distribution
- Age distribution by default status
- LIMIT_BAL distribution by default status
- PAY_0 vs default rate
- Correlation heatmap
- ROC curves
- Precision-recall curves
- Confusion matrix
- Calibration curve
- SHAP beeswarm / summary
- SHAP bar plot
- SHAP waterfall high-risk customer
- LIME borderline customer
- SHAP stability top-5 frequency
- Probability shift after perturbation

Eksik/kısmi:

- Decision plot yok.
- Force plot yok.
- 3 ayrı local SHAP müşteri grafiği yok.

Kanıt dosyaları:

- `outputs/figures/*.png`
- `outputs/shap_results/*.png`
- `outputs/shap_results/*.html`

## 23. En Sık Yapılan Hatalar ve Önleme Listesi

Status: DONE

Kontrol:

- SMOTE/SMOTENC test setine uygulanmadı.
- Train-test split önce yapıldı.
- Sadece accuracy raporlanmadı.
- Target leakage yaratacak hedefe bağlı feature üretilmedi.
- EDUCATION/MARRIAGE anomalileri düzeltildi.
- Threshold 0.5'e sabitlenmedi.
- Calibration ve Brier score kullanıldı.
- Tek random seed ile açıklama yapılmadı; 10 seed stability eklendi.

Kanıt dosyaları:

- `src/model_training.py`
- `src/run_enhanced_modeling.py`
- `src/run_final_evaluation.py`
- `src/run_explainability_analysis.py`

## 24. Teslim Edilecek Nihai Proje Çıktıları

Status: PARTIAL

Hazır olanlar:

- Jupyter notebooks
- Cleaned dataset
- Model comparison tables
- Final trained model
- SHAP plots
- LIME example
- Evaluation metrics table
- README file

Eksik:

- Final report PDF/DOCX
- Presentation slides

Kanıt dosyaları:

- `notebooks/*.ipynb`
- `data/interim/cleaned_data.csv`
- `data/processed/model_ready_data.csv`
- `outputs/models/final_model.joblib`
- `outputs/tables/*.csv`
- `outputs/figures/*.png`
- `outputs/shap_results/*`
- `README.md`

## 25. Final Metodoloji Cümlesi

Status: PARTIAL

Yapılanlar:

- Metodoloji teknik olarak uygulandı.
- README ve report outline içinde proje çerçevesi yazıldı.

Eksik:

- Verilen final proposal paragrafı final rapora henüz yerleştirilmedi.

Kanıt dosyaları:

- `README.md`
- `report/final_report_outline.md`

## 26. Uygulama Sırası

Status: PARTIAL

Tamamlanan adımlar:

1. Dataset indirildi.
2. Klasör yapısı kuruldu.
3. Dataset okuma ve kolon düzeltme kodu yazıldı.
4. Missing, duplicate, dtypes kontrol edildi.
5. EDUCATION ve MARRIAGE anomalileri düzeltildi.
6. Target distribution çıkarıldı.
7. EDA grafiklerinin önemli kısmı üretildi.
8. Train-test split yapıldı.
9. Feature engineering fonksiyonları yazıldı.
10. Preprocessing pipeline kuruldu.
11. Logistic Regression eğitildi.
12. Random Forest eğitildi.
13. XGBoost eğitildi.
14. LightGBM eğitildi.
15. Resampling stratejileri karşılaştırıldı.
16. Model karşılaştırması ve CV/threshold kontrolü yapıldı.
17. Test setinde karşılaştırma yapıldı.
18. ROC-AUC, PR-AUC, F1, Recall, Precision hesaplandı.
19. Confusion matrix çıkarıldı.
20. Threshold optimization yapıldı.
21. Calibration uygulandı.
22. Final model seçildi.
23. SHAP global explanation üretildi.
24. SHAP local waterfall üretildi.
25. LIME borderline örneği üretildi.
26. SHAP stability analizi yapıldı.
27. Faithfulness testi yapıldı.
28. Tablolar outputs/tables içine kaydedildi.
29. Grafikler outputs/figures ve outputs/shap_results içine kaydedildi.
32. README tamamlandı.

Eksik/kısmi adımlar:

- 30. Final rapor henüz yazılmadı.
- 31. Presentation henüz hazırlanmadı.
- 33. Tüm notebook'ların baştan sona çalıştırılması yapılmadı; script tabanlı reproducibility çalıştırıldı.

## Completion Addendum

Bu checklist yazıldıktan sonra kalan eksiklerin büyük kısmı tamamlandı:

1. Final report Markdown ve DOCX taslakları üretildi.
2. Presentation PPTX ve Markdown slide notes üretildi.
3. LIME ve local SHAP için ek müşteri örnekleri üretildi.
4. EDA'ya default rate by EDUCATION, MARRIAGE, SEX ve delay_count grafikleri eklendi.
5. BILL_AMT ve PAY_AMT dağılım grafikleri eklendi.
6. Küçük RandomizedSearchCV tuning tablosu üretildi.
7. G-mean ve KS metrikleri eklendi.
8. Dataset variable description, reviewed papers summary, feature engineering list ve hyperparameter search space tabloları üretildi.
9. Remaining limitations ayrıca yazıldı.

PDF export Word COM tarafında bu ortamda güvenilir çalışmadığı için otomatik üretilemedi. DOCX dosyası hazırdır; PDF gerekiyorsa Word üzerinden Save as PDF yapılmalıdır.

Güncel final model:

- Tuned XGBoost, no resampling
- Recommended threshold: 0.15
- ROC-AUC: yaklaşık 0.783
- PR-AUC: yaklaşık 0.563
- Cost-sensitive recall: yaklaşık 0.784
- Expected cost: 3294
- SHAP stability mean Spearman rho: yaklaşık 0.972

## Son Karar

İlk yol haritasındaki ana bilimsel hedefler karşılandı:

- Büyük Taiwan dataset kullanıldı.
- Prediction yapıldı.
- Imbalance-aware evaluation yapıldı.
- Threshold optimization yapıldı.
- Calibration yapıldı.
- SHAP ve LIME üretildi.
- SHAP stability yapıldı.
- Faithfulness analysis yapıldı.

Proje rapor yazımına geçecek seviyededir. Eksik kalanlar daha çok rapor/presentation ve birkaç ek görsel-local explanation geliştirmesidir.
