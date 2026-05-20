# Temporal Robustness Summary

## 1. Gerçek zamanlı validation mümkün mü?
**NO**. No verified timestamp/application-date variable is available for Taiwan or HELOC.

## 2. Mümkünse sonuçlar stabil mi?
Not applicable. Rolling-forward validation was not run because there is no real time variable.

## 3. Değilse bunu nasıl limitation olarak yazmalıyız?
Safe wording: *The datasets do not include verified application timestamps, so the project cannot claim true temporal or prospective deployment validation. We report only pseudo/order-based locked-test diagnostics and random-partition variability checks as limitations/appendix evidence.*

## 4. Pseudo/order-based diagnostic
- Taiwan: Row-order recall range=0.527-0.656, specificity range=0.839-0.881, PR-AUC range=0.513-0.641, mean score PSI=0.020; random-partition recall sd=0.031, specificity sd=0.009.
- HELOC: Row-order recall range=0.772-0.894, specificity range=0.525-0.611, PR-AUC range=0.755-0.856, mean score PSI=0.059; random-partition recall sd=0.020, specificity sd=0.030.

## 5. Real deployment eksikliği ne kadar kapanıyor?
Only slightly. These diagnostics check whether locked-test performance varies across row-order blocks and random partitions, but they do not replace real production-time validation.

## 6. Bu analiz ana sonuç mu, appendix mi?
**LIMITATION ONLY / APPENDIX DIAGNOSTIC.** It should not be used as main evidence of temporal robustness.
