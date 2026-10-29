### RQ1 — computed findings (default configurations)

**Diabetes (BRFSS 2015)**
- vs **Logistic Regression**: mean ROC-AUC Δ over sizes +0.9 pts; TabPFN-3 ahead on the seed-mean in 9/11 sizes and ahead in all 5 seeds in 7/11 sizes; largest gap +3.9 pts at n = 64.
- vs **Random Forest**: mean ROC-AUC Δ over sizes +1.2 pts; TabPFN-3 ahead on the seed-mean in 8/11 sizes and ahead in all 5 seeds in 6/11 sizes; largest gap +2.5 pts at n = 512.
- vs **XGBoost**: mean ROC-AUC Δ over sizes +4.4 pts; TabPFN-3 ahead on the seed-mean in 11/11 sizes and ahead in all 5 seeds in 10/11 sizes; largest gap +10.1 pts at n = 8.
  - regime tiny (8–16): seed-win rate Logistic Regression 40%, Random Forest 30%, XGBoost 80%; mean Δ Logistic Regression -0.9 pts, Random Forest -0.9 pts, XGBoost +5.5 pts.
  - regime small (32–128): seed-win rate Logistic Regression 80%, Random Forest 67%, XGBoost 100%; mean Δ Logistic Regression +2.2 pts, Random Forest +0.7 pts, XGBoost +3.4 pts.
  - regime medium (256–1,024): seed-win rate Logistic Regression 100%, Random Forest 100%, XGBoost 100%; mean Δ Logistic Regression +1.0 pts, Random Forest +2.2 pts, XGBoost +5.4 pts.
  - regime large (2,048–8,192): seed-win rate Logistic Regression 100%, Random Forest 100%, XGBoost 100%; mean Δ Logistic Regression +0.7 pts, Random Forest +2.1 pts, XGBoost +3.7 pts.

**Cardiovascular**
- vs **Logistic Regression**: mean ROC-AUC Δ over sizes +1.5 pts; TabPFN-3 ahead on the seed-mean in 10/11 sizes and ahead in all 5 seeds in 8/11 sizes; largest gap +5.5 pts at n = 64.
- vs **Random Forest**: mean ROC-AUC Δ over sizes +2.2 pts; TabPFN-3 ahead on the seed-mean in 10/11 sizes and ahead in all 5 seeds in 9/11 sizes; largest gap +4.5 pts at n = 128.
- vs **XGBoost**: mean ROC-AUC Δ over sizes +5.0 pts; TabPFN-3 ahead on the seed-mean in 11/11 sizes and ahead in all 5 seeds in 9/11 sizes; largest gap +8.5 pts at n = 128.
  - regime tiny (8–16): seed-win rate Logistic Regression 50%, Random Forest 70%, XGBoost 70%; mean Δ Logistic Regression -0.1 pts, Random Forest -0.9 pts, XGBoost +3.1 pts.
  - regime small (32–128): seed-win rate Logistic Regression 93%, Random Forest 80%, XGBoost 100%; mean Δ Logistic Regression +3.8 pts, Random Forest +2.7 pts, XGBoost +6.3 pts.
  - regime medium (256–1,024): seed-win rate Logistic Regression 100%, Random Forest 100%, XGBoost 100%; mean Δ Logistic Regression +0.8 pts, Random Forest +3.0 pts, XGBoost +6.3 pts.
  - regime large (2,048–8,192): seed-win rate Logistic Regression 100%, Random Forest 100%, XGBoost 100%; mean Δ Logistic Regression +0.9 pts, Random Forest +2.7 pts, XGBoost +3.7 pts.

**Data efficiency (default configs), TabPFN-3 at n = 128:**
- Diabetes (BRFSS 2015): Logistic Regression needs ≈ 218 rows; Random Forest needs ≈ 303 rows; XGBoost needs ≈ 3,571 rows.
- Cardiovascular: Logistic Regression needs ≈ 181 rows; Random Forest needs ≈ 1,012 rows; XGBoost needs ≈ 6,150 rows.

**Sub-chance runs:** 8 of 440 default/TabPFN runs have ROC-AUC < 0.5; 2 subset(s) are below 0.55 on the model-average (Figure 1) — indicating difficult data draws across the evaluated models.

### RQ2 — computed findings (sizes 32–8,192)

**Diabetes (BRFSS 2015)**
- **Logistic Regression**: HPO changes ROC-AUC by +1.2 pts on average (largest +4.9 pts at n = 64; improved on the seed-mean in 8/9 sizes). TabPFN-3 stays ahead after HPO in 6/9 sizes, is overtaken in 3, and the conventional model is ahead in both settings in 0; median share of the default advantage closed: 32%.
- **Random Forest**: HPO changes ROC-AUC by +1.3 pts on average (largest +1.8 pts at n = 512; improved on the seed-mean in 9/9 sizes). TabPFN-3 stays ahead after HPO in 7/9 sizes, is overtaken in 1, and the conventional model is ahead in both settings in 1; median share of the default advantage closed: 67%.
- **XGBoost**: HPO changes ROC-AUC by +2.7 pts on average (largest +4.0 pts at n = 1,024; improved on the seed-mean in 9/9 sizes). TabPFN-3 stays ahead after HPO in 9/9 sizes, is overtaken in 0, and the conventional model is ahead in both settings in 0; median share of the default advantage closed: 67%.

**Cardiovascular**
- **Logistic Regression**: HPO changes ROC-AUC by +0.9 pts on average (largest +3.6 pts at n = 32; improved on the seed-mean in 8/9 sizes). TabPFN-3 stays ahead after HPO in 9/9 sizes, is overtaken in 0, and the conventional model is ahead in both settings in 0; median share of the default advantage closed: 14%.
- **Random Forest**: HPO changes ROC-AUC by +1.9 pts on average (largest +2.8 pts at n = 512; improved on the seed-mean in 8/9 sizes). TabPFN-3 stays ahead after HPO in 9/9 sizes, is overtaken in 0, and the conventional model is ahead in both settings in 0; median share of the default advantage closed: 85%.
- **XGBoost**: HPO changes ROC-AUC by +4.0 pts on average (largest +5.8 pts at n = 256; improved on the seed-mean in 9/9 sizes). TabPFN-3 stays ahead after HPO in 9/9 sizes, is overtaken in 0, and the conventional model is ahead in both settings in 0; median share of the default advantage closed: 88%.

**At n = 8,192 after HPO** (ROC-AUC advantage of TabPFN-3, seeds won): Cardiovascular vs Logistic Regression +1.0 pts (5/5); Cardiovascular vs Random Forest +0.4 pts (5/5); Cardiovascular vs XGBoost +0.2 pts (5/5); Diabetes (BRFSS 2015) vs Logistic Regression +0.8 pts (5/5); Diabetes (BRFSS 2015) vs Random Forest +0.6 pts (5/5); Diabetes (BRFSS 2015) vs XGBoost +0.2 pts (5/5).

**CV optimism at n = 32** (best CV − test ROC-AUC): Cardiovascular Logistic Regression +4.3 pts; Cardiovascular Random Forest +6.7 pts; Cardiovascular XGBoost +4.5 pts; Diabetes (BRFSS 2015) Logistic Regression +4.0 pts; Diabetes (BRFSS 2015) Random Forest +11.2 pts; Diabetes (BRFSS 2015) XGBoost +11.2 pts.

### RQ3 — computed findings

**Pooled seed-level SD ratio (conventional ÷ TabPFN-3), default configurations** (>1 ⇒ TabPFN-3 varies less across subsets)
- Diabetes (BRFSS 2015) (LR/RF/XGB): tiny (8–16): 0.9×/1.0×/0.9× · small (32–128): 0.9×/0.7×/0.7× · medium (256–1,024): 3.0×/1.4×/3.1× · large (2,048–8,192): 1.6×/5.7×/8.1×
- Cardiovascular (LR/RF/XGB): tiny (8–16): 1.1×/1.1×/0.6× · small (32–128): 1.0×/1.0×/1.3× · medium (256–1,024): 1.1×/1.4×/2.2× · large (2,048–8,192): 0.7×/3.9×/4.7×

**Pooled seed-level SD ratio (conventional ÷ TabPFN-3), conventional models with HPO** (>1 ⇒ TabPFN-3 varies less across subsets)
- Diabetes (BRFSS 2015) (LR/RF/XGB): small (32–128): 0.8×/0.6×/0.8× · medium (256–1,024): 2.0×/2.0×/2.6× · large (2,048–8,192): 1.5×/4.2×/5.3×
- Cardiovascular (LR/RF/XGB): small (32–128): 1.0×/1.2×/1.0× · medium (256–1,024): 0.8×/2.0×/2.5× · large (2,048–8,192): 0.7×/1.2×/1.3×

Share of (dataset × model × size) cells in which TabPFN-3 has the smaller SD: default configurations 68%, conventional models with HPO 57% (single-cell SDs come from only five seeds).

Mean shared-variance share between TabPFN-3 and the other models by regime: tiny (8–16) 0.68, small (32–128) 0.56, medium (256–1,024) 0.38, large (2,048–8,192) 0.12.
Mean slope of log₂(SD) per doubling of n (default): Logistic Regression -0.82, Random Forest -0.53, TabPFN-3 -0.82, XGBoost -0.54.
Identified ROC-AUC asymptotes (default, n ≥ 32; fits on a bound are omitted): Cardiovascular: TabPFN-3 0.812, Logistic Regression 0.805; Diabetes (BRFSS 2015): TabPFN-3 0.825, Logistic Regression 0.827, Random Forest 0.814. Fits on a bound: 3/8.