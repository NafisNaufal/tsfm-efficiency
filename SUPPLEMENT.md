# Supplement to "How Efficient Is In-Context Forecasting?"

This supplement is not part of the paper and was not reviewed with it. ESANN permits linking supplementary material of this kind. Every number below is regenerated from the files in `results/`.

## S1. Data-generating processes and parameter distributions (`src/dgp.py`)
Each series draws its own parameters independently. Burn-in is 500 steps. Innovation variance is 1 unless stated otherwise.

| Family | Model | Parameter distribution |
|---|---|---|
| AR(1) | x_t = φ x_{t−1} + e_t | φ ~ U(−0.95, 0.95) |
| AR(3) | stationary AR(3) | partial autocorrelations ~ U(−0.9, 0.9)³, mapped by Durbin–Levinson |
| ARMA(1,1) | x_t = φ x_{t−1} + e_t + θ e_{t−1} | φ, θ ~ U(−0.9, 0.9), rejecting \|φ+θ\| ≤ 0.2 (near cancellation) |
| Seasonal AR | (1−φB)(1−ΦB¹²) x_t = e_t | φ ~ U(−0.6, 0.6), Φ ~ U(0.4, 0.95) |
| Local level | x_t = μ_t + ε_t, μ_t = μ_{t−1} + η_t | signal-to-noise ratio q = Var η / Var ε, log-uniform on [0.01, 10] |
| MS-AR | 2 regimes, AR(1) per regime | p11, p22 ~ U(0.9, 0.99); φ₁ ~ U(0.5, 0.95), φ₂ ~ U(−0.8, 0); σ₁ = 1, σ₂ ~ U(1.5, 3) |
| GARCH(1,1) | r_t = √h_t z_t | α ~ U(0.03, 0.15), α+β ~ U(0.85, 0.99), ω = 1 − (α+β) (unconditional variance 1); z Gaussian or Student-t(5), standardised |
| Random walk | x_t = x_{t−1} + e_t | no drift |
| White noise | x_t = e_t | none |

**Oracle (Bayes predictor given the same context and the true parameters).**
- AR and seasonal AR: exact AR recursion.
- ARMA(1,1): exact Kalman filter with a stationary prior.
- Local level: exact diffuse Kalman filter.
- MS-AR: Hamilton filter. For h > 1 the forecast is the Monte Carlo mean of 2000 paths.
- GARCH: the conditional mean is exactly 0. The conditional-variance recursion over the context starts from the unconditional variance, which is an approximation at short contexts. The paper's variance results use T ≥ 256.
- In all families, the oracle MSE matched E[Var(y | context)] within Monte Carlo error.

## S2. Models (zero-shot, default settings)
| Model | Checkpoint (HF revision) | Max context | Point forecast | Synthetic pretraining data, as documented in the cited paper |
|---|---|---|---|---|
| Chronos-Bolt-S / -B | amazon/chronos-bolt-small 772f3d25 / -base 5d9f166d | 2048 | median | not verified here |
| Chronos-2 | amazon/chronos-2 29ec3766 | 8192 | median | AR, ETS, KernelSynth (Gaussian-process) base generators, plus multivariatizers |
| TimesFM-2.5 | google/timesfm-2.5-200m-pytorch 1d952420 | set to max(1024, T) | mean | the original TimesFM corpus includes synthetic ARMA(p, q) generators; the 2.5 and 3 corpora are not verified here |
| TimesFM-3 | google/timesfm-3.0-pytorch 43046b85 | 15360 | median (quantile head) | as above |
| TiRex | NX-AI/TiRex 43ebaaff | 2048 | median | Chronos training data, Gaussian-process data inspired by KernelSynth, part of the GIFT-Eval pretraining set |
| Moirai-2.0-S | Salesforce/moirai-2.0-R-small 30f43ff0 | T (passed explicitly) | median | not verified here |
| Toto | Datadog/Toto-Open-Base-1.0 0411ceb2 | no truncation in the inference code; patch size 64 | mean of 256 samples (pre-specified); median (deviation D5) | not verified here |

No model truncates the context at T ≤ 2048. Contexts shorter than TiRex's training context are NaN-padded by its own code.

## S3. Decision rules (pre-specified; see ANALYSIS_PLAN.md)
- **H1, per model × family.** Two one-sided paired-bootstrap tests: RE < 1 at the shortest admissible T, and RE > 1 at T = 2048. H1 holds only if both reject after Holm correction over all 2 × 64 tests. T\* is the first T at which RE rises above 1, found by log-linear interpolation of RE in log T. Its interval is the percentile interval over bootstrap replicates that cross.
- **H2, per model × family.** Ordinary least-squares slope of log ER on log T over T ∈ {256, 512, 1024, 2048}. A one-sided bootstrap test of slope > −0.5, Holm-corrected over 64 tests.
- **H3, per model on the random walk and white noise at T = 2048.** The primary test is a one-sided paired t-test of the realised squared-error difference against the oracle (null: equal MSE). The benchmark is the oracle, which for the driftless random walk is the no-change forecast. The Wilcoxon signed-rank test is a robustness check. Both are Holm-corrected over 16 tests.
- **Intervals.** All intervals are pointwise 95% percentile intervals from the joint paired bootstrap (B = 10⁴; series resampled once per replicate and reused at every T and for every method). They are not simultaneous.

## S4. H2 cells not supported (16 of 64)
| Family | Model(s) | Reason |
|---|---|---|
| White noise | Chronos-Bolt-S, Chronos-Bolt-B, Chronos-2, TimesFM-2.5, TimesFM-3, TiRex, Toto | slopes are at or below −0.5 (TimesFM and Chronos-2 ≈ −1, near-parametric) |
| AR(3) | Chronos-2 | slope −0.40 [−0.48, −0.32]; Holm p = 0.097 |
| Local level | TimesFM-2.5, TimesFM-3 | slopes −0.40 and −0.56 |
| Random walk | TimesFM-3 | slope −0.47 |
| AR(1), AR(3), local level, MS-AR, random walk | Toto (sample mean) | intervals too wide (sample-mean outliers; deviation D5) |

Slope intervals for all cells are in `results/main/H2.csv`. For the seven non-Toto models the median interval width is 0.08 and the maximum is 0.20. The MLE slopes lie between −1.07 and −0.97, and every MLE interval lies within [−1.15, −0.88].

## S5. MS-AR sensitivity with a fallback for failed fits (deviation D7; `results/main/msar_fallback_D7.csv`)
When the MS-AR MLE fails (125, 19 and 1 of 2000 series at T = 16, 32 and 64), it is replaced by the single-regime AR(1) least-squares fit. All 2000 series are then kept. At T = 16, RE ranges from 1.38 [1.16, 1.66] for TimesFM-2.5 to 2.03 [1.72, 2.40] for TiRex across the seven non-Toto models; Toto is 3.73. Every interval lies above 1, and likewise at T = 32 and 64. The confirmatory H1 conclusion is unchanged.

## S6. Volatility (deviation D8; `results/main/garch_interval_width.csv`)
The metric is the mean squared log ratio of each method's own 80% interval width to the oracle's (h = 1).

| Family | T | GARCH MLE | best TSFM | worst TSFM |
|---|---|---|---|---|
| GARCH | 256 / 512 / 1024 / 2048 | 0.0078 / 0.0038 / 0.0019 / 0.0009 | 0.0148 / 0.0137 / 0.0126 / 0.0108 | 0.0293 / 0.0280 / 0.0294 / 0.0237 |
| GARCH-t | 256 / 512 / 1024 / 2048 | 0.0261 / 0.0197 / 0.0169 / 0.0148 | 0.0190 / 0.0168 / 0.0141 / 0.0113 | 0.0370 / 0.0388 / 0.0406 / 0.0345 |

Under Student-t innovations the Gaussian GARCH MLE's intervals are themselves misspecified, and the best TSFM (TimesFM-3 or TiRex) is slightly better. QLIKE and the quantile-implied log-variance error remain in `summary.csv`.

## S7. Further numbers requested in review
- **Toto, pre-specified sample-mean forecast.** At the shortest admissible context RE ranges from 3.40 to 119.6 across families. Table 1 uses the pre-specified sample mean; Figure 1 shows the median variant (D5).
- **Median penalty floor.** The oracle's own predictive median has ER = 0.009 on MS-AR and 0.0003 on GARCH-t, against the oracle mean.
- **Random walk, h = 8, T = 16.**
  - The no-change forecast equals the oracle (ER = 0) and the plug-in MLE, which estimates a drift, has ER = 0.53.
  - The TSFMs range from 0.09 (Moirai) to 0.54 (TimesFM-3), with Toto at 3.4.
  - So the TSFMs beat the drift-estimating MLE but not the no-change forecast.
- **Pilot (estimator agreement, deviation D3).** At h = 1 there were 896 non-oracle cells (N = 100 per family, separate seeds). In 80 of them (9%) the two estimators differed by 2 SE or more, using `src/pilot_check.py`'s definition, which counts cells where both estimates are exactly 0 as disagreements. The disagreeing cells by family: GARCH 21, AR(3) 17, random walk 14, GARCH-t 11, local level 8, AR(1) 4, seasonal AR 3, ARMA 1, MS-AR 1.
- **Freeze and run dates.**
  - The analysis plan was frozen on 2026-11-24, before the first main-run forecast (same day). Its SHA-256 is `dc76521814c3d0847185fc44bbbcedcd746071a0a61cbc9c738d48c3ebc8b57c`.
  - No version-control commit existed at freeze time.
  - First public commit: `7998192` (2026-11-25). The D6 correction is commit `e2d3209`.
- **Integrity check coverage.**
  - An independent AI-run check reviewed the oracle code of all ten families, the analysis, bootstrap and Holm code, and the Jacobian code. It found the MS-AR MLE error (D6).
  - Of the MLEs, only MS-AR was re-derived independently. The others are validated by reproducing the parametric rate (Table 1 slopes, S4).

## S8. Local linearity of the effective filters (review item S18)
The local linear approximation explains most of the forecast variation under random perturbations of the last 32 context values. Across the analysed models, families and T, the median R² ranges from 0.65 to 0.98. TiRex is the exception: its R² is negative in every cell, so no TiRex filter is reported. See `results/jacobian/filters_eps0.001.csv`, column `r2_med`.

## S9. Literature search (systematic review)
- **Databases:** arXiv, OpenAlex and Semantic Scholar, searched on 23 Nov. 2026; scripts are in `review/scripts/`.
- **Records:** 918 after de-duplication.
- **Screening:** two blinded AI reviewer agents screened titles and abstracts, then full texts, with AI adjudication of conflicts. Full-text agreement: κ = 0.906, PABAK = 0.92.
- **Full texts:** 112 of 361 were not retrievable, which is a limitation. 249 full texts were assessed and 174 included.
- **Decision log:** `review/screening/decision_log.csv` (metadata and decisions only).
- **Scope of the novelty claim:** it is limited to the retrieved corpus.

## S10. Random-walk growth test and the H3 comparator (deviation D9)
- **Growth test:** a one-sided joint-bootstrap test of slope > 0 (T ∈ [256, 2048]), Holm-corrected over the 8 pre-specified models (`results/main/rw_growth_holm.csv`).
  - Supported (Holm p = 0.0008): Chronos-Bolt-S, Chronos-Bolt-B, Chronos-2, TiRex, Moirai-2.0-S.
  - Not supported: TimesFM-2.5, TimesFM-3, Toto.
- **Comparator:** against the correctly specified MLE at T = 2048 (H1's pre-specified test of RE > 1, Holm over all H1 tests), every model is less efficient.
  - Random walk: RE 21–1206.
  - White noise: RE 1.5–444.
  - All Holm p ≤ 0.013 (the bootstrap floor).
