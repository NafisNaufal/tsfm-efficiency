# Pre-registration: oracle-relative efficiency of zero-shot time-series foundation models

Frozen on 2026-11-24, before any main-run forecast was produced. Main-run seeds are fixed in `src/run.py`: SEED = 1000 + family index.
Pilot runs (tag `pilot`) use different seeds (offset 7919). They serve only to set computational parameters (batch size, runtime) and to debug code. Pilot results are never pooled with main results, and any pilot result shown in the paper is labelled as such.
Every deviation from this document is logged in the "Deviations" section at the end, with date and reason.

## 1. Questions and hypotheses (confirmatory)
- **H1 (crossover).** For each family f with a correctly specified in-context estimator, and for each TSFM m:
  - define RE_m,f(T) = ER_m,f(T) / ER_MLE,f(T) at horizon h = 1;
  - prediction: RE < 1 at the smallest context length T = 16, and RE > 1 at the largest T available to the model;
  - therefore a finite crossover T\* exists.
- **H2 (saturation).** For TSFMs, the log–log slope of ER versus T over T ∈ [256, 2048] is shallower than −0.5, i.e. it does not reach the parametric rate of −1. This is tested against the MLE slope, which is expected to be ≈ −1.
- **H3 (spurious predictability under the martingale null).** On random_walk and white_noise, for at least one TSFM, ER_m(T = 2048) > 0, assessed with a one-sided paired test against the oracle.

## 2. Design (ADEMP; Morris et al., 2019)
- **Data-generating mechanisms.**
  - Families (see `src/dgp.py`): ar1, ar3, arma11, sar (s = 12), local_level, msar, garch, garch_t, random_walk, white_noise.
  - Parameters are drawn per series from the priors coded in `dgp.py`.
  - N = 2000 series per family.
  - All context lengths T ∈ {16, 32, 64, 128, 256, 512, 1024, 2048} are suffixes of the same series, with the same forecast origin. This makes the design paired across T and across methods.
  - H = 8.
- **Estimands.**
  - ER_m(T,h) = (MSE_m − MSE_oracle) / MSE_oracle, where MSE is computed with the point forecast specified below.
  - RE = ER_m / ER_MLE, and T\*.
  - Slope of log ER versus log T.
  - Distributional quantities:
    - QS = mean pinball loss over the levels 0.1…0.9, reported as a ratio to the oracle QS;
    - coverage of the 80% interval [q0.1, q0.9].
- **Methods.**
  - TSFMs:
    - chronos-bolt-small, chronos-bolt-base, chronos-2: amazon/*, chronos-forecasting 2.3.2;
    - timesfm-2.5 (google/timesfm-2.5-200m-pytorch) and timesfm-3 (google/timesfm-3.0-pytorch), timesfm 3.0.2;
    - tirex (NX-AI/TiRex, tirex-ts 1.4.2);
    - moirai-2-small (Salesforce/moirai-2.0-R-small, uni2ts);
    - toto (Datadog/Toto-Open-Base-1.0, toto-ts).
  - TSFM settings: zero-shot, default inference settings, univariate, no covariates.
  - Point forecast: the model's mean output where one exists (TimesFM-2.5, Toto sample mean), otherwise the median. The difference is flagged in the outputs (`mean_is_median`).
  - Baselines fitted on the same context window:
    - mle: correctly specified class, parameters and mean estimated;
    - autoarima, ets: statsforecast;
    - naive.
  - Oracle: the Bayes predictor given the same context and the true parameters (`dgp.py`).
- **Performance measures and inference.**
  - Monte-Carlo standard errors.
  - Paired bootstrap CIs over series: 10,000 resamples, percentile method, for ER, RE and T\*.
  - T\* is obtained by log-linear interpolation of RE in log T, with a bootstrap CI.
  - H3 is tested with a one-sided paired t-test on squared-error differences, with a Wilcoxon signed-rank test as robustness.
  - Holm correction across all (model × family) tests within each hypothesis.
  - Effect sizes are reported together with the p-values.

## 3. Exclusions and failures
- A method/series pair that fails to produce a forecast (NaN) is excluded only for that method. Failure counts are reported.
- The primary RE uses series where both the TSFM and the MLE produced forecasts.
- Models whose maximum context is below T are not evaluated at that T; this is reported.

## 4. Secondary and exploratory analyses (not confirmatory)
- Effective-filter analysis on a 200-series subset for ar1, ar3 and local_level at T ∈ {64, 256, 1024}:
  - Jacobian of the h = 1 point forecast with respect to the context, by central finite differences with relative step 1e-3;
  - distance to the oracle filter, total filter mass, and local-linearity R².
- GARCH variance calibration: QLIKE of the implied predictive variance, and coverage.
- A real-data anchor. This is exploratory and will be specified before it is run, in a dated amendment.

## Deviations
| Date | Deviation | Reason |
|------|-----------|--------|
| 2026-11-24 | D1: a classical-baseline forecast more than 20 context standard deviations from the context mean, or non-finite, counts as a failed fit (NaN, counted under §3). TSFM outputs are never filtered. | In the pilot (N=100, separate seeds) the 2-regime MarkovAutoregression MLE at T=16 converged to a degenerate solution with ER ≈ 7.6e4. Decided before any main-run forecast was produced. |
| 2026-11-24 | D2: Toto inference uses adaptive batch and sample-batch sizes (num_samples = 256 unchanged). | CUDA grid limit in Toto's space-wise attention for T ≥ 256 (pilot). Affects throughput only, not the forecast distribution. |

| 2026-11-24 | D3: the primary ER estimator becomes E[(f−o)²] / E[Var(y \| context)]. Here o is the oracle conditional mean and Var(y \| ctx) is the oracle predictive variance. RE is the ratio of E[(f−o)²] values on common series. The pre-registered realized-loss estimator (MSE_m − MSE_o)/MSE_o is still reported for every cell, as a robustness check. H3's paired test on realized squared-error differences is unchanged. | The two estimators target the same estimand: o is the conditional mean given the same context, so E[(f−y)²] − E[(o−y)²] = E[(f−o)²]. In the pilot (separate seeds) the realized-loss estimator was too noisy at large T (SE ≈ 2√ER/√N), so RE was undefined whenever ER_MLE ≈ 0. Pilot check: the estimators agree (91% of cells within 2 SE, mean difference −0.006), the direct one has 3.4× narrower CIs, and the MLE reproduces the asymptotic rate ER ≈ (p+1)/T. Decided before any main-run result was inspected. |
| 2026-11-24 | D4: for SAR (s = 12), H1's "smallest context" comparison uses T = 32 instead of T = 16. The GARCH families are excluded from H1/H2 and analysed for variance only (§4). | At T = 16 the SAR MLE (lags 1, 12, 13) has 3 usable equations for 4 parameters, so it is under-identified and not a consistent-estimator reference (150/2000 fits failed). For GARCH the oracle and MLE conditional means are both 0, so ER_MLE ≡ 0 and RE is undefined. Decided before any main-run result was inspected. |
| 2026-11-24 | D5 (**post hoc, after inspecting main results**): Toto is also scored with its predictive median as the point forecast ("toto-median"). The pre-registered "toto" (mean of 256 samples) is still reported in every table. | Toto's 256-sample mean is dominated by single extreme draws from its heavy-tailed predictive mixture. Example: worst garch_t series at T = 64, sample mean −22.9 vs median 0.67 and q10–q90 [−0.49, 2.62]. The top 10 of 2000 series carry 74–98% of Toto's excess risk in several cells (TimesFM-2.5: 7–38%). The sample mean is then an unstable Monte Carlo estimate of the predictive mean, not a property of the model's in-context inference. Because this deviation is post hoc, conclusions about Toto are labelled exploratory. |
| 2026-11-25 | D6 (**bug fix, after main results were inspected**): the MS-AR MLE forecast now uses statsmodels' column-stochastic transition matrix and the mean-form model. One-step: the exact mean Σ_{j,k} P(s_T=j\|data) P(k\|j)[μ_k + φ_k(y_T − μ_j)]. h > 1: 4000-path Monte Carlo from the fitted model. MS-AR MLE results, H1/H2 for MS-AR, Table 1 and Fig. 1 were regenerated. | The Stage 2.5 integrity check (independent agent) found that the earlier code used the transposed transition matrix and treated the regime means μ as intercepts. The earlier code reproduced the saved forecasts exactly, so this was a confirmed implementation error. It restores the pre-specified method (correctly specified MLE); it does not change the method. |
| 2026-11-25 | Timing note for D4 (no new deviation): D4 was logged on 2026-11-24 before main results were inspected. Its consequence for T\* (the SAR crossover search starts at T = 32) was implemented only after the first confirmatory run had produced spurious SAR crossovers at T ≈ 17–25; the corrected run (v2) is the one reported. | Integrity check finding (Stage 2.5). |
| 2026-11-25 | D7 (sensitivity analysis requested by Stage 3 review; declared before running): the MS-AR H1 comparison is repeated on all 2000 series, with failed MS-AR MLE fits replaced by a pre-declared fallback: the single-regime AR(1) least-squares fit with intercept on the same context (the nested model). RE at T = 16/32/64 is reported with a paired bootstrap 95% CI (B = 10⁴). Reported as a sensitivity analysis; the confirmatory H1 result is unchanged. | Reviewers noted that excluding failed fits (125/2000 at T = 16) conditions the comparison on the comparator's success. |
| 2026-11-25 | D8 (post hoc, requested by Stage 3 review): GARCH-family volatility comparison in the paper uses each method's own 80% interval width relative to the oracle's 80% interval width (mean squared log ratio, h = 1). This replaces the variance read off TSFM quantiles under a normality assumption, which is biased for heavy-tailed predictive laws (applied to the oracle's own quantiles it already gives 0.053 on GARCH-t). QLIKE and the quantile-implied log-variance error stay in the released tables. | The reviewers found the earlier comparison not like-for-like. Under the new metric the GARCH-t conclusion reverses (best TSFM 0.011 vs Gaussian GARCH MLE 0.015), and the paper reports this. |
| 2026-11-25 | D9 (post hoc, requested in re-review): the claim that error grows with context on the random walk is tested by a one-sided joint-bootstrap test of slope > 0 (T ∈ [256, 2048]), Holm-corrected over the 8 pre-specified models (`src/rw_growth.py`, `results/main/rw_growth_holm.csv`). H3's informative comparator is the correctly specified MLE, using the pre-specified H1 test of RE > 1 at T = 2048. The pre-specified oracle-null t-test is still reported. | The re-review noted that a null of ER = 0 against the oracle is uninformative (any non-oracle forecaster has E(f−o)² > 0), and that the growth claim lacked a multiplicity adjustment. |
| 2026-11-25 | D10 (bug fix, found by the Stage 4.5 final integrity check): the MS-AR oracle one-step mean is the exact Hamilton-filter mean, replacing a 2000-path Monte Carlo mean. `run.py` had called `oracle_point` with H = 8, so its exact h = 1 branch never ran. MS-AR oracle outputs, all MS-AR analyses and the tables and figures that depend on them are regenerated. The pre-fix files are archived. | The paper described the oracle as an exact filter. The Monte Carlo error added ≈ 0.0005 to every MS-AR ER, which biases efficiency ratios conservatively; no verdict changed in the integrity agent's exact recomputation. |
