# Effective-filter analysis: exploratory notes (prereg §4; not confirmatory)

Validity checks:
- **Oracle.** The finite-difference Jacobian is exact: on AR(1) the lag-1 weight equals φ and all other lags are 0, at both ε.
- **MLE.** It converges to the oracle filter, with distance 0.44, 0.14 and 0.06 at T = 64, 256 and 1024 on AR(1).
- **Step-size stability.** Excluding TiRex, the filter metrics at ε = 1e-3 and 1e-2 agree (corr 1.000, max |Δ filter distance| 0.008, all 81 cells).
- **TiRex.**
  - It is deterministic: repeat calls differ by exactly 0.
  - Its finite-difference lag-1 derivative does *not* converge as ε shrinks. The median is 0.33, 0.14, 0.08 and 0.05 and the IQR is 5.0, 0.68, 0.63 and 0.60 for ε = 1e-4, 1e-3, 1e-2 and 1e-1 (CPU check on 32 AR(1) series, T = 256).
  - Its local-linearity R² is strongly negative in every cell.
  - Conclusion: TiRex's forecast is non-smooth in the context at fine scales, so a local linear filter is not defined for it. It is **excluded from the Jacobian filter metrics**, and this is reported as a finding with its reason. It is not silently dropped.

Observations (ε = 1e-3, 200 series per cell, all 81 cells complete for each ε; final as of 2026-11-24):
- **Shift response.** Every TSFM, and the MLE with its intercept, has a level-shift response of exactly 1.000: the forecasts are location-equivariant. The zero-mean oracle has Σφ. So part of every TSFM's filter mass sits on "estimate the mean", like an intercept-fitting estimator.
- **Distance to the oracle filter.** It falls with T for every smooth TSFM, but much more slowly than for the MLE. On AR(3) at T = 1024 it is 0.32–0.45 for TimesFM-2.5/3 and Chronos-2, 0.73 for Moirai-2-S and 0.94–0.97 for Chronos-Bolt S/B, against 0.05 for the MLE.
- **Chronos-Bolt S/B.** Their filters barely change with T: distance 0.97 → 0.97 (S) and 0.98 → 0.94 (B) on AR(3) from T = 64 to 1024. Their lag-1 weight stays near 0.28 against an oracle mean of 0.07. This is consistent with a fixed smoothing prior and little in-context estimation, and with the pilot φ < 0 observation. To be confirmed with per-φ stratification on the main data.
- **Local level.** All smooth TSFMs come within 0.24–0.36 of the Kalman filter at T = 1024, against 0.05 for the MLE. The lag-1 weight is close to the oracle (0.49–0.56 vs 0.47).
