"""Reproduces the estimator-agreement numbers quoted in the paper (deviation D3) from results/pilot/summary.csv:
share of (family, T, method) cells at h=1 where |ER_direct - ER_realized| < 2 SE(realized), mean difference, and the
ratio of median 95% CI widths.  Also prints the oracle quantile-implied log-variance error for the GARCH families."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = pd.read_csv(f"{ROOT}/results/pilot/summary.csv"); s = S[(S.h == 1) & (S.method != "oracle")]
se = (s.ER_real_hi - s.ER_real_lo) / 3.92; d = s.ER - s.ER_real
print(f"cells {len(s)}  within 2 SE {np.mean(np.abs(d) < 2 * se):.3f}  mean diff {d.mean():.4f}  "
      f"CI width ratio {(s.ER_real_hi - s.ER_real_lo).median() / (s.ER_hi - s.ER_lo).median():.2f}")
Z = 1.2815515655446004
for fam in ["garch", "garch_t"]:
    o = np.load(f"{ROOT}/results/main/{fam}/T2048/oracle.npz"); q = o["q"][:, 0]; v = o["sd"][:, 0] ** 2
    print(fam, "oracle quantile-implied log-variance error", round(float(np.mean((np.log(((q[:, 8] - q[:, 0]) / (2 * Z)) ** 2) - np.log(v)) ** 2)), 4))
