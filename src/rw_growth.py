"""Review round 2 (R5): one-sided bootstrap test of slope > 0 (log ER on log T, T in [256, 2048]) on the random walk,
Holm-corrected over the 8 pre-specified models; same joint paired bootstrap and seed scheme as hypotheses.py."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from hypotheses import load_dev, slope, holm, T_GRID
from figures import MODELS
P = [m for m in MODELS if m != "toto-median"]; B = 10000
rng = np.random.default_rng(20261126); rows = []
_, _, ovar = load_dev("main", "random_walk", "mle"); N = ovar.shape[1]; idx = rng.integers(0, N, (B, N))
for m in P:
    dev, _, _ = load_dev("main", "random_walk", m)
    er_b = np.nanmean(dev[:, idx], 2) / np.nanmean(ovar[:, idx], 2)
    sl_b = np.array([slope(er_b[:, b]) for b in range(B)])
    rows.append(dict(model=m, slope=slope(np.nanmean(dev, 1) / np.nanmean(ovar, 1)), p_one_sided=max(np.mean(sl_b <= 0), 1 / B)))
R = pd.DataFrame(rows); R["p_holm"] = holm(R.p_one_sided); R["growth_supported"] = R.p_holm < .05
R.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/main/rw_growth_holm.csv"), index=False)
print(R.round(4).to_string())
