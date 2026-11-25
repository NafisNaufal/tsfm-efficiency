"""Deviation D7: MS-AR H1 sensitivity with a pre-declared fallback for failed MS-AR MLE fits (AR(1) OLS with intercept)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from run import ROOT
from baselines import _ols_ar
from figures import MODELS
P = [m for m in MODELS if m != "toto-median"]
d = f"{ROOT}/results/main/msar"; x = np.load(f"{d}/data.npz", allow_pickle=True)["x"]; Tm = 2048
rng = np.random.default_rng(20261125); B = 10000; rows = []
for T in [16, 32, 64]:
    o = np.load(f"{d}/T{T}/oracle.npz")["mean"][:, 0]
    mle = np.load(f"{d}/T{T}/mle.npz")["mean"][:, 0].copy(); fail = ~np.isfinite(mle)
    for i in np.where(fail)[0]: mle[i] = _ols_ar(x[i, Tm - T:Tm], [1], 1)[0][0]
    dl = (mle - o) ** 2; idx = rng.integers(0, len(o), (B, len(o)))
    for m in P:
        f = np.load(f"{d}/T{T}/{m}.npz", allow_pickle=True)["mean"][:, 0].astype(float); dm = (f - o) ** 2
        rb = dm[idx].mean(1) / dl[idx].mean(1)
        rows.append(dict(T=T, model=m, n=len(o), n_fallback=int(fail.sum()), RE=dm.mean() / dl.mean(),
                         RE_lo=np.percentile(rb, 2.5), RE_hi=np.percentile(rb, 97.5)))
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/main/msar_fallback_D7.csv", index=False); print(R.round(3).to_string())
