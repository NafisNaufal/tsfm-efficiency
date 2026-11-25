"""Like-for-like volatility comparison for GARCH families (review item S14): squared log ratio of each method's own
80% interval width to the oracle's 80% interval width, h=1. TSFMs: q0.9-q0.1; MLE: Gaussian 80% interval from its sd."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from run import ROOT
from figures import MODELS
P = [m for m in MODELS if m != "toto-median"]; Z = 1.2815515655446004; rows = []
for fam in ["garch", "garch_t"]:
    for T in [256, 512, 1024, 2048]:
        d = f"{ROOT}/results/main/{fam}/T{T}"; oq = np.load(f"{d}/oracle.npz")["q"][:, 0]; wo = oq[:, 8] - oq[:, 0]
        for m in ["mle"] + P:
            z = np.load(f"{d}/{m}.npz", allow_pickle=True)
            w = 2 * Z * z["sd"][:, 0] if m == "mle" else z["q"][:, 0, 8] - z["q"][:, 0, 0]
            ok = np.isfinite(w) & (w > 0)
            rows.append(dict(family=fam, T=T, method=m, width_logerr=float(np.mean(np.log(w[ok] / wo[ok]) ** 2)), n=int(ok.sum())))
R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/main/garch_interval_width.csv", index=False)
print(R.pivot_table(index=["family", "method"], columns="T", values="width_logerr").round(4).to_string())
