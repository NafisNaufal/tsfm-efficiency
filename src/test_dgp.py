"""Sanity checks of the oracle: oracle one-step MSE must match its own predictive variance (calibration)."""
import numpy as np, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from dgp import GENERATORS, oracle_point, oracle_gaussian, oracle_sampler

rng = np.random.default_rng(123)
N, T, H = 800, 256, 4
for fam, gen in GENERATORS.items():
    b = gen(N, T + H, rng)
    err, pv = [], []
    for i in range(N):
        ctx = b.x[i, :T]; y = b.x[i, T:T + H]
        m = oracle_point(fam, ctx, b.params[i], H, rng=rng, k=2000)
        err.append((y - m) ** 2)
        if fam in ("msar", "garch", "garch_t"):
            s = oracle_sampler(fam, ctx, b.params[i], H, 2000, rng); pv.append(s.var(0))
        else:
            pv.append(oracle_gaussian(fam, ctx, b.params[i], H)[1])
    err, pv = np.array(err).mean(0), np.array(pv).mean(0)
    se = np.array(err) * np.sqrt(2.0 / N) * 3
    ok = np.all(np.abs(err - pv) < np.maximum(se, 0.1 * pv))
    print(f"{fam:12s} MSE {np.round(err,3)} predVar {np.round(pv,3)} {'OK' if ok else 'CHECK'}")
