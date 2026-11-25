"""Pre-registered analysis (PREREGISTRATION.md §2). python analyze.py --tag main [--B 10000]
Writes results/<tag>/summary.csv (per family x T x method x h) and results/<tag>/re.csv (RE vs MLE, h=1).
Paired design: every method is scored on the same series, same origin, same target.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from run import ROOT, T_GRID, H, QL, FAMILIES

Z80 = 1.2815515655446004
VOL = ("garch", "garch_t")


def pinball(q, y):  # q (N,H,9), y (N,H) -> (N,H) mean over levels
    d = y[..., None] - q
    return np.mean(np.maximum(QL * d, (QL - 1) * d), axis=-1)


def load_method(d, m):
    z = np.load(f"{d}/{m}.npz", allow_pickle=True)
    mean = z["mean"].astype(np.float64)
    q = z["q"].astype(np.float64) if "q" in z.files else None
    sd = z["sd"].astype(np.float64) if "sd" in z.files else None
    if sd is not None and np.all(np.isnan(sd)): sd = None
    if sd is None and q is not None: sd = (q[..., 8] - q[..., 0]) / (2 * Z80)   # implied by 80% interval
    return mean, q, sd


def boot_ratio(num, den, B, rng):  # ratio of means of paired per-series arrays, percentile CI
    n = len(num); idx = rng.integers(0, n, (B, n))
    r = num[idx].mean(1) / den[idx].mean(1)
    return num.mean() / den.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5), r


def main(tag, B):
    rng = np.random.default_rng(20261124)
    rows, rerows, boots = [], [], {}
    for fam in FAMILIES:
        fd = f"{ROOT}/results/{tag}/{fam}"
        if not os.path.exists(f"{fd}/data.npz"): continue
        x = np.load(f"{fd}/data.npz", allow_pickle=True)["x"]; Tm = max(T_GRID); y = x[:, Tm:Tm + H]
        for T in T_GRID:
            d = f"{fd}/T{T}"
            if not os.path.exists(f"{d}/oracle.npz"): continue
            om, oq, osd = load_method(d, "oracle")
            se_o = (om - y) ** 2; ql_o = y ** 2 / osd ** 2 + np.log(osd ** 2)
            methods = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{d}/*.npz") if ".tmp" not in f)
            ovar = osd ** 2   # oracle predictive variance = E[(o - y)^2 | context]
            for m in methods:
                mm, mq, msd = load_method(d, m)
                v = np.all(np.isfinite(mm), axis=1)
                for h in range(H):
                    Bh = B if h == 0 else min(B, 1000)
                    se = (mm[v, h] - y[v, h]) ** 2; so = se_o[v, h]
                    dev = (mm[v, h] - om[v, h]) ** 2          # (f - o)^2
                    row = dict(family=fam, T=T, method=m, h=h + 1, n=int(v.sum()), n_fail=int((~v).sum()),
                               mse=se.mean(), mse_oracle=so.mean())
                    if m != "oracle":
                        # primary (deviation D3): ER = E[(f-o)^2] / E[Var(y|ctx)], same estimand, Rao-Blackwellised
                        er, lo, hi, _ = boot_ratio(dev, ovar[v, h], Bh, rng)
                        # pre-registered realized-loss estimator, kept as robustness check
                        er2, lo2, hi2, _ = boot_ratio(se - so, so, Bh, rng)
                        row.update(ER=er, ER_lo=lo, ER_hi=hi, ER_real=er2, ER_real_lo=lo2, ER_real_hi=hi2)
                        if h == 0: boots[(fam, T, m)] = (dev, se - so, v)
                    else:
                        row.update(ER=0., ER_lo=0., ER_hi=0., ER_real=0., ER_real_lo=0., ER_real_hi=0.)
                    if mq is not None and oq is not None:
                        ps, po = pinball(mq[v], y[v])[:, h], pinball(oq[v], y[v])[:, h]
                        row.update(QS_ratio=ps.mean() / po.mean(),
                                   cov80=np.mean((y[v, h] >= mq[v, h, 0]) & (y[v, h] <= mq[v, h, 8])))
                    if fam in VOL and msd is not None:
                        s2 = np.maximum(msd[v, h], 1e-12) ** 2
                        ql = y[v, h] ** 2 / s2 + np.log(s2)
                        row.update(QLIKE_excess=(ql - ql_o[v, h]).mean(),
                                   logvar_err=np.mean((np.log(s2) - np.log(ovar[v, h])) ** 2))
                    rows.append(row)
            # RE vs MLE at h=1 on series where both produced forecasts (primary RE, prereg §3)
            if (fam, T, "mle") in boots:
                for m in methods:
                    if m in ("oracle", "mle") or (fam, T, m) not in boots: continue
                    dm, rm, v_m = boots[(fam, T, m)]; dl, rl, v_l = boots[(fam, T, "mle")]
                    both = v_m & v_l; a = dm[both[v_m]]; b = dl[both[v_l]]
                    re, lo, hi, _ = boot_ratio(a, b, B, rng)
                    rerows.append(dict(family=fam, T=T, method=m, n=int(both.sum()), RE=re, RE_lo=lo, RE_hi=hi, logRE=np.log(re)))
    S = pd.DataFrame(rows); R = pd.DataFrame(rerows)
    S.to_csv(f"{ROOT}/results/{tag}/summary.csv", index=False); R.to_csv(f"{ROOT}/results/{tag}/re.csv", index=False)
    return S, R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args(); S, R = main(a.tag, a.B)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(S[S.h == 1].pivot_table(index=["family", "method"], columns="T", values="ER").round(3))
