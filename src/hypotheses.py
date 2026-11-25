"""Confirmatory tests H1-H3 (PREREGISTRATION.md §1-2, with deviation D3 for the ER estimator).

Joint paired bootstrap: series are resampled ONCE per replicate and the same indices are used for every T and method
(all contexts are suffixes of the same series with the same forecast origin), so CIs for T* and slopes are valid.
h = 1 throughout.  python hypotheses.py --tag main --B 10000
Families for H1/H2: those whose MLE point forecast is not identical to the oracle's (GARCH families excluded: both
oracle and MLE predict a zero conditional mean, so ER_MLE = 0 and RE is undefined; they enter the variance analysis).
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from run import ROOT, T_GRID, FAMILIES
from figures import MODELS

H1H2_FAMS = [f for f in FAMILIES if f not in ("garch", "garch_t")]
SLOPE_T = [256, 512, 1024, 2048]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.
    for k, i in enumerate(o):
        run = max(run, min(1., (m - k) * p[i])); adj[i] = run
    return adj


def load_dev(tag, fam, meth):
    """(T, N) arrays of (f - o)^2 and realized se - se_o at h=1 (NaN where the method failed), and oracle variance."""
    x = np.load(f"{ROOT}/results/{tag}/{fam}/data.npz", allow_pickle=True)["x"]; y = x[:, max(T_GRID)]
    dev, real, ovar = [], [], []
    for T in T_GRID:
        d = f"{ROOT}/results/{tag}/{fam}/T{T}"
        o = np.load(f"{d}/oracle.npz"); om = o["mean"][:, 0]; ov = o["sd"][:, 0] ** 2
        f = np.load(f"{d}/{meth}.npz", allow_pickle=True)["mean"][:, 0].astype(np.float64)
        dev.append((f - om) ** 2); real.append((f - y) ** 2 - (om - y) ** 2); ovar.append(ov)
    return np.array(dev), np.array(real), np.array(ovar)


def crossover(re, start=0):  # first crossing of RE through 1 at or after T_GRID[start], log-linear in log T; nan if none
    lr = np.log(re); lt = np.log(T_GRID)
    for k in range(start, len(lr) - 1):
        if np.isfinite(lr[k]) and np.isfinite(lr[k + 1]) and lr[k] < 0 <= lr[k + 1]:
            return float(np.exp(lt[k] + (0 - lr[k]) * (lt[k + 1] - lt[k]) / (lr[k + 1] - lr[k])))
    return np.nan


def slope(er):
    idx = [T_GRID.index(t) for t in SLOPE_T]; return np.polyfit(np.log(SLOPE_T), np.log(er[idx]), 1)[0]


def main(tag, B, models=MODELS, suffix=""):
    rng = np.random.default_rng(20261125)
    h1, h2, h3 = [], [], []
    for fam in FAMILIES:
        devM, realM, ovar = load_dev(tag, fam, "mle")
        N = devM.shape[1]; idx = rng.integers(0, N, (B, N))
        for m in models:
            try: dev, real, _ = load_dev(tag, fam, m)
            except FileNotFoundError: print("missing", fam, m); continue
            if fam in H1H2_FAMS:
                ok = np.isfinite(devM)                      # common series per T (prereg §3)
                num = np.where(ok, dev, np.nan); den = np.where(ok, devM, np.nan)
                re = np.nanmean(num, 1) / np.nanmean(den, 1)
                er = np.nanmean(dev, 1) / np.nanmean(ovar, 1); erM = np.nanmean(devM, 1) / np.nanmean(ovar, 1)
                re_b = np.nanmean(num[:, idx], 2) / np.nanmean(den[:, idx], 2)              # (T, B)
                er_b = np.nanmean(dev[:, idx], 2) / np.nanmean(ovar[:, idx], 2)
                erM_b = np.nanmean(devM[:, idx], 2) / np.nanmean(ovar[:, idx], 2)
                i16, iT = 0, len(T_GRID) - 1
                if fam == "sar": i16 = 1   # SAR MLE is under-identified at T=16 (3 equations, 4 parameters): use T=32
                p_lo = np.mean(re_b[i16] >= 1); p_hi = np.mean(re_b[iT] <= 1)
                ts = np.array([crossover(re_b[:, b], i16) for b in range(B)])
                h1.append(dict(family=fam, model=m, T_low=T_GRID[i16], RE_low=re[i16], RE_low_lo=np.percentile(re_b[i16], 2.5),
                               RE_low_hi=np.percentile(re_b[i16], 97.5), RE_2048=re[iT], RE_2048_lo=np.percentile(re_b[iT], 2.5),
                               RE_2048_hi=np.percentile(re_b[iT], 97.5), p_low=max(p_lo, 1 / B), p_high=max(p_hi, 1 / B),
                               Tstar=crossover(re, i16), Tstar_lo=np.nanpercentile(ts, 2.5) if np.isfinite(ts).any() else np.nan,
                               Tstar_hi=np.nanpercentile(ts, 97.5) if np.isfinite(ts).any() else np.nan,
                               frac_boot_with_crossing=np.isfinite(ts).mean()))
                sl = slope(er); slM = slope(erM)
                sl_b = np.array([slope(er_b[:, b]) for b in range(B)]); slM_b = np.array([slope(erM_b[:, b]) for b in range(B)])
                h2.append(dict(family=fam, model=m, slope=sl, slope_lo=np.percentile(sl_b, 2.5), slope_hi=np.percentile(sl_b, 97.5),
                               slope_mle=slM, slope_mle_lo=np.percentile(slM_b, 2.5), slope_mle_hi=np.percentile(slM_b, 97.5),
                               diff=sl - slM, diff_lo=np.percentile(sl_b - slM_b, 2.5), diff_hi=np.percentile(sl_b - slM_b, 97.5),
                               p_shallower=max(np.mean(sl_b <= -0.5), 1 / B)))
            if fam in ("random_walk", "white_noise"):
                r = real[-1]; r = r[np.isfinite(r)]
                t = stats.ttest_1samp(r, 0, alternative="greater"); w = stats.wilcoxon(r, alternative="greater")
                h3.append(dict(family=fam, model=m, T=2048, mean_excess_sq=r.mean(), ER_direct=np.mean(dev[-1]) / np.mean(ovar[-1]),
                               t=t.statistic, p_t=t.pvalue, p_wilcoxon=w.pvalue, n=len(r)))
    H1, H2, H3 = pd.DataFrame(h1), pd.DataFrame(h2), pd.DataFrame(h3)
    # H1 holds for a (model, family) if BOTH one-sided tests reject; Holm over all 2 x cells tests
    pa = holm(np.r_[H1.p_low, H1.p_high]); n = len(H1)
    H1["p_low_holm"], H1["p_high_holm"] = pa[:n], pa[n:]
    H1["H1_supported"] = (H1.p_low_holm < .05) & (H1.p_high_holm < .05)
    H2["p_holm"] = holm(H2.p_shallower); H2["H2_supported"] = H2.p_holm < .05
    H3["p_t_holm"] = holm(H3.p_t); H3["p_w_holm"] = holm(H3.p_wilcoxon); H3["H3_supported"] = H3.p_t_holm < .05
    out = f"{ROOT}/results/{tag}"
    H1.to_csv(f"{out}/H1{suffix}.csv", index=False); H2.to_csv(f"{out}/H2{suffix}.csv", index=False); H3.to_csv(f"{out}/H3{suffix}.csv", index=False)
    return H1, H2, H3


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--exploratory-toto-median", action="store_true",
                    help="D5 (post hoc): Toto scored by its median; separate Holm family, outputs *_toto_median.csv")
    a = ap.parse_args()
    PRIMARY = [m for m in MODELS if m != "toto-median"]          # the 8 pre-registered models (Holm family)
    H1, H2, H3 = (main(a.tag, a.B, ["toto-median"], "_toto_median") if a.exploratory_toto_median else main(a.tag, a.B, PRIMARY))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(H1.round(3).to_string()); print(H2.round(3).to_string()); print(H3.round(4).to_string())
