"""Exploratory effective-filter analysis (PREREGISTRATION.md §4).

For a subset of S series (the first S of the main data), family f in {ar1, ar3, local_level}, T in {64, 256, 1024}:
  * Jacobian of the h=1 point forecast w.r.t. the last L context values, by central finite differences with step
    eps * sd(context) (eps = 1e-3 pre-registered; 1e-2 run as a stability check)
  * level-shift response: d f / d delta for c + delta * 1 (total filter mass incl. lags beyond L)
  * local-linearity R^2: K random perturbations (N(0, (0.3 sd)^2) on the last L values); actual vs Jacobian-predicted change
The same perturbed contexts are given to every TSFM, the oracle and the MLE. Toto is excluded (sample-mean forecast
is not differentiable by finite differences at 256 samples).

python jacobian.py build|tsfm|cpu|analyze [--eps 1e-3]
"""
import argparse, os, subprocess, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from run import ROOT, TSFMS, T_GRID
from dgp import oracle_point
from baselines import fit_predict

S, L, K = 200, 32, 8
FAMS = ["ar1", "ar3", "local_level"]
TS = [64, 256, 1024]
MODELS = [m for m in TSFMS if m != "toto"]


def jdir(fam, T, eps): d = f"{ROOT}/results/jacobian/eps{eps:g}/{fam}/T{T}"; os.makedirs(d, exist_ok=True); return d


def build(eps):
    rng = np.random.default_rng(777)
    for fam in FAMS:
        x = np.load(f"{ROOT}/results/main/{fam}/data.npz", allow_pickle=True)["x"][:S]; Tm = max(T_GRID)
        for T in TS:
            c = x[:, Tm - T:Tm]; sd = c.std(1, keepdims=True); blocks = [c]
            for j in range(L):          # lag j+1 = position T-1-j
                for s in (+1, -1):
                    p = c.copy(); p[:, T - 1 - j] += s * eps * sd[:, 0]; blocks.append(p)
            for s in (+1, -1): blocks.append(c + s * eps * sd)
            dz = rng.standard_normal((K, S, L)) * 0.3
            for k in range(K):
                p = c.copy(); p[:, T - L:] += dz[k][:, ::-1] * sd; blocks.append(p)   # dz[k][:, j] perturbs lag j+1
            arr = np.stack(blocks, 1)  # (S, 1 + 2L + 2 + K, T)
            d = jdir(fam, T, eps); np.save(f"{d}/ctx.npy", arr.reshape(-1, T).astype(np.float32)); np.save(f"{d}/dz.npy", dz)
            np.save(f"{d}/ctx64.npy", arr)


def tsfm(eps, models, batch):
    for fam in FAMS:
        for T in TS:
            d = jdir(fam, T, eps)
            for m in models:
                f = f"{d}/{m}.npz"
                if os.path.exists(f): continue
                for attempt in range(8):   # shared GPU: wait and retry on CUDA OOM (atomic outputs)
                    r = subprocess.run([f"{ROOT}/{TSFMS[m]}/bin/python", f"{ROOT}/src/run_tsfm.py", "--model", m, "--ctx", f"{d}/ctx.npy",
                                        "--H", "1", "--out", f, "--batch", str(batch)], capture_output=True, text=True)
                    if r.returncode == 0 or "out of memory" not in r.stderr.lower(): break
                    print(fam, T, m, f"OOM, retry {attempt + 1} in 120s", flush=True); import time; time.sleep(120)
                print(fam, T, m, (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else "FAILED " + r.stderr[-300:], flush=True)


def _one(args):
    meth, fam, c, prm = args
    return oracle_point(fam, c, prm, 1)[0] if meth == "oracle" else fit_predict("mle", fam, c, 1, prm)[0][0]


def cpu(eps, procs=60):
    from multiprocessing import Pool
    with Pool(procs) as pool:
        for fam in FAMS:
            prm = list(np.load(f"{ROOT}/results/main/{fam}/data.npz", allow_pickle=True)["params"])[:S]
            for T in TS:
                d = jdir(fam, T, eps); arr = np.load(f"{d}/ctx64.npy"); n = arr.shape[1]
                for meth in ("oracle", "mle"):
                    f = f"{d}/{meth}.npz"
                    if os.path.exists(f): continue
                    out = pool.map(_one, [(meth, fam, arr[i, b], prm[i]) for i in range(S) for b in range(n)], chunksize=16)
                    np.savez(f, mean=np.asarray(out, float).reshape(-1, 1)); print(fam, T, meth, "done", flush=True)


def analyze(eps):
    rows = []
    for fam in FAMS:
        for T in TS:
            d = jdir(fam, T, eps)
            if not os.path.exists(f"{d}/oracle.npz"): continue
            arr = np.load(f"{d}/ctx64.npy"); dz = np.load(f"{d}/dz.npy"); n = arr.shape[1]
            sd = arr[:, 0].std(1); h = eps * sd
            def jac(meth):
                z = np.load(f"{d}/{meth}.npz", allow_pickle=True)["mean"].astype(np.float64).reshape(S, n)
                J = np.stack([(z[:, 1 + 2 * j] - z[:, 2 + 2 * j]) / (2 * h) for j in range(L)], 1)   # (S, L) lag 1..L
                mass = (z[:, 1 + 2 * L] - z[:, 2 + 2 * L]) / (2 * h)
                act = z[:, 3 + 2 * L:] - z[:, :1]                                                     # (S, K)
                pred = np.einsum("skl,sl->sk", dz.transpose(1, 0, 2), J) * sd[:, None]
                r2 = 1 - ((act - pred) ** 2).sum(1) / np.maximum(((act - act.mean(1, keepdims=True)) ** 2).sum(1), 1e-12)
                return J, mass, r2
            Jo, mo, _ = jac("oracle")
            for meth in ["oracle", "mle"] + MODELS:
                if not os.path.exists(f"{d}/{meth}.npz"): continue
                J, mass, r2 = jac(meth)
                dist = np.linalg.norm(J - Jo, axis=1) / np.maximum(np.linalg.norm(Jo, axis=1), 1e-8)
                rows.append(dict(family=fam, T=T, method=meth, eps=eps, filt_dist_med=np.median(dist), filt_dist_mean=dist.mean(),
                                 w1_mean=J[:, 0].mean(), w1_oracle=Jo[:, 0].mean(), mass_mean=mass.mean(), mass_oracle=mo.mean(),
                                 r2_med=np.median(r2), tail_mass=np.mean(mass - J.sum(1))))
                np.save(f"{d}/J_{meth}.npy", J)
    R = pd.DataFrame(rows); R.to_csv(f"{ROOT}/results/jacobian/filters_eps{eps:g}.csv", index=False); return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--models", nargs="*", default=MODELS); ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    if a.stage == "build": build(a.eps)
    elif a.stage == "tsfm": tsfm(a.eps, a.models, a.batch)
    elif a.stage == "cpu": cpu(a.eps)
    elif a.stage == "analyze":
        pd.set_option("display.width", 250); print(analyze(a.eps).round(3).to_string())
