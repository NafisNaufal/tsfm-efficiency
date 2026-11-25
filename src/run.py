"""Main experiment driver (pre-registered design; see experiments/PREREGISTRATION.md).

python run.py --stage {data,oracle,baselines,tsfm,all} --tag main --N 2000 [--families ...] [--models ...]
Outputs: results/<tag>/<family>/data.npz, T<T>/<method>.npz with mean (N,H), q (N,H,9) (if available), sd (N,H).
"""
import argparse, json, os, subprocess, sys, time, hashlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from dgp import GENERATORS, oracle_point, oracle_gaussian, oracle_sampler
from baselines import fit_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048]
H = 8
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
FAMILIES = ["ar1", "ar3", "arma11", "sar", "local_level", "msar", "garch", "garch_t", "random_walk", "white_noise"]
TSFMS = {  # name -> python env
    "chronos-bolt-small": "env", "chronos-bolt-base": "env", "chronos-2": "env", "timesfm-2.5": "env",
    "timesfm-3": "env", "tirex": "env", "moirai-2-small": "env_moirai", "toto": "env_toto"}
BASELINES = ["mle", "autoarima", "ets", "naive"]
SEED = {f: 1000 + i for i, f in enumerate(FAMILIES)}


def fam_dir(tag, fam): d = f"{ROOT}/results/{tag}/{fam}"; os.makedirs(d, exist_ok=True); return d


def stage_data(tag, fam, N):
    rng = np.random.default_rng(SEED[fam] + (0 if tag == "main" else 7919))
    b = GENERATORS[fam](N, max(T_GRID) + H, rng)
    np.savez(f"{fam_dir(tag, fam)}/data.npz", x=b.x.astype(np.float64), params=np.array(b.params, dtype=object), family=fam)


def load(tag, fam):
    d = np.load(f"{fam_dir(tag, fam)}/data.npz", allow_pickle=True); return d["x"], list(d["params"])


def _oracle_one(args):
    fam, ctx, prm, seed = args
    rng = np.random.default_rng(seed)
    mean = oracle_point(fam, ctx, prm, H, rng=rng, k=2000)
    s = oracle_sampler(fam, ctx, prm, H, 4000, rng)
    return mean, np.quantile(s, QL, axis=0).T, s.std(0)


def _base_one(args):
    meth, fam, ctx, prm = args
    m, sd = fit_predict(meth, fam, ctx, H, prm)
    sd = np.full(H, np.nan) if sd is None else sd
    return m, sd


def stage_cpu(tag, fam, methods, procs):
    x, params = load(tag, fam); Tm = max(T_GRID)
    with Pool(procs) as pool:
        for T in T_GRID:
            out = f"{fam_dir(tag, fam)}/T{T}"; os.makedirs(out, exist_ok=True)
            ctxs = [x[i, Tm - T:Tm] for i in range(len(x))]
            for meth in methods:
                f = f"{out}/{meth}.npz"
                if os.path.exists(f): continue
                t0 = time.time()
                if meth == "oracle":
                    r = pool.map(_oracle_one, [(fam, c, p, 17 * i + T) for i, (c, p) in enumerate(zip(ctxs, params))], chunksize=8)
                    np.savez(f, mean=np.stack([a for a, _, _ in r]), q=np.stack([b for _, b, _ in r]), sd=np.stack([c for _, _, c in r]))
                else:
                    r = pool.map(_base_one, [(meth, fam, c, p) for c, p in zip(ctxs, params)], chunksize=4)
                    np.savez(f, mean=np.stack([a for a, _ in r]), sd=np.stack([b for _, b in r]))
                print(f"{fam} T={T} {meth}: {time.time() - t0:.0f}s", flush=True)


def stage_tsfm(tag, fam, models, batch):
    x, _ = load(tag, fam); Tm = max(T_GRID)
    for T in T_GRID:
        out = f"{fam_dir(tag, fam)}/T{T}"; os.makedirs(out, exist_ok=True)
        cpath = f"{out}/ctx.npy"
        if not os.path.exists(cpath): np.save(cpath, x[:, Tm - T:Tm].astype(np.float32))
        for m in models:
            f = f"{out}/{m}.npz"
            if os.path.exists(f): continue
            py = f"{ROOT}/{TSFMS[m]}/bin/python"
            for attempt in range(8):   # shared GPU: on CUDA OOM wait and retry (output is atomic, so retries are safe)
                r = subprocess.run([py, f"{ROOT}/src/run_tsfm.py", "--model", m, "--ctx", cpath, "--H", str(H), "--out", f,
                                    "--batch", str(batch)], capture_output=True, text=True)
                if r.returncode == 0 or "out of memory" not in r.stderr.lower(): break
                print(f"{fam} T={T} {m}: OOM, retry {attempt + 1} in 120s", flush=True); time.sleep(120)
            msg = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else "FAILED: " + r.stderr.strip().splitlines()[-1][:200]
            print(f"{fam} T={T} {m}: {msg}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all"); ap.add_argument("--tag", default="main"); ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--families", nargs="*", default=FAMILIES); ap.add_argument("--models", nargs="*", default=list(TSFMS))
    ap.add_argument("--baselines", nargs="*", default=BASELINES); ap.add_argument("--procs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    for fam in a.families:
        if a.stage in ("data", "all") and not os.path.exists(f"{fam_dir(a.tag, fam)}/data.npz"): stage_data(a.tag, fam, a.N)
        if a.stage in ("oracle", "all"): stage_cpu(a.tag, fam, ["oracle"], a.procs)
        if a.stage in ("baselines", "all"): stage_cpu(a.tag, fam, a.baselines, a.procs)
        if a.stage in ("tsfm", "all"): stage_tsfm(a.tag, fam, a.models, a.batch)
