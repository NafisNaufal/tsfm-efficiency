"""Deterministic Boolean title/abstract filter applied identically to all database exports (search step 2)."""
import re, sys, glob, os, json
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
A = r"time[- ]series|forecast|dynamical system|autoregress|stochastic process"
B = r"foundation model|zero-shot|in-context|large language model|\bLLMs?\b|global (forecasting )?models?|pre-?trained (time[- ]series|forecasting) models?"
C = r"oracle|bayes|context (length|window|size)|look-?back|random walk|martingale|white noise|impulse response|jacobian|\bARIMA\b|\bGARCH\b|kalman|autoregressive process|\bAR\(\d|least squares|optimal predict|autoregressive (model|dependence|coefficient)|\bAR model|(window|input|history) (size|length)|observation window|long context|linear (dynamical )?system|interpretab|mechanis"
I, T = r"in-context", r"transformer|attention"
def blocks(p):
    cur = []
    for line in open(p, encoding="utf-8"):
        cur.append(line)
        if line.startswith("ER  -"): yield "".join(cur); cur = []
log = {}
for p in sorted(glob.glob(IN + "/*.ris")):
    keep, tot = [], 0
    for b in blocks(p):
        tot += 1
        t = " ".join(l[6:] for l in b.splitlines() if l[:2] in ("TI", "AB"))
        f1 = re.search(A, t, re.I) and re.search(B, t, re.I) and re.search(C, t, re.I)
        f2 = re.search(A, t, re.I) and re.search(I, t, re.I) and re.search(T, t, re.I)
        if f1 or f2: keep.append(b)
    open(OUT + "/" + os.path.basename(p), "w").write("\n".join(keep)); log[os.path.basename(p)] = {"in": tot, "kept": len(keep)}
json.dump({"filter": {"F1": "A AND B AND C", "F2": "A AND I AND T", "A": A, "B": B, "C": C, "I": I, "T": T}, "per_file": log}, open(OUT + "/../filter_log.json", "w"), indent=1)
print(json.dumps(log))
