"""Keyword-assisted coding of included full texts on the novelty dimensions (D1-D6).
A study is flagged on a dimension if its text matches the dimension's patterns at least `min_hits` times.
Flags are candidates only: every study that is flagged on >= 3 dimensions, or on D1 together with D2, is read by the
orchestrator and its coding is confirmed or corrected in research/05_dimension_coding_verified.csv.
python3 scripts/code_dimensions.py -> research/05_dimension_coding_auto.csv
"""
import csv, re, os, collections
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMS = {
    "D1_true_param_oracle": (r"bayes[- ]optimal|oracle predictor|optimal predictor|true parameters?|ground[- ]truth parameters?|"
                             r"known (?:data[- ]generating|dgp|parameters?)|theoretical (?:minimum|optimal) (?:mse|error|risk)|irreducible error|kalman filter", 2),
    "D2_same_context_estimator": (r"\bols\b|least[- ]squares|maximum likelihood|\bmle\b|yule[- ]walker|autoarima|auto[- ]arima|"
                                  r"fitted (?:on|to) the (?:same )?context|in-context (?:ols|least)|ar\(\d\) (?:fit|estimat)", 3),
    "D3_context_length_curve": (r"context[- ]length|lookback|history length|look-back|context window size|number of in-context|scaling (?:with|in) (?:context|history)", 4),
    "D4_martingale_null": (r"random walk|martingale|white noise|unpredictab|efficient market|spurious (?:predictab|pattern)", 3),
    "D5_volatility": (r"\bgarch\b|volatility|conditional variance|heteroscedastic", 3),
    "D6_mechanism_filter": (r"jacobian|impulse response|effective (?:filter|kernel)|linear filter|attention (?:map|weights) .{0,40}(?:lag|coefficient)|"
                            r"mesa[- ]optimi|implements? (?:gradient descent|least squares|ols|kalman)|saliency|mechanistic|probing", 2),
}
rows = []
inc = list(csv.DictReader(open(f"{HERE}/Screening_FT/FT_screening_log_Included.csv", encoding="utf-8-sig")))
for r in inc:
    rid = r["ID"]; p = f"{HERE}/fulltext_txt/{rid}.txt"
    txt = open(p, encoding="utf-8", errors="ignore").read().lower() if os.path.exists(p) else ""
    txt = re.split(r"\n(?:references|bibliography)\s*\n", txt)[0]          # do not count the reference list
    out = dict(ID=rid, year=r["Year"], first_author=r["First author"], title=r["Title"][:140])
    for k, (pat, mh) in DIMS.items():
        n = len(re.findall(pat, txt)); out[k + "_hits"] = n; out[k] = int(n >= mh)
    out["n_dims"] = sum(out[k] for k in DIMS)
    rows.append(out)
rows.sort(key=lambda d: -d["n_dims"])
w = csv.DictWriter(open(f"{os.path.dirname(HERE)}/05_dimension_coding_auto.csv", "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
c = collections.Counter(d["n_dims"] for d in rows); print("n_dims distribution", sorted(c.items()))
for k in DIMS: print(k, sum(d[k] for d in rows))
for d in rows[:40]:
    print(d["ID"], d["n_dims"], "".join(str(d[k]) for k in DIMS), d["year"], d["first_author"][:18], "|", d["title"][:90])
