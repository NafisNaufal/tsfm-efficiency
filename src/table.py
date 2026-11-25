"""Table 1 (LaTeX) generated directly from results/main CSVs. python table.py > ../paper/tab_main.tex"""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = ["chronos-bolt-small", "chronos-bolt-base", "chronos-2", "timesfm-2.5", "timesfm-3", "tirex", "moirai-2-small", "toto"]
AB = {"chronos-bolt-small": "Bolt-S", "chronos-bolt-base": "Bolt-B", "chronos-2": "Chr-2", "timesfm-2.5": "TFM-2.5",
      "timesfm-3": "TFM-3", "tirex": "TiRex", "moirai-2-small": "Moirai", "toto": "Toto"}
NAME = {"ar1": "AR(1)", "ar3": "AR(3)", "arma11": "ARMA(1,1)", "sar": "Seasonal AR", "local_level": "Local level",
        "msar": "MS-AR", "random_walk": "Random walk", "white_noise": "White noise", "garch": "GARCH", "garch_t": "GARCH-$t$"}
S = pd.read_csv(f"{ROOT}/results/main/summary.csv"); S = S[S.h == 1]
H1 = pd.read_csv(f"{ROOT}/results/main/H1.csv"); H2 = pd.read_csv(f"{ROOT}/results/main/H2.csv")
f3 = lambda v: f"{v:.3f}" if v >= 0.001 else f"{v:.1e}".replace("e-0", "e$-$")
print(r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{2.1pt}")
print(r"\begin{tabular}{lcccccc}\hline")
print(r"& \multicolumn{2}{c}{MLE ER} & \multicolumn{2}{c}{TSFM ER, $T{=}2048$} & slope & RE$_{\min}$ at \\")
print(r"family & $T{=}16$ & $T{=}2048$ & best & worst & MLE / TSFMs & $T_{\rm low}$ \\ \hline")
for fam in ["ar1", "ar3", "arma11", "sar", "local_level", "msar", "random_walk", "white_noise"]:
    d = S[S.family == fam].set_index(["method", "T"])
    e = {m: d.loc[(m, 2048), "ER"] for m in P}; b, w = min(e, key=e.get), max(e, key=e.get)
    h2 = H2[H2.family == fam]; h1 = H1[H1.family == fam]; r = h1.loc[h1.RE_low.idxmin()]
    print(f"{NAME[fam]} & {f3(d.loc[('mle',16),'ER'])} & {f3(d.loc[('mle',2048),'ER'])} & {f3(e[b])} ({AB[b]}) & {f3(e[w])} ({AB[w]}) & "
          f"${h2.slope_mle.iloc[0]:.2f}$ / ${h2.slope.median():.2f}$ & {r.RE_low:.2f} ({AB[r.model]}) \\\\")
print(r"\hline")
for fam in ["garch", "garch_t"]:
    d = S[(S.family == fam) & (S["T"] == 2048)].set_index("method")
    e = {m: d.loc[m, "logvar_err"] for m in P}; b, w = min(e, key=e.get), max(e, key=e.get)
    print(f"{NAME[fam]}$^\\dagger$ & -- & {d.loc['mle','logvar_err']:.3f} & {e[b]:.3f} ({AB[b]}) & {e[w]:.3f} ({AB[w]}) & -- & -- \\\\")
print(r"\hline\end{tabular}")
print(r"\caption{Excess risk (ER, $h{=}1$) over the Bayes predictor. Slopes: $\log\mathrm{ER}$ vs.\ $\log T$ on $T\in[256,2048]$ "
      r"(TSFMs: median over 8 models). RE$_{\min}$: smallest ER$_{\rm TSFM}$/ER$_{\rm MLE}$ at the shortest admissible context "
      r"($T_{\rm low}{=}16$; 32 for seasonal AR). $^\dagger$GARCH rows: log-variance error of the predictive variance (conditional mean is 0); "
      r"TSFM variances are read off the 80\% interval assuming normality, which applied to the oracle's own quantiles already gives 0.0003 (GARCH) and 0.053 (GARCH-$t$). Bolt = Chronos-Bolt, Chr = Chronos, TFM = TimesFM. 95\% bootstrap CIs in the released tables.}\label{tab:main}")
print(r"\end{table}")
