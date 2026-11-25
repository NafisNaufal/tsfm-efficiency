"""Publication figures (ESANN: single column, ~12 cm text width). python figures.py --tag main
Reads results/<tag>/summary.csv and re.csv written by analyze.py; writes paper/figs/*.pdf.
Palette: validated 8-slot categorical order (dataviz validator, light mode); every model also has its own marker,
because three hues fall below 3:1 contrast on white (secondary encoding); all numbers are also in the tables.
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figs")
MODELS = ["chronos-bolt-small", "chronos-bolt-base", "chronos-2", "timesfm-2.5", "timesfm-3", "tirex", "moirai-2-small", "toto", "toto-median"]
LABEL = {"chronos-bolt-small": "Chronos-Bolt-S", "chronos-bolt-base": "Chronos-Bolt-B", "chronos-2": "Chronos-2",
         "timesfm-2.5": "TimesFM-2.5", "timesfm-3": "TimesFM-3", "tirex": "TiRex", "moirai-2-small": "Moirai-2-S", "toto": "Toto (sample mean)", "toto-median": "Toto",
         "mle": "MLE (correct class)", "autoarima": "AutoARIMA", "ets": "ETS", "naive": "Naive"}
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
MARK = dict(zip(MODELS, ["o", "s", "^", "D", "v", "P", "X", "*"]))
COLOR["toto-median"], MARK["toto-median"] = COLOR["toto"], MARK["toto"]
PLOT_MODELS = [m for m in MODELS if m != "toto"]   # figures show Toto's median (D5); its sample mean is in the tables
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3df"
FAMNAME = {"ar1": "AR(1)", "ar3": "AR(3)", "arma11": "ARMA(1,1)", "sar": "Seasonal AR", "local_level": "Local level",
           "msar": "Markov-switching AR", "garch": "GARCH(1,1)", "garch_t": "GARCH-t", "random_walk": "Random walk",
           "white_noise": "White noise"}

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.spines.top": False, "axes.spines.right": False})


def _style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_xscale("log", base=2); ax.set_xticks([16, 64, 256, 1024]); ax.set_xticklabels(["16", "64", "256", "1024"])


def fig_learning_curves(S, fams, fname, floor=5e-4):
    """Excess risk over the Bayes predictor vs context length (h=1), log-log; values below `floor` drawn at floor (hollow)."""
    fig, axes = plt.subplots(1, len(fams), figsize=(4.85, 2.15), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), fams):
        d = S[(S.family == fam) & (S.h == 1)]
        for m in ["mle"] + PLOT_MODELS:
            e = d[d.method == m].sort_values("T")
            if e.empty: continue
            y = e.ER.clip(lower=floor).values; lo = e.ER_lo.clip(lower=floor).values; hi = e.ER_hi.clip(lower=floor).values
            if m == "mle":
                ax.plot(e["T"], y, color=INK, lw=1.4, ls="--", zorder=5, label=LABEL[m])
                ax.fill_between(e["T"], lo, hi, color=INK, alpha=0.08, lw=0)
            else:
                ax.plot(e["T"], y, color=COLOR[m], lw=1.0, marker=MARK[m], ms=3.2, mew=0.4, mec="white", label=LABEL[m])
        Ts = np.array([16, 2048.]); ax.plot(Ts, 1.0 * 16 / Ts, color=MUTED, lw=0.6, ls=":")   # reference slope -1
        ax.set_yscale("log"); _style(ax); ax.set_title(FAMNAME[fam], color=INK); ax.set_xlabel("context length $T$")
    np.atleast_1d(axes)[0].set_ylabel("excess risk $\\mathrm{ER}$ (h=1)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.subplots_adjust(top=0.70)
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.8, columnspacing=1.0)
    fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_relative_efficiency(R, fams, fname):
    fig, axes = plt.subplots(1, len(fams), figsize=(4.85, 1.6), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), fams):
        d = R[R.family == fam]
        ax.axhline(1, color=INK, lw=0.8)
        for m in PLOT_MODELS:
            e = d[d.method == m].sort_values("T")
            if e.empty: continue
            ax.plot(e["T"], e.RE, color=COLOR[m], lw=1.0, marker=MARK[m], ms=3, mew=0.4, mec="white", label=LABEL[m])
        ax.set_yscale("log"); _style(ax); ax.set_title(FAMNAME[fam], color=INK); ax.set_xlabel("context length $T$")
    np.atleast_1d(axes)[0].set_ylabel("$\\mathrm{ER}_{\\mathrm{TSFM}}/\\mathrm{ER}_{\\mathrm{MLE}}$")
    fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_filters(fname="fig_filter"):
    """(a) implied lag-1 coefficient vs true phi (AR(1), T=1024, 200 series; binned means); (b,c) filter distance vs T."""
    J = f"{ROOT}/results/jacobian/eps0.001"
    prm = list(np.load(f"{ROOT}/results/main/ar1/data.npz", allow_pickle=True)["params"])[:200]
    phi = np.array([p["phi"][0] for p in prm]); bins = np.linspace(-1, 1, 9); mid = (bins[:-1] + bins[1:]) / 2
    F = pd.read_csv(f"{ROOT}/results/jacobian/filters_eps0.001.csv"); F = F[F.method != "tirex"]
    show = ["chronos-bolt-base", "chronos-2", "timesfm-2.5", "moirai-2-small"]
    fig, axes = plt.subplots(1, 3, figsize=(4.85, 1.75), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axes[0]; ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.6, ls=":")
    for m in ["mle"] + show:
        w = np.load(f"{J}/ar1/T1024/J_{m}.npy")[:, 0]; idx = np.digitize(phi, bins) - 1
        y = np.array([w[idx == k].mean() if (idx == k).sum() >= 5 else np.nan for k in range(len(mid))])
        st = dict(color=INK, lw=1.4, ls="--") if m == "mle" else dict(color=COLOR[m], lw=1.0, marker=MARK[m], ms=3, mew=0.4, mec="white")
        ax.plot(mid, y, label=LABEL[m], **st)
    ax.set_xlabel("true $\\phi$ (AR(1))"); ax.set_ylabel("implied lag-1 weight"); ax.set_title("(a) $T=1024$", color=INK)
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    for ax, fam, ttl in [(axes[1], "ar3", "(b) AR(3)"), (axes[2], "local_level", "(c) Local level")]:
        d = F[F.family == fam]
        for m in ["mle"] + [k for k in PLOT_MODELS if k not in ("tirex", "toto-median")]:
            e = d[d.method == m].sort_values("T")
            if e.empty: continue
            st = dict(color=INK, lw=1.4, ls="--") if m == "mle" else dict(color=COLOR[m], lw=1.0, marker=MARK[m], ms=3, mew=0.4, mec="white")
            ax.plot(e["T"], e.filt_dist_med, label=LABEL[m], **st)
        ax.set_xscale("log", base=2); ax.set_xticks([64, 256, 1024]); ax.set_xticklabels(["64", "256", "1024"])
        ax.set_ylim(0, 1.25); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title(ttl, color=INK)
        ax.set_xlabel("context length $T$")
    axes[1].set_ylabel("distance to Bayes filter")
    h, l = axes[1].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, w_pad=1.0); fig.subplots_adjust(top=0.70)
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.8, columnspacing=1.0)
    fig.savefig(f"{OUT}/{fname}.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main")
    ap.add_argument("--fams", nargs="*", default=["ar3", "arma11", "local_level", "random_walk"])
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    S = pd.read_csv(f"{ROOT}/results/{a.tag}/summary.csv"); R = pd.read_csv(f"{ROOT}/results/{a.tag}/re.csv")
    fig_learning_curves(S, a.fams, f"fig_er_{a.tag}"); fig_relative_efficiency(R, a.fams, f"fig_re_{a.tag}")
    if a.tag == "main": fig_filters()
    print("written", OUT)

