"""Classical estimators fitted on the SAME context window as the TSFMs (CPU, parallel over series).

fit_predict(method, family, ctx, H, prm) -> (mean (H,), sd (H,) or None)
  'mle'       correctly specified model class (order/structure known, parameters + mean estimated)
  'autoarima' statsforecast AutoARIMA (practitioner baseline; season_length = s for seasonal families)
  'ets'       statsforecast AutoETS
  'naive'     last value (random-walk forecast)
Failures are returned as NaN and counted; they are never silently replaced.
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")


def _ols_ar(ctx, lags, H):
    lags = list(lags); L = max(lags)
    y = ctx[L:]; X = np.column_stack([np.ones(len(y))] + [ctx[L - l:len(ctx) - l] for l in lags])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; s2 = resid.var() * len(y) / max(len(y) - X.shape[1], 1)
    hist = list(ctx); mean = []
    for _ in range(H):
        m = beta[0] + sum(beta[i + 1] * hist[-l] for i, l in enumerate(lags)); mean.append(m); hist.append(m)
    # h-step variance via psi weights of the fitted AR polynomial
    phi = np.zeros(L);
    for i, l in enumerate(lags): phi[l - 1] = beta[i + 1]
    psi = [1.0]
    for j in range(1, H): psi.append(sum(phi[k] * psi[j - k - 1] for k in range(min(L, j))))
    return np.array(mean), np.sqrt(np.cumsum(np.array(psi) ** 2) * s2)


def _mle(family, ctx, H, prm):
    if family == "ar1": return _ols_ar(ctx, [1], H)
    if family == "ar3": return _ols_ar(ctx, [1, 2, 3], H)
    if family == "sar":
        s = prm["s"]; return _ols_ar(ctx, [1, s, s + 1], H)
    if family == "arma11":
        from statsmodels.tsa.arima.model import ARIMA
        r = ARIMA(ctx, order=(1, 0, 1), trend="c").fit(method_kwargs={"maxiter": 200})
        f = r.get_forecast(H); return np.asarray(f.predicted_mean), np.sqrt(np.asarray(f.var_pred_mean))
    if family == "local_level":
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        r = UnobservedComponents(ctx, level="llevel").fit(disp=False, maxiter=200)
        f = r.get_forecast(H); return np.asarray(f.predicted_mean), np.sqrt(np.asarray(f.var_pred_mean))
    if family == "random_walk":   # correctly specified up to an unknown drift
        d = np.diff(ctx); mu, s = d.mean(), d.std(ddof=1); h = np.arange(1, H + 1)
        return ctx[-1] + mu * h, s * np.sqrt(h)
    if family == "white_noise":   # correctly specified up to an unknown mean
        return np.full(H, ctx.mean()), np.full(H, ctx.std(ddof=1))
    if family == "msar":
        from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
        r = MarkovAutoregression(ctx, k_regimes=2, order=1, switching_ar=True, switching_variance=True).fit(disp=False, maxiter=200)
        P = r.regime_transition[:, :, -1].T if r.regime_transition.ndim == 3 else r.regime_transition.T
        xi = np.asarray(r.filtered_marginal_probabilities)[-1]
        par = r.params; names = r.model.param_names
        c = [par[names.index(f"const[{k}]")] for k in range(2)]; ph = [par[names.index(f"ar.L1[{k}]")] for k in range(2)]
        mean, x = [], ctx[-1]; pr = xi
        for _ in range(H):   # exact one-step mean, plug-in recursion beyond (documented approximation)
            pr = P @ pr if P.shape == (2, 2) else pr
            m = sum(pr[k] * (c[k] + ph[k] * x) for k in range(2)); mean.append(m); x = m
        return np.array(mean), None
    if family in ("garch", "garch_t"):
        from arch import arch_model
        r = arch_model(ctx * 10, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        v = r.forecast(horizon=H, reindex=False).variance.values[-1] / 100.0
        return np.zeros(H), np.sqrt(v)
    raise KeyError(family)


def _statsforecast(kind, ctx, H, season):
    from statsforecast.models import AutoARIMA, AutoETS
    m = AutoARIMA(season_length=season) if kind == "autoarima" else AutoETS(season_length=season)
    f = m.forecast(y=ctx.astype(float), h=H, level=[80])
    mean = np.asarray(f["mean"]); hi = np.asarray(f["hi-80"])
    return mean, (hi - mean) / 1.2815515655446004


GUARD = 20.0   # pre-registration deviation D1: a fitted forecast > GUARD context-sds from the context mean is a failed fit


def fit_predict(method, family, ctx, H, prm):
    try:
        if method == "naive": return np.full(H, ctx[-1]), None
        if method == "mle": m, sd = _mle(family, ctx, H, prm)
        else:
            season = prm.get("s", 1) if family == "sar" else 1
            m, sd = _statsforecast(method, ctx, H, season)
        m = np.asarray(m, float)
        if not np.all(np.isfinite(m)) or np.max(np.abs(m - ctx.mean())) > GUARD * max(ctx.std(), 1e-8):
            return np.full(H, np.nan), None
        return m, sd
    except Exception:
        return np.full(H, np.nan), None
