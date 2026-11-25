"""Data-generating processes with exact (context-)Bayes oracle predictors.

Every generator returns a Batch with
  x      : (N, T_max + H) array; the forecast origin is T_max (contexts are x[:, T_max-T:T_max])
  params : list of dicts (one per series)
and every family exposes  oracle(x_ctx, params, H) -> (mean (N,H), samples or quantile fn).

Oracle definition (used throughout the paper): the Bayes predictor of the future given the SAME
context window and the TRUE parameters (for linear-Gaussian families computed exactly with a Kalman
filter / linear recursion; for MS-AR with the Hamilton filter; for GARCH with the variance recursion
initialised at the stationary variance, whose influence decays as beta^T).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

BURN = 500


@dataclass
class Batch:
    family: str
    x: np.ndarray            # (N, L)
    params: list = field(default_factory=list)


# ---------------------------------------------------------------- helpers
def pacf_to_ar(pac):
    """Durbin-Levinson map from partial autocorrelations in (-1,1) to stationary AR coefficients."""
    phi = np.array([pac[0]])
    for k in range(1, len(pac)):
        phi = np.concatenate([phi - pac[k] * phi[::-1], [pac[k]]])
    return phi


def _ar_sim(phi, n, L, rng, sigma=1.0):
    p = len(phi)
    e = rng.standard_normal((n, L + BURN)) * sigma
    x = np.zeros((n, L + BURN))
    for t in range(L + BURN):
        acc = e[:, t].copy()
        for j in range(p):
            if t - j - 1 >= 0:
                acc += phi[j] * x[:, t - j - 1]
        x[:, t] = acc
    return x[:, BURN:]


# ---------------------------------------------------------------- generators
def gen_ar1(n, L, rng, phi=None):
    phis = rng.uniform(-0.95, 0.95, n) if phi is None else np.full(n, phi)
    e = rng.standard_normal((n, L + BURN))
    x = np.zeros_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(1 - phis ** 2)
    for t in range(1, L + BURN):
        x[:, t] = phis * x[:, t - 1] + e[:, t]
    return Batch("ar1", x[:, BURN:], [{"phi": np.array([p]), "sigma": 1.0} for p in phis])


def gen_ar3(n, L, rng):
    out, params = np.zeros((n, L)), []
    for i in range(n):
        phi = pacf_to_ar(rng.uniform(-0.9, 0.9, 3))
        out[i] = _ar_sim(phi, 1, L, rng)[0]
        params.append({"phi": phi, "sigma": 1.0})
    return Batch("ar3", out, params)


def gen_sar(n, L, rng, s=12):
    """Multiplicative (1,0,0)x(1,0,0)_s: (1-phi B)(1-Phi B^s) x_t = e_t."""
    out, params = np.zeros((n, L)), []
    for i in range(n):
        ph, Ph = rng.uniform(-0.6, 0.6), rng.uniform(0.4, 0.95)
        phi = np.zeros(s + 1); phi[0] = ph; phi[s - 1] = Ph; phi[s] = -ph * Ph
        out[i] = _ar_sim(phi, 1, L, rng)[0]
        params.append({"phi": phi, "sigma": 1.0, "s": s})
    return Batch("sar", out, params)


def gen_arma11(n, L, rng):
    out, params = np.zeros((n, L)), []
    for i in range(n):
        while True:
            ph, th = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
            if abs(ph + th) > 0.2:      # avoid near pole-zero cancellation (near white noise)
                break
        e = rng.standard_normal(L + BURN + 1)
        x = np.zeros(L + BURN + 1)
        for t in range(1, L + BURN + 1):
            x[t] = ph * x[t - 1] + e[t] + th * e[t - 1]
        out[i] = x[BURN + 1:]
        params.append({"phi": ph, "theta": th, "sigma": 1.0})
    return Batch("arma11", out, params)


def gen_local_level(n, L, rng):
    qs = np.exp(rng.uniform(np.log(0.01), np.log(10.0), n))
    eta = rng.standard_normal((n, L)) * np.sqrt(qs)[:, None]
    mu = np.cumsum(eta, axis=1)
    x = mu + rng.standard_normal((n, L))
    return Batch("local_level", x, [{"q": q} for q in qs])


def gen_random_walk(n, L, rng):
    x = np.cumsum(rng.standard_normal((n, L)), axis=1)
    return Batch("random_walk", x, [{} for _ in range(n)])


def gen_white_noise(n, L, rng):
    return Batch("white_noise", rng.standard_normal((n, L)), [{} for _ in range(n)])


def gen_msar(n, L, rng):
    """2-regime Markov-switching AR(1): regime k has (phi_k, sigma_k); persistent transitions."""
    out, params = np.zeros((n, L)), []
    for i in range(n):
        p11, p22 = rng.uniform(0.9, 0.99, 2)
        ph = np.array([rng.uniform(0.5, 0.95), rng.uniform(-0.8, 0.0)])
        sg = np.array([1.0, rng.uniform(1.5, 3.0)])
        P = np.array([[p11, 1 - p11], [1 - p22, p22]])
        s = 0; xv = 0.0; xs = np.zeros(L + BURN)
        for t in range(L + BURN):
            s = rng.choice(2, p=P[s])
            xv = ph[s] * xv + sg[s] * rng.standard_normal()
            xs[t] = xv
        out[i] = xs[BURN:]
        params.append({"P": P, "phi": ph, "sigma": sg})
    return Batch("msar", out, params)


def gen_garch(n, L, rng, dist="normal", nu=5.0):
    out, params = np.zeros((n, L)), []
    for i in range(n):
        a = rng.uniform(0.03, 0.15); per = rng.uniform(0.85, 0.99); b = per - a
        w = 1.0 - per                                          # unconditional variance 1
        h = 1.0; r = np.zeros(L + BURN); hs = np.zeros(L + BURN + 1); hs[0] = h
        for t in range(L + BURN):
            z = rng.standard_normal() if dist == "normal" else rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
            r[t] = np.sqrt(h) * z
            h = w + a * r[t] ** 2 + b * h
            hs[t + 1] = h
        out[i] = r[BURN:]
        params.append({"omega": w, "alpha": a, "beta": b, "dist": dist, "nu": nu})
    return Batch("garch" if dist == "normal" else "garch_t", out, params)


GENERATORS = {
    "ar1": gen_ar1, "ar3": gen_ar3, "sar": gen_sar, "arma11": gen_arma11,
    "local_level": gen_local_level, "random_walk": gen_random_walk, "white_noise": gen_white_noise,
    "msar": gen_msar, "garch": gen_garch, "garch_t": lambda n, L, rng: gen_garch(n, L, rng, "t"),
}


# ---------------------------------------------------------------- oracles
# Each oracle returns (mean (H,), sampler(k, rng) -> (k, H)) for one context.

def _linear_ar_oracle(ctx, phi, H, sigma=1.0):
    p = len(phi)
    hist = list(ctx[-p:]) if len(ctx) >= p else [0.0] * (p - len(ctx)) + list(ctx)
    mean = []
    for _ in range(H):
        m = float(np.dot(phi, hist[::-1][:p]))
        mean.append(m); hist.append(m)
    # psi weights for h-step variance
    psi = [1.0]
    for j in range(1, H):
        psi.append(sum(phi[k] * psi[j - k - 1] for k in range(min(p, j))))
    var = np.cumsum(np.array(psi) ** 2) * sigma ** 2
    return np.array(mean), var


def _kalman_arma11(ctx, ph, th, H):
    """Exact finite-context Gaussian predictor of ARMA(1,1) via state-space (Harvey form)."""
    T_ = np.array([[ph, 1.0], [0.0, 0.0]]); R = np.array([1.0, th]); Z = np.array([1.0, 0.0])
    Q = np.outer(R, R)
    # stationary initial covariance
    P = np.linalg.solve(np.eye(4) - np.kron(T_, T_), Q.flatten()).reshape(2, 2)
    a = np.zeros(2)
    for y in ctx:
        F = Z @ P @ Z; K = P @ Z / F
        a = a + K * (y - Z @ a); P = P - np.outer(K, Z @ P)
        a = T_ @ a; P = T_ @ P @ T_.T + Q
    mean, var = [], []
    for _ in range(H):
        mean.append(Z @ a); var.append(Z @ P @ Z)
        a = T_ @ a; P = T_ @ P @ T_.T + Q
    return np.array(mean), np.array(var)


def _kalman_local_level(ctx, q, H):
    """Local level with diffuse prior on the level (exact large-variance limit)."""
    a, P = ctx[0], 1.0 + q            # after first observation with diffuse prior: a=y1, P=1 (+q at predict)
    for y in ctx[1:]:
        F = P + 1.0; K = P / F
        a = a + K * (y - a); P = P * (1 - K) + q
    mean = np.full(H, a)
    var = P + 1.0 + q * np.arange(H)  # P already includes one-step state noise
    return mean, var


def oracle_gaussian(family, ctx, prm, H):
    if family in ("ar1", "ar3", "sar"):
        return _linear_ar_oracle(ctx, prm["phi"], H)
    if family == "arma11":
        return _kalman_arma11(ctx, prm["phi"], prm["theta"], H)
    if family == "local_level":
        return _kalman_local_level(ctx, prm["q"], H)
    if family == "random_walk":
        return np.full(H, ctx[-1]), np.arange(1, H + 1, dtype=float)
    if family == "white_noise":
        return np.zeros(H), np.ones(H)
    raise KeyError(family)


def hamilton_filter(ctx, P, phi, sigma):
    """Filtered regime probabilities P(s_T | x_1..x_T) for MS-AR(1)."""
    pi = np.linalg.matrix_power(P.T, 200)[:, 0]
    xi = pi.copy()
    for t in range(1, len(ctx)):
        pred = P.T @ xi
        lik = np.exp(-0.5 * ((ctx[t] - phi * ctx[t - 1]) / sigma) ** 2) / sigma
        xi = pred * lik; xi = xi / xi.sum()
    return xi


def oracle_sampler(family, ctx, prm, H, k, rng):
    """Draw k future paths from the oracle predictive (used for non-Gaussian families and CRPS)."""
    if family == "msar":
        xi = hamilton_filter(ctx, prm["P"], prm["phi"], prm["sigma"])
        out = np.zeros((k, H)); s = rng.choice(2, size=k, p=xi); xv = np.full(k, ctx[-1])
        for h in range(H):
            u = rng.random(k); s = np.where(u < prm["P"][s, 0], 0, 1)
            xv = prm["phi"][s] * xv + prm["sigma"][s] * rng.standard_normal(k); out[:, h] = xv
        return out
    if family in ("garch", "garch_t"):
        w, a, b = prm["omega"], prm["alpha"], prm["beta"]
        h = 1.0
        for r in ctx:
            h = w + a * r ** 2 + b * h
        hv = np.full(k, h); out = np.zeros((k, H))
        for j in range(H):
            if prm["dist"] == "normal":
                z = rng.standard_normal(k)
            else:
                nu = prm["nu"]; z = rng.standard_t(nu, k) * np.sqrt((nu - 2) / nu)
            out[:, j] = np.sqrt(hv) * z
            hv = w + a * out[:, j] ** 2 + b * hv
        return out
    m, v = oracle_gaussian(family, ctx, prm, H)
    # Gaussian paths with correct marginals per step (sufficient for marginal CRPS / quantiles)
    return m[None, :] + np.sqrt(v)[None, :] * rng.standard_normal((k, H))


def oracle_point(family, ctx, prm, H, rng=None, k=4000):
    """Oracle conditional mean (MSE-optimal point forecast)."""
    if family == "msar":
        xi = hamilton_filter(ctx, prm["P"], prm["phi"], prm["sigma"])
        # exact mean recursion: E[x_{T+h}] = sum_s P(s_{T+h}) ... approximate by MC only for h>1
        if H == 1:
            pred = prm["P"].T @ xi
            return np.array([np.dot(pred, prm["phi"]) * ctx[-1]])
        rng = rng or np.random.default_rng(0)
        return oracle_sampler(family, ctx, prm, H, k, rng).mean(0)
    if family in ("garch", "garch_t"):
        return np.zeros(H)
    return oracle_gaussian(family, ctx, prm, H)[0]
