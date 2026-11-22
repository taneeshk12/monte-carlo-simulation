"""Monte Carlo option pricing with analytical benchmarks."""
import numpy as np
from scipy.stats import norm

from .paths import gbm_paths


def black_scholes(S0, K, r, sigma, T, kind="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def _summary(discounted_payoffs):
    n = len(discounted_payoffs)
    price = discounted_payoffs.mean()
    se = discounted_payoffs.std(ddof=1) / np.sqrt(n)
    return {"price": price, "std_err": se, "ci95": (price - 1.96 * se, price + 1.96 * se)}


def mc_european(S0, K, r, sigma, T, n_paths, kind="call", antithetic=False, seed=None):
    S = gbm_paths(S0, r, sigma, T, 1, n_paths, antithetic=antithetic, seed=seed)[:, -1]
    payoff = np.maximum(S - K, 0) if kind == "call" else np.maximum(K - S, 0)
    if antithetic:  # average each pair so the std error reflects the variance reduction
        h = len(payoff) // 2
        payoff = 0.5 * (payoff[:h] + payoff[h:])
    return _summary(np.exp(-r * T) * payoff)


def mc_asian(S0, K, r, sigma, T, n_steps, n_paths, kind="call", control_variate=True, seed=None):
    """Arithmetic-average Asian option. Optional control variate: the geometric
    Asian option, which has a closed-form price."""
    S = gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)[:, 1:]
    arith = S.mean(axis=1)
    sign = 1 if kind == "call" else -1
    disc = np.exp(-r * T)
    Y = disc * np.maximum(sign * (arith - K), 0)
    if not control_variate:
        return _summary(Y)

    geo = np.exp(np.log(S).mean(axis=1))
    X = disc * np.maximum(sign * (geo - K), 0)
    # Closed form for discretely-monitored geometric Asian
    n = n_steps
    sig_g = sigma * np.sqrt((n + 1) * (2 * n + 1) / (6 * n**2))
    mu_g = 0.5 * sig_g**2 + (r - 0.5 * sigma**2) * (n + 1) / (2 * n)
    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sig_g**2) * T) / (sig_g * np.sqrt(T))
    d2 = d1 - sig_g * np.sqrt(T)
    if kind == "call":
        geo_exact = disc * (S0 * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        geo_exact = disc * (K * norm.cdf(-d2) - S0 * np.exp(mu_g * T) * norm.cdf(-d1))
    beta = np.cov(Y, X)[0, 1] / X.var(ddof=1)
    return _summary(Y - beta * (X - geo_exact))


def mc_barrier_up_and_out(S0, K, B, r, sigma, T, n_steps, n_paths, seed=None):
    """Up-and-out call (discrete monitoring)."""
    S = gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    alive = S.max(axis=1) < B
    payoff = np.where(alive, np.maximum(S[:, -1] - K, 0), 0.0)
    return _summary(np.exp(-r * T) * payoff)
