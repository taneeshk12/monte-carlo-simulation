"""Path generators for asset prices."""
import numpy as np


def gbm_paths(S0, mu, sigma, T, n_steps, n_paths, antithetic=False, seed=None):
    """Simulate Geometric Brownian Motion paths (exact log-normal discretisation).

    Returns array of shape (n_paths, n_steps + 1), including S0 in column 0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if antithetic:
        half = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.vstack([half, -half])
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.hstack([np.zeros((Z.shape[0], 1)), log_paths])
    return S0 * np.exp(log_paths)


def correlated_gbm_terminal(S0, mu, sigma, corr, T, n_paths, seed=None):
    """Terminal prices of several correlated GBM assets (one step to horizon T)."""
    rng = np.random.default_rng(seed)
    S0, mu, sigma = map(np.asarray, (S0, mu, sigma))
    L = np.linalg.cholesky(np.asarray(corr))
    Z = rng.standard_normal((n_paths, len(S0))) @ L.T
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
