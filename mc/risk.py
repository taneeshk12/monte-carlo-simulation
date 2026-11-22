"""Portfolio risk via Monte Carlo: Value-at-Risk and Expected Shortfall."""
import numpy as np

from .paths import correlated_gbm_terminal


def portfolio_var(S0, shares, mu, sigma, corr, horizon, n_paths, alpha=0.99, seed=None):
    S0 = np.asarray(S0, dtype=float)
    shares = np.asarray(shares, dtype=float)
    V0 = S0 @ shares
    ST = correlated_gbm_terminal(S0, mu, sigma, corr, horizon, n_paths, seed=seed)
    pnl = ST @ shares - V0
    var = -np.quantile(pnl, 1 - alpha)
    es = -pnl[pnl <= -var].mean()
    return {"V0": V0, "VaR": var, "ES": es, "pnl": pnl}
