"""Market simulators that produce training paths for the hedging agent.

Every simulator returns (S, v): arrays of shape (n_paths, n_steps + 1) with the
stock price and the instantaneous variance at each hedging date.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class HestonParams:
    v0: float = 0.04      # current variance (vol 20%)
    kappa: float = 3.0    # speed of mean reversion
    theta: float = 0.04   # long-run variance
    xi: float = 0.6       # vol of variance
    rho: float = -0.7     # correlation between price and variance shocks
    mu: float = 0.0       # drift of the stock


@dataclass
class JumpParams:
    lam: float = 1.0      # expected jumps per year
    mu_j: float = -0.03   # mean log jump size
    sig_j: float = 0.05   # std of log jump size


def gbm(S0, sigma, T, n_steps, n_paths, mu=0.0, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    logS = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), logS]))
    v = np.full_like(S, sigma**2)
    return S, v


def heston(S0, p: HestonParams, T, n_steps, n_paths, substeps=10, seed=None,
           jumps: "JumpParams | None" = None):
    """Heston model, Euler scheme with full truncation. Simulated on a finer grid
    (substeps per hedging date) for accuracy, then sampled at hedging dates.

    With `jumps` this becomes the Bates model: Heston plus Merton-style jumps,
    sudden moves that arrive at random (Poisson) with normally distributed
    log size. The drift is compensated so jumps don't change the expected return."""
    rng = np.random.default_rng(seed)
    dt = T / (n_steps * substeps)
    S = np.empty((n_paths, n_steps + 1)); v = np.empty_like(S)
    s = np.full(n_paths, float(S0)); var = np.full(n_paths, p.v0)
    S[:, 0], v[:, 0] = s, var
    comp = jumps.lam * (np.exp(jumps.mu_j + 0.5 * jumps.sig_j**2) - 1) if jumps else 0.0
    for i in range(n_steps):
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)
            vp = np.maximum(var, 0.0)
            s = s * np.exp((p.mu - comp - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1)
            if jumps:
                n = rng.poisson(jumps.lam * dt, n_paths)
                s *= np.exp(n * jumps.mu_j + np.sqrt(n) * jumps.sig_j * rng.standard_normal(n_paths))
            var = var + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp * dt) * z2
        S[:, i + 1], v[:, i + 1] = s, np.maximum(var, 0.0)
    return S, v
