"""Fit Heston parameters to real market history (SPY prices + VIX).

This is a simple time-series fit, not a full option-surface calibration:
  * variance proxy  v_t = (VIX_t / 100)^2
  * v_{t+1} = a + b v_t + noise   ->  kappa, theta (AR(1) regression)
  * noise / sqrt(v_t dt)          ->  xi (vol of variance)
  * corr(SPY returns, dv)         ->  rho
Good enough to make the simulated world look like the real one.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .sim import HestonParams, JumpParams

DATA = Path(__file__).resolve().parent.parent / "data"


def load_market(start="2005-01-01", end=None, refresh=False):
    """Daily SPY close and VIX close, cached to data/market.csv."""
    cache = DATA / "market.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf
    raw = yf.download(["SPY", "^VIX"], start=start, end=end, auto_adjust=True, progress=False)
    df = raw["Close"].rename(columns={"SPY": "spy", "^VIX": "vix"}).dropna()
    if df.empty:
        raise RuntimeError("Download from Yahoo failed. Try: pip install --upgrade yfinance")
    DATA.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df


def fit_heston(df, dt=1 / 252) -> HestonParams:
    v = (df["vix"].values / 100) ** 2
    ret = np.diff(np.log(df["spy"].values))
    x, y = v[:-1], v[1:]
    b, a = np.polyfit(x, y, 1)
    kappa = (1 - b) / dt
    theta = a / (1 - b)
    resid = y - (a + b * x)
    xi = np.std(resid / np.sqrt(x * dt))
    rho = np.corrcoef(ret, np.diff(v))[0, 1]
    return HestonParams(v0=float(v[-1]), kappa=float(kappa), theta=float(theta),
                        xi=float(xi), rho=float(rho))


def fit_jumps(df, threshold=3.0) -> JumpParams:
    """A day is a 'jump' when the SPY return is bigger than `threshold` times
    the daily move the VIX predicted the day before."""
    ret = np.diff(np.log(df["spy"].values))
    expected = df["vix"].values[:-1] / 100 / np.sqrt(252)
    j = ret[np.abs(ret / expected) > threshold]
    years = len(ret) / 252
    return JumpParams(lam=len(j) / years, mu_j=float(j.mean()), sig_j=float(j.std()))
