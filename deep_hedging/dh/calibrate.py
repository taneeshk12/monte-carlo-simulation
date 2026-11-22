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


def fit_heston(df, price="spy", vol="vix", per_year=252) -> HestonParams:
    """`vol` is an implied-vol index in % (VIX for SPY, DVOL for BTC)."""
    dt = 1 / per_year
    v = (df[vol].values / 100) ** 2
    ret = np.diff(np.log(df[price].values))
    x, y = v[:-1], v[1:]
    b, a = np.polyfit(x, y, 1)
    kappa = (1 - b) / dt
    theta = a / (1 - b)
    resid = y - (a + b * x)
    xi = np.std(resid / np.sqrt(x * dt))
    rho = np.corrcoef(ret, np.diff(v))[0, 1]
    return HestonParams(v0=float(v[-1]), kappa=float(kappa), theta=float(theta),
                        xi=float(xi), rho=float(rho))


def fit_jumps(df, price="spy", vol="vix", per_year=252, threshold=3.0) -> JumpParams:
    """A day is a 'jump' when the return is bigger than `threshold` times
    the daily move the vol index predicted the day before."""
    ret = np.diff(np.log(df[price].values))
    expected = df[vol].values[:-1] / 100 / np.sqrt(per_year)
    j = ret[np.abs(ret / expected) > threshold]
    years = len(ret) / per_year
    return JumpParams(lam=len(j) / years, mu_j=float(j.mean()), sig_j=float(j.std()))
