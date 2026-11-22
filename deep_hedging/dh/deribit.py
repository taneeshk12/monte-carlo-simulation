"""Free historical Bitcoin data from Deribit's public API (no account needed).

  * daily BTC index price at 08:00 UTC (Deribit's expiry/settlement time)
  * DVOL: Deribit's 30-day implied volatility index for BTC (the "crypto VIX")
  * real market prices of expired monthly BTC options

Everything is cached in data/ so it is downloaded only once.
"""
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
LIVE = "https://www.deribit.com/api/v2/public"
HIST = "https://history.deribit.com/api/v2/public"
DAY_MS = 86_400_000


def _get(base, method, **params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base}/{method}?{q}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)["result"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def _ms(d):  # date -> 08:00 UTC timestamp in ms
    return int(datetime(d.year, d.month, d.day, 8, tzinfo=timezone.utc).timestamp() * 1000)


def load_btc_market(refresh=False):
    """DataFrame indexed by date: btc (index price at 08:00 UTC), dvol (DVOL, %)."""
    cache = DATA / "btc_market.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    rows, offset = [], 0
    while True:
        res = _get(LIVE, "get_delivery_prices", index_name="btc_usd", offset=offset, count=100)
        rows += res["data"]
        offset += len(res["data"])          # the API returns at most 100 rows per call
        if not res["data"] or offset >= res["records_total"]:
            break
    btc = pd.Series({pd.Timestamp(r["date"]): r["delivery_price"] for r in rows}).sort_index()

    dvol = {}
    start = pd.Timestamp("2021-03-01")
    while start < btc.index[-1]:
        end = start + pd.Timedelta(days=900)
        res = _get(LIVE, "get_volatility_index_data", currency="BTC", resolution="1D",
                   start_timestamp=int(start.timestamp() * 1000), end_timestamp=int(end.timestamp() * 1000))
        for ts, o, h, l, c in res["data"]:
            dvol[pd.Timestamp(ts, unit="ms").normalize()] = o   # value at 00:00, close to 08:00
        start = end
    df = pd.DataFrame({"btc": btc, "dvol": pd.Series(dvol)}).dropna()
    DATA.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df


def load_btc_options(market, days=30, refresh=False):
    """For every monthly BTC expiry: the call closest to the money `days` before
    expiry, with its real market price at that moment.

    Returns DataFrame: start, expiry, strike, S0, premium_usd, iv.
    """
    cache = DATA / f"btc_options_{days}d.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["start", "expiry"])

    insts = _get(HIST, "get_instruments", currency="BTC", kind="option", expired="true")
    monthly = [i for i in insts if i["settlement_period"] == "month" and i["option_type"] == "call"]
    by_expiry = {}
    for i in monthly:
        by_expiry.setdefault(i["expiration_timestamp"], []).append(i)

    out = []
    for exp_ms in sorted(by_expiry):
        expiry = pd.Timestamp(exp_ms, unit="ms").normalize()
        start = expiry - pd.Timedelta(days=days)
        if start not in market.index or expiry not in market.index:
            continue
        S0 = market.loc[start, "btc"]
        t0 = _ms(start)
        listed = [i for i in by_expiry[exp_ms] if i["creation_timestamp"] <= t0]
        if not listed:
            continue
        inst = min(listed, key=lambda i: abs(i["strike"] - S0))
        trades = _get(HIST, "get_last_trades_by_instrument_and_time",
                      instrument_name=inst["instrument_name"], start_timestamp=t0 - 6 * 3600_000,
                      end_timestamp=t0 + 6 * 3600_000, count=1000)["trades"]
        if not trades:
            continue
        tr = min(trades, key=lambda t: abs(t["timestamp"] - t0))   # nearest to 08:00 UTC
        out.append({"start": start, "expiry": expiry, "strike": inst["strike"], "S0": S0,
                    "premium_usd": tr["mark_price"] * tr["index_price"], "iv": tr["iv"],
                    "instrument": inst["instrument_name"]})
        time.sleep(0.1)
    df = pd.DataFrame(out)
    df.to_csv(cache, index=False)
    return df
