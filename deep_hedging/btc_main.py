"""Deep hedging on REAL Bitcoin options (Deribit).

  1. download daily BTC prices, DVOL (BTC implied vol index) and 67 real expired
     monthly BTC call options, each with its market price 30 days before expiry
  2. calibrate a Bates model (Heston + jumps) on 2021-2023 ONLY
  3. train a deep hedger in that simulated world (random strikes 94..106). Each path
     starts in a volatility regime drawn from real 2021-23 DVOL levels
     ("domain randomization"), so the agent learns to adapt to calm and wild markets
  4. test, split into the calibration period (2021-23) and the unseen period (2024-26):
       a) every 30-day BTC window (as with SPY, premium excluded)
       b) the real options: sell at the real market price, hedge daily, settle at expiry.
          P&L = premium received + hedge gains - trading costs - payoff
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dh.calibrate import fit_heston, fit_jumps
from dh.deribit import load_btc_market, load_btc_options
from dh.hedging import DeepHedger, train, agent_deltas, hedge_pnl, bs_delta, risk_stats
from dh.sim import heston

OUT = Path(__file__).parent / "outputs" / "btc"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

S0 = 100.0
N_STEPS = 30                 # daily hedging, 30 days (BTC trades every day)
T = N_STEPS / 365
COST = 0.001                 # 0.1% per trade: exchange fee + spread on BTC
ALPHA = 0.95
SPLIT = pd.Timestamp("2024-01-01")   # calibration data ends here
N_TRAIN = 200_000


def section(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def strategies(S, v, K, model):
    return {
        "No hedge": hedge_pnl(S, np.zeros((len(S), N_STEPS)), K, COST),
        "Black-Scholes": hedge_pnl(S, bs_delta(S, v, T, K, N_STEPS), K, COST),
        "Deep hedger": hedge_pnl(S, agent_deltas(model, S, v, T, K), K, COST),
    }


def window(mkt, start):
    seg = mkt.loc[start:start + pd.Timedelta(days=N_STEPS)]
    return seg["btc"].values / seg["btc"].values[0] * S0, (seg["dvol"].values / 100) ** 2


# 1. Data
section("1. Deribit data")
mkt = load_btc_market()
opts = load_btc_options(mkt, days=N_STEPS)
print(f"BTC + DVOL daily: {mkt.index[0].date()} -> {mkt.index[-1].date()}  ({len(mkt)} days)")
print(f"Real monthly options: {len(opts)}  ({opts.start.min().date()} -> {opts.expiry.max().date()})")
print(f"Average market premium: {(opts.premium_usd / opts.S0).mean():.2%} of BTC price, "
      f"average implied vol {opts.iv.mean():.0f}%")

# 2. Calibration on 2021-2023 only
section("2. Bates calibration on 2021-2023 only")
calib = mkt[mkt.index < SPLIT]
hp = fit_heston(calib, price="btc", vol="dvol", per_year=365)
hp.v0 = hp.theta
jp = fit_jumps(calib, price="btc", vol="dvol", per_year=365)
print(f"Heston: kappa={hp.kappa:.2f}  long-run vol {np.sqrt(hp.theta):.0%}  xi={hp.xi:.2f}  rho={hp.rho:.2f}")
print(f"Jumps:  {jp.lam:.2f}/year, mean {jp.mu_j:+.1%}, std {jp.sig_j:.1%}")

# 3. Train
section("3. Training deep hedger (Bates world, random strikes, random vol regimes)")
# Domain randomization: each path's starting and long-run variance are drawn from
# the DVOL values actually seen in 2021-23. No 2024-26 data is used.
rng = np.random.default_rng(4)
hist_var = (calib["dvol"].values / 100) ** 2
hp.v0 = rng.choice(hist_var, N_TRAIN)
hp.theta = rng.choice(hist_var, N_TRAIN)
S_tr, v_tr = heston(S0, hp, T, N_STEPS, N_TRAIN, seed=1, jumps=jp)
K_tr = np.random.default_rng(3).uniform(94, 106, N_TRAIN)
model = DeepHedger()
train(model, S_tr, v_tr, T, K_tr, COST, alpha=ALPHA)
torch.save(model.state_dict(), OUT / "deep_hedger_btc.pt")

# 4a. All 30-day windows, ATM, premium excluded
section("4a. Every 30-day BTC window (at-the-money, premium excluded)")
starts = mkt.index[:-N_STEPS][::3]
paths = [window(mkt, s) for s in starts]
S_w, v_w = np.array([p[0] for p in paths]), np.array([p[1] for p in paths])
res_w = strategies(S_w, v_w, S0, model)
for label, mask in [("2021-23 (calibration period)", starts < SPLIT), ("2024-26 (unseen)", starts >= SPLIT)]:
    print(f"\n{label}: {mask.sum()} windows")
    print(f"{'strategy':<16}{'mean':>9}{'std':>9}{'VaR95':>9}{'CVaR95':>9}")
    for name, pnl in res_w.items():
        r = risk_stats(pnl[mask], ALPHA)
        print(f"{name:<16}{r['mean']:>9.3f}{r['std']:>9.3f}{r['VaR']:>9.3f}{r['CVaR']:>9.3f}")

# 4b. Real options at real prices
section("4b. REAL options: sell at market price, hedge, settle at expiry")
paths = [window(mkt, s) for s in opts.start]
S_o, v_o = np.array([p[0] for p in paths]), np.array([p[1] for p in paths])
K_o = (opts.strike / opts.S0 * S0).values
prem = (opts.premium_usd / opts.S0 * S0).values
res_o = {k: prem + pnl for k, pnl in strategies(S_o, v_o, K_o, model).items()}
for label, mask in [("2021-23", (opts.start < SPLIT).values), ("2024-26 (unseen)", (opts.start >= SPLIT).values),
                    ("All", np.ones(len(opts), bool))]:
    print(f"\n{label}: {mask.sum()} options   (profit per option, % of BTC price)")
    print(f"{'strategy':<16}{'mean':>9}{'std':>9}{'worst':>9}{'win %':>8}")
    for name, pnl in res_o.items():
        p = pnl[mask]
        print(f"{name:<16}{p.mean():>9.2f}{p.std():>9.2f}{p.min():>9.2f}{(p > 0).mean():>8.0%}")

# Plots
plt.figure(figsize=(10, 5))
for name in ("Black-Scholes", "Deep hedger"):
    plt.plot(opts.expiry, np.cumsum(res_o[name]), marker="o", ms=3, label=name)
plt.plot(opts.expiry, np.cumsum(res_o["No hedge"]), color="grey", alpha=0.5, label="No hedge")
plt.axvline(SPLIT, color="k", ls=":", label="calibration ends")
plt.title("Selling one real BTC call each month and hedging it: cumulative profit")
plt.ylabel("Cumulative profit (% of BTC price per option)"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "real_options_cumulative.png", dpi=130); plt.close()

plt.figure(figsize=(9, 5))
plt.scatter(res_o["Black-Scholes"], res_o["Deep hedger"], c=(opts.start >= SPLIT), cmap="coolwarm")
lim = [min(res_o["Black-Scholes"].min(), res_o["Deep hedger"].min()) - 0.5,
       max(res_o["Black-Scholes"].max(), res_o["Deep hedger"].max()) + 0.5]
plt.plot(lim, lim, "k--", lw=1)
plt.xlabel("Black-Scholes hedge profit"); plt.ylabel("Deep hedger profit")
plt.title("Each dot = one real option (red = unseen 2024-26). Above line = AI did better")
plt.tight_layout(); plt.savefig(OUT / "real_options_scatter.png", dpi=130); plt.close()
print("\nPlots saved to outputs/btc/")
