"""Deep hedging pipeline:
  1. load SPY + VIX and calibrate a Heston model, plus jumps (Bates model)
  2. for each simulated world (Heston, Bates): simulate paths, train a hedger
  3. compare both hedgers with Black-Scholes delta hedging and no hedging
       a) on fresh simulated paths from the jump world (the harder one)
       b) on REAL historical SPY paths (sliding 30-day windows)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from dh.calibrate import load_market, fit_heston, fit_jumps
from dh.hedging import (DeepHedger, train, agent_deltas, hedge_pnl, bs_delta,
                        bs_price, risk_stats)
from dh.sim import heston

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

S0, K = 100.0, 100.0            # at-the-money call
N_STEPS = 30                    # daily hedging, 30 trading days
T = N_STEPS / 252
COST = 0.002                    # 0.2% proportional transaction cost
ALPHA = 0.95                    # CVaR level
N_TRAIN, N_TEST = 200_000, 50_000


def section(t):
    print(f"\n{'=' * 64}\n{t}\n{'=' * 64}")


def table(results):
    print(f"{'strategy':<22}{'mean':>9}{'std':>9}{'VaR95':>9}{'CVaR95':>9}")
    for name, pnl in results.items():
        r = risk_stats(pnl, ALPHA)
        print(f"{name:<22}{r['mean']:>9.3f}{r['std']:>9.3f}{r['VaR']:>9.3f}{r['CVaR']:>9.3f}")


def compare(S, v, models):
    res = {
        "No hedge": hedge_pnl(S, np.zeros((len(S), N_STEPS)), K, COST),
        "Black-Scholes": hedge_pnl(S, bs_delta(S, v, T, K, N_STEPS), K, COST),
    }
    for name, m in models.items():
        res[name] = hedge_pnl(S, agent_deltas(m, S, v, T, K), K, COST)
    return res


def hist(results, title, fname):
    plt.figure(figsize=(9, 5))
    lo, hi = np.percentile(np.concatenate(list(results.values())), [0.5, 99.9])
    bins = np.linspace(lo, hi, 120)
    for name, pnl in results.items():
        if name != "No hedge":
            plt.hist(pnl, bins=bins, alpha=0.45, density=True, label=name)
    plt.yscale("log")
    plt.title(title); plt.xlabel("Hedging P&L (per option, S0=100)"); plt.ylabel("Density (log scale)")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / fname, dpi=130); plt.close()


# 1. Data + calibration
section("1. Market data and calibration")
mkt = load_market()
print(f"SPY + VIX daily: {mkt.index[0].date()} -> {mkt.index[-1].date()}  ({len(mkt)} days)")
params = fit_heston(mkt)
params.v0 = params.theta          # train around typical, not today's, vol
print(f"Heston: kappa={params.kappa:.2f}  theta={params.theta:.4f} (vol {np.sqrt(params.theta):.1%})  "
      f"xi={params.xi:.2f}  rho={params.rho:.2f}")
jumps = fit_jumps(mkt)
print(f"Jumps:  {jumps.lam:.2f} per year, mean size {jumps.mu_j:+.1%}, std {jumps.sig_j:.1%}")

# 2. Train one hedger per simulated world
worlds = {"Deep hedger (Heston)": None, "Deep hedger (Bates)": jumps}
models = {}
for i, (name, jp) in enumerate(worlds.items()):
    section(f"2.{i + 1} Training {name}")
    torch.manual_seed(0)
    S_tr, v_tr = heston(S0, params, T, N_STEPS, N_TRAIN, seed=1, jumps=jp)
    model = DeepHedger()
    loss = train(model, S_tr, v_tr, T, K, COST, alpha=ALPHA)
    print(f"Indifference price: {model.w.item():.3f}   "
          f"(Black-Scholes: {bs_price(S0, K, T, np.sqrt(params.theta)):.3f})")
    plt.plot(loss, label=name)
    models[name] = model
plt.title("Training loss (CVaR 95% of hedging loss)"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "training_loss.png", dpi=130); plt.close()

# 3a. Simulated test in the jump world
section("3a. Test on NEW simulated paths WITH jumps (Bates)")
S_te, v_te = heston(S0, params, T, N_STEPS, N_TEST, seed=2, jumps=jumps)
sim_res = compare(S_te, v_te, models)
table(sim_res)
hist(sim_res, "Hedging P&L on simulated Bates paths (0.2% costs)", "pnl_simulated.png")

# 3b. Real SPY backtest
section("3b. Backtest on REAL SPY history (30-day windows, step 5 days)")
spy, vix = mkt["spy"].values, (mkt["vix"].values / 100) ** 2
starts = range(0, len(spy) - N_STEPS - 1, 5)
S_real = np.array([spy[s:s + N_STEPS + 1] / spy[s] * S0 for s in starts])
v_real = np.array([vix[s:s + N_STEPS + 1] for s in starts])
print(f"{len(S_real):,} real windows")
real_res = compare(S_real, v_real, models)
table(real_res)
hist(real_res, "Hedging P&L on real SPY windows, 2005-today (0.2% costs)", "pnl_real_spy.png")

# 4. What did the networks learn? Hedge ratio vs moneyness, halfway to maturity
section("4. Learned hedge vs Black-Scholes delta")
i = N_STEPS // 2
grid = np.linspace(85, 115, 200)
vol = np.sqrt(params.theta)
plt.figure(figsize=(9, 5))
with torch.no_grad():
    for name, m in models.items():
        x = torch.tensor(np.column_stack([np.full_like(grid, (N_STEPS - i) / N_STEPS),
                                          np.log(grid / K), np.full_like(grid, vol),
                                          np.full_like(grid, 0.5)]), dtype=torch.float32)
        plt.plot(grid, m.net(x).numpy(), label=name)
tau = (N_STEPS - i) * T / N_STEPS
d1 = (np.log(grid / K) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
plt.plot(grid, norm.cdf(d1), "k--", label="Black-Scholes delta")
plt.title("Hedge ratio halfway to maturity (current position 0.5)")
plt.xlabel("Stock price"); plt.ylabel("Shares held"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "learned_delta.png", dpi=130); plt.close()
print("Plots saved to outputs/")

for name, m in models.items():
    tag = "heston" if "Heston" in name else "bates"
    torch.save(m.state_dict(), OUT / f"deep_hedger_{tag}.pt")
