"""Run all Monte Carlo experiments and save plots to outputs/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mc.paths import gbm_paths
from mc.pricing import black_scholes, mc_european, mc_asian, mc_barrier_up_and_out
from mc.risk import portfolio_var

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0


def fmt(res):
    lo, hi = res["ci95"]
    return f"{res['price']:.4f}  (SE {res['std_err']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# 1. Sample paths
section("1. GBM sample paths")
paths = gbm_paths(S0, r, sigma, T, 252, 50, seed=SEED)
plt.figure(figsize=(9, 5))
plt.plot(np.linspace(0, T, 253), paths.T, lw=0.8, alpha=0.7)
plt.title("50 simulated GBM paths (S0=100, σ=20%, r=5%)")
plt.xlabel("Time (years)"); plt.ylabel("Price")
plt.tight_layout(); plt.savefig(OUT / "gbm_paths.png", dpi=130); plt.close()
print("saved outputs/gbm_paths.png")

# 2. European options vs Black-Scholes
section("2. European options: Monte Carlo vs Black-Scholes")
for kind in ("call", "put"):
    bs = black_scholes(S0, K, r, sigma, T, kind)
    plain = mc_european(S0, K, r, sigma, T, 200_000, kind, seed=SEED)
    anti = mc_european(S0, K, r, sigma, T, 200_000, kind, antithetic=True, seed=SEED)
    print(f"{kind.upper():4s} Black-Scholes : {bs:.4f}")
    print(f"     MC plain      : {fmt(plain)}")
    print(f"     MC antithetic : {fmt(anti)}")

# 3. Convergence plot
ns = np.logspace(2, 6, 20).astype(int)
est, se = [], []
for n in ns:
    res = mc_european(S0, K, r, sigma, T, n, seed=SEED)
    est.append(res["price"]); se.append(res["std_err"])
est, se = np.array(est), np.array(se)
bs_call = black_scholes(S0, K, r, sigma, T)
plt.figure(figsize=(9, 5))
plt.semilogx(ns, est, "o-", label="MC estimate")
plt.fill_between(ns, est - 1.96 * se, est + 1.96 * se, alpha=0.25, label="95% CI")
plt.axhline(bs_call, color="k", ls="--", label=f"Black-Scholes = {bs_call:.4f}")
plt.title("Convergence of MC price for a European call (error ∝ 1/√N)")
plt.xlabel("Number of paths"); plt.ylabel("Price"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "convergence.png", dpi=130); plt.close()
print("saved outputs/convergence.png")

# 4. Exotics
section("3. Path-dependent options")
a_plain = mc_asian(S0, K, r, sigma, T, 252, 50_000, control_variate=False, seed=SEED)
a_cv = mc_asian(S0, K, r, sigma, T, 252, 50_000, control_variate=True, seed=SEED)
print(f"Asian call (plain)           : {fmt(a_plain)}")
print(f"Asian call (control variate) : {fmt(a_cv)}")
print(f"  -> variance reduction factor ≈ {(a_plain['std_err'] / a_cv['std_err'])**2:.0f}x")
barrier = mc_barrier_up_and_out(S0, K, 130, r, sigma, T, 252, 100_000, seed=SEED)
print(f"Up-and-out call (B=130)      : {fmt(barrier)}")

# 5. Portfolio VaR
section("4. Portfolio risk: 10-day VaR / ES at 99%")
S0s = [150.0, 300.0, 50.0]
shares = [100, 50, 400]
mu = [0.08, 0.10, 0.06]
vols = [0.25, 0.30, 0.20]
corr = [[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]]
risk = portfolio_var(S0s, shares, mu, vols, corr, 10 / 252, 500_000, seed=SEED)
print(f"Portfolio value : {risk['V0']:,.0f}")
print(f"99% VaR         : {risk['VaR']:,.0f}  ({risk['VaR'] / risk['V0']:.2%})")
print(f"99% ES (CVaR)   : {risk['ES']:,.0f}  ({risk['ES'] / risk['V0']:.2%})")
plt.figure(figsize=(9, 5))
plt.hist(risk["pnl"], bins=200, density=True, alpha=0.7)
plt.axvline(-risk["VaR"], color="r", ls="--", label=f"VaR 99% = {risk['VaR']:,.0f}")
plt.axvline(-risk["ES"], color="darkred", label=f"ES 99% = {risk['ES']:,.0f}")
plt.title("Simulated 10-day portfolio P&L distribution")
plt.xlabel("P&L"); plt.ylabel("Density"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "portfolio_pnl.png", dpi=130); plt.close()
print("saved outputs/portfolio_pnl.png")
