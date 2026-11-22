# Monte Carlo Simulation for Quantitative Finance

| Module | Contents |
|---|---|
| `mc/paths.py` | Geometric Brownian Motion paths, correlated assets (Cholesky) |
| `mc/pricing.py` | European (checked against Black-Scholes), Asian, up-and-out barrier options |
| `mc/risk.py` | Portfolio Value-at-Risk and Expected Shortfall |

Variance reduction: antithetic variates, and a geometric-Asian control variate.

## Run
```bash
pip install numpy scipy matplotlib
python main.py
```
Plots are saved to `outputs/`.

## Ideas to extend
- Heston stochastic volatility / jump diffusion (Merton)
- American options via Longstaff–Schwartz regression
- Greeks via pathwise or likelihood-ratio estimators
- Quasi-Monte Carlo (Sobol sequences)
