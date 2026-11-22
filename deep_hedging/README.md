# Deep Hedging: a neural network that hedges options under transaction costs

The textbook Black-Scholes hedge assumes free, continuous trading and constant volatility.
Real markets have neither. This project trains a neural network to hedge a short
call option in a market calibrated to real SPY and VIX data, with 0.2% trading costs. It
minimises tail risk (CVaR 95%) and is evaluated on both simulated and **real** SPY history.

Based on Buehler, Gonon, Teichmann & Wood, *Deep Hedging* (Quantitative Finance, 2019).

## Pipeline
1. **Data:** daily SPY and VIX since 2005 (yfinance, cached to `data/`).
2. **Calibration:** Heston stochastic volatility model, fitted to the VIX time series
   (κ = 9.8, long-run vol 21%, vol-of-vol 0.65, ρ = −0.75).
3. **Simulation:** 200k Heston paths, 30 daily hedging dates.
4. **Training:** a feed-forward network maps (time left, log-moneyness, current vol, current
   position) to the number of shares held. The loss is CVaR of the hedging loss, via the
   Rockafellar–Uryasev formulation. The learned threshold is the seller's indifference price.
5. **Evaluation:** compared against the Black-Scholes delta hedge and no hedge, on new
   simulated paths and on 1,096 real 30-day SPY windows.

## Results (P&L per option sold, S0 = 100, premium excluded)

**Simulated Heston test set (50k paths)**
| Strategy | Mean | Std | VaR 95 | CVaR 95 |
|---|---|---|---|---|
| No hedge | −2.84 | 3.69 | 10.39 | 12.46 |
| Black-Scholes delta | −3.37 | 0.69 | 4.60 | 5.01 |
| **Deep hedger** | **−3.24** | 0.79 | **4.40** | **4.74** |

**Real SPY, 2005–2026 (1,096 windows)**
| Strategy | Mean | Std | VaR 95 | CVaR 95 |
|---|---|---|---|---|
| No hedge | −2.76 | 3.19 | 8.57 | 11.71 |
| Black-Scholes delta | −2.66 | 1.19 | 4.62 | 6.45 |
| **Deep hedger** | **−2.57** | 1.36 | 4.61 | 6.46 |

### Interpretation
- In the simulated world it was trained for, the agent cuts tail risk (CVaR) by about 5% and
  trading costs by about 4% compared with Black-Scholes. It does this by **trading less**:
  its hedge curve is flatter, and it depends on the current position, a soft "no-trade band".
- On real data, it keeps its cost advantage (the best mean P&L) but its tail risk only
  **ties** Black-Scholes. This is the simulation-to-reality gap: real SPY has jumps and
  crashes that the Heston model underestimates.
- Its standard deviation is slightly higher on purpose: it minimises CVaR, not variance.

## Run
```bash
pip install numpy scipy pandas matplotlib torch "yfinance>=0.2.50"
python main.py      # about 3 min on a laptop CPU
```
Outputs: `outputs/pnl_simulated.png`, `pnl_real_spy.png`, `learned_delta.png`,
`training_loss.png`, `deep_hedger.pt`.

## Next steps
- Add jumps (Bates model) to close the gap on real data
- Recurrent network (LSTM) policy, with the VIX term structure as an extra input
- Hedge with options as well as stock (vega hedging)
- Compare different risk preferences: entropic risk vs CVaR vs variance
