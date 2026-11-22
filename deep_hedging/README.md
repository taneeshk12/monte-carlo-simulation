# Deep Hedging: a neural network that hedges options under transaction costs

The textbook Black-Scholes hedge assumes free, continuous trading and constant volatility.
Real markets have neither. This project trains a neural network to hedge a short
call option in a market calibrated to real SPY and VIX data, with 0.2% trading costs. It
minimises tail risk (CVaR 95%) and is evaluated on both simulated and **real** SPY history.

Based on Buehler, Gonon, Teichmann & Wood, *Deep Hedging* (Quantitative Finance, 2019).

## Pipeline
1. **Data:** daily SPY and VIX since 2005 (yfinance, cached to `data/`).
2. **Calibration:** Heston stochastic volatility model, fitted to the VIX time series
   (κ = 9.8, long-run vol 21%, vol-of-vol 0.65, ρ = −0.75), plus jumps for the Bates model.
3. **Simulation:** 200k paths per world (Heston, and Bates with jumps), 30 daily hedging dates.
4. **Training:** a feed-forward network maps (time left, log-moneyness, current vol, current
   position) to the number of shares held. The loss is CVaR of the hedging loss, via the
   Rockafellar–Uryasev formulation. The learned threshold is the seller's indifference price.
5. **Evaluation:** compared against the Black-Scholes delta hedge and no hedge, on new
   simulated paths and on 1,096 real 30-day SPY windows.

## Results (P&L per option sold, S0 = 100, premium excluded)

Two hedgers are trained: one in a Heston world (smooth moves only) and one in a **Bates**
world (Heston plus sudden jumps). Jumps are calibrated from SPY: about 1 per year, mean −3.0%,
std 5.6%. A "jump" is a day whose move is more than 3× what the VIX predicted.

**Simulated test set with jumps (50k Bates paths)**
| Strategy | Mean | Std | VaR 95 | CVaR 95 |
|---|---|---|---|---|
| No hedge | −2.96 | 3.87 | 10.82 | 13.18 |
| Black-Scholes delta | −3.49 | 0.90 | 4.91 | 6.02 |
| Deep hedger (Heston) | −3.36 | 0.91 | **4.65** | 5.52 |
| **Deep hedger (Bates)** | **−3.32** | 1.01 | 4.72 | **5.45** |

**Real SPY, 2005–2026 (1,096 windows)**
| Strategy | Mean | Std | VaR 95 | CVaR 95 |
|---|---|---|---|---|
| No hedge | −2.76 | 3.19 | 8.57 | 11.71 |
| Black-Scholes delta | −2.66 | 1.19 | 4.62 | 6.45 |
| Deep hedger (Heston) | −2.57 | 1.36 | 4.61 | 6.46 |
| **Deep hedger (Bates)** | **−2.55** | 1.40 | **4.27** | **6.42** |

### Interpretation
- Both hedgers save money by **trading less** than Black-Scholes. Their hedge is flatter
  (see `learned_delta.png`), which cuts transaction costs.
- On simulated jump paths, both beat Black-Scholes on tail risk (CVaR −9% to −10%).
- On real SPY, adding jumps to training gives the **best result on every measure except
  std**: VaR improves by 8% over Black-Scholes. CVaR improves only slightly; the
  Heston-trained hedger just ties Black-Scholes there.
- **Caveat:** the real test has about 55 tail windows, and the windows overlap, so small
  CVaR differences are within noise. The gap left on real data suggests jumps alone
  don't explain real crashes: volatility clustering and regime changes matter too.
- Std is slightly higher on purpose: the agent minimises CVaR, not variance.

## Run
```bash
pip install numpy scipy pandas matplotlib torch "yfinance>=0.2.50"
python main.py      # about 6 min on a laptop CPU
```
Outputs: `outputs/pnl_simulated.png`, `pnl_real_spy.png`, `learned_delta.png`,
`training_loss.png`, `deep_hedger_heston.pt`, `deep_hedger_bates.pt`.

## Next steps
- Test on real traded options (Deribit BTC options)
- Generative model (GAN or diffusion) trained on real paths as the simulator
- Recurrent network (LSTM) policy, with the VIX term structure as an extra input
- Hedge with options as well as stock (vega hedging)
- Compare different risk preferences: entropic risk vs CVaR vs variance
