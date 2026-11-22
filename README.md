# Quant Projects

| Project | What it does |
|---|---|
| [monte_carlo](monte_carlo/) | Monte Carlo engine: option pricing (checked against Black-Scholes), Asian and barrier options, variance reduction, portfolio VaR/ES |
| [deep_hedging](deep_hedging/) | A neural network that learns to hedge options under transaction costs. Trained on a Heston model calibrated to SPY/VIX, then backtested on real SPY history |

Each folder has its own README with results. Run either with `python main.py` from inside the folder.
