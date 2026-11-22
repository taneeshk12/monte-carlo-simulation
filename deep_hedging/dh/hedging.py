"""The deep hedging agent, the P&L engine and the Black-Scholes baseline.

Setup: we have SOLD one call option (strike K, maturity T). At each hedging date
we choose how many shares (delta) to hold. Trading costs `cost * |trade| * S`.

    Hedging P&L = sum_t delta_t (S_{t+1} - S_t)  -  costs  -  payoff(S_T)

The agent minimises CVaR (the average of the worst (1-alpha) outcomes) of
-P&L, using the Rockafellar-Uryasev trick: risk = w + E[(loss - w)^+]/(1-alpha),
with w learnt alongside the network. The optimal w is the minimum premium the
seller needs to charge: the "indifference price".
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm


def features(S, v, i, n_steps, T, K, prev_delta):
    """Network input at hedging date i: time left, moneyness, vol, current position."""
    tau = torch.full_like(S[:, i], (n_steps - i) * T / n_steps)
    return torch.stack([tau / T, torch.log(S[:, i] / K), torch.sqrt(v[:, i]), prev_delta], dim=1)


class DeepHedger(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),  # a short call is hedged with 0..1 shares
        )
        self.w = nn.Parameter(torch.tensor(0.0))  # CVaR threshold, learned

    def forward(self, S, v, T, K):
        n_steps = S.shape[1] - 1
        delta = torch.zeros_like(S[:, 0])
        deltas = []
        for i in range(n_steps):
            delta = self.net(features(S, v, i, n_steps, T, K, delta)).squeeze(1)
            deltas.append(delta)
        return torch.stack(deltas, dim=1)  # (n_paths, n_steps)


def hedge_pnl(S, deltas, K, cost):
    """Works with torch tensors or numpy arrays. Position is closed at maturity."""
    lib = torch if isinstance(S, torch.Tensor) else np
    dS = S[:, 1:] - S[:, :-1]
    gains = (deltas * dS).sum(1)
    zeros = lib.zeros_like(deltas[:, :1])
    pos = lib.concatenate([zeros, deltas, zeros], 1) if lib is np else torch.cat([zeros, deltas, zeros], 1)
    trades = lib.abs(pos[:, 1:] - pos[:, :-1])
    costs = cost * (trades * S).sum(1)  # trade k happens at date k, price S[:, k]
    payoff = lib.maximum(S[:, -1] - K, lib.zeros_like(S[:, -1])) if lib is np else torch.relu(S[:, -1] - K)
    return gains - costs - payoff


def cvar_loss(pnl, w, alpha):
    return w + torch.relu(-pnl - w).mean() / (1 - alpha)


def bs_delta(S, v, T, K, n_steps, r=0.0):
    """Black-Scholes delta at each hedging date, using the current vol sqrt(v)."""
    out = np.empty((S.shape[0], n_steps))
    for i in range(n_steps):
        tau = (n_steps - i) * T / n_steps
        sig = np.sqrt(np.maximum(v[:, i], 1e-8))
        d1 = (np.log(S[:, i] / K) + (r + 0.5 * sig**2) * tau) / (sig * np.sqrt(tau))
        out[:, i] = norm.cdf(d1)
    return out


def bs_price(S0, K, T, sigma):
    d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S0 * norm.cdf(d1) - K * norm.cdf(d1 - sigma * np.sqrt(T))


def train(model, S, v, T, K, cost, alpha=0.95, epochs=40, batch=4096, lr=1e-3, log=print):
    S, v = torch.tensor(S, dtype=torch.float32), torch.tensor(v, dtype=torch.float32)
    # w (CVaR threshold) is a price-sized number: start it at the loss quantile of
    # the untrained policy and give it a faster learning rate than the network.
    with torch.no_grad():
        idx = torch.randperm(S.shape[0])[:batch]
        pnl0 = hedge_pnl(S[idx], model(S[idx], v[idx], T, K), K, cost)
        model.w.fill_(torch.quantile(-pnl0, alpha).item())
    opt = torch.optim.Adam([{"params": model.net.parameters(), "lr": lr},
                            {"params": [model.w], "lr": 0.05}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = S.shape[0]
    history = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for j in range(0, n, batch):
            idx = perm[j:j + batch]
            pnl = hedge_pnl(S[idx], model(S[idx], v[idx], T, K), K, cost)
            loss = cvar_loss(pnl, model.w, alpha)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        sched.step()
        history.append(total / n)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d}  CVaR loss {history[-1]:.4f}")
    return history


@torch.no_grad()
def agent_deltas(model, S, v, T, K):
    S_t, v_t = torch.tensor(S, dtype=torch.float32), torch.tensor(v, dtype=torch.float32)
    return model(S_t, v_t, T, K).numpy().astype(float)


def risk_stats(pnl, alpha=0.95):
    q = np.quantile(pnl, 1 - alpha)
    return {"mean": pnl.mean(), "std": pnl.std(), "VaR": -q, "CVaR": -pnl[pnl <= q].mean()}
