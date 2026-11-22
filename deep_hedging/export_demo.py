"""Build demo/index.html: inline the trained SPY (Bates) hedger's weights and the
real-BTC-option results into demo/template.html. Run after main.py and btc_main.py."""
import json
from pathlib import Path

import numpy as np
import torch

from dh.deribit import load_btc_market, load_btc_options
from dh.hedging import DeepHedger, agent_deltas, hedge_pnl, bs_delta
from dh.calibrate import load_market, fit_heston, fit_jumps

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"


def weights(model):
    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    return [{"W": l.weight.detach().numpy().round(5).tolist(),
             "b": l.bias.detach().numpy().round(5).tolist()} for l in layers]


spy_model = DeepHedger()
spy_model.load_state_dict(torch.load(OUT / "deep_hedger_bates.pt"))
mkt = load_market()
hp, jp = fit_heston(mkt), fit_jumps(mkt)

# Real BTC options, same setup as btc_main.py
btc_model = DeepHedger()
btc_model.load_state_dict(torch.load(OUT / "btc" / "deep_hedger_btc.pt"))
bm = load_btc_market()
opts = load_btc_options(bm, days=30)
N, T, COST = 30, 30 / 365, 0.001
S, V = [], []
for s in opts.start:
    seg = bm.loc[s:s + np.timedelta64(N, "D")]
    S.append(seg.btc.values / seg.btc.values[0] * 100); V.append((seg.dvol.values / 100) ** 2)
S, V = np.array(S), np.array(V)
K = (opts.strike / opts.S0 * 100).values
prem = (opts.premium_usd / opts.S0 * 100).values
res = {
    "none": prem + hedge_pnl(S, np.zeros((len(S), N)), K, COST),
    "bs": prem + hedge_pnl(S, bs_delta(S, V, T, K, N), K, COST),
    "ai": prem + hedge_pnl(S, agent_deltas(btc_model, S, V, T, K), K, COST),
}
btc = [{"d": opts.expiry[i].strftime("%Y-%m"), "k": opts.instrument[i], "p": round(prem[i], 2),
        "iv": opts.iv[i], **{k: round(float(v[i]), 2) for k, v in res.items()}} for i in range(len(opts))]

data = {
    "net": weights(spy_model),
    "heston": {"kappa": hp.kappa, "theta": hp.theta, "xi": hp.xi, "rho": hp.rho},
    "jumps": {"lam": jp.lam, "mu": jp.mu_j, "sig": jp.sig_j},
    "btc": btc,
}
tpl = (ROOT / "demo" / "template.html").read_text()
(ROOT / "demo" / "index.html").write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
print("wrote demo/index.html", len(btc), "BTC options")
