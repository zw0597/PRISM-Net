"""Commissioning chain: PGA-R + AdaBN followed by conformal recalibration on 25 labeled radiographs (protocol addendum)."""
import json, math, numpy as np
W = "./"; P = "../latex/"
S = np.load(W + "swrd/swrd_tiles80.npz"); Y = S["y"]; IMG = S["img"]; sp = json.load(open(W + "swrd/swrd_split.json"))
te = np.isin(IMG, sp["test"]); pool = np.array(sp["pool"])
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
P0 = np.load(W + "swrd/swrd_probs.npz"); PG = np.load(W + "swrd/swrd_probs_pga.npz"); EPS = 0.10


def qhat(p, y):
    n = len(y); s = 1 - p[np.arange(n), y]
    return np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - EPS)) / n), method="higher")


def setstats(p, y, q):
    sets = p >= 1 - q
    return sets[np.arange(len(y)), y].mean(), sets.sum(1).mean(), (sets.sum(1) == 1).mean()


rng = np.random.default_rng(0); NC = [25, 50, 100, 200]; REP = 200
draws = {n: [rng.choice(pool, n, replace=False) for _ in range(REP)] for n in NC}   # identical to swrd_adapt_analysis.py
out = {}
for m in MODELS:
    r = {}
    for tag, Pm in (("frozen", P0[m]), ("chain", PG[f"{m}|R_bn"])):
        st = np.array([setstats(Pm[f][te], Y[te], qhat(Pm[f][c], Y[c])) for d in draws[25] for c in [np.isin(IMG, d)] for f in range(5)])
        r[tag] = dict(cov=float(st[:, 0].mean()), cov5=float(np.percentile(st[:, 0], 5)), size=float(st[:, 1].mean()), single=float(st[:, 2].mean()))
    out[m] = r
    print(f"{m:26s} " + " | ".join(f"{t}: cov {r[t]['cov']*100:.1f} ({r[t]['cov5']*100:.1f}) size {r[t]['size']:.2f} single {r[t]['single']*100:.1f}" for t in r), flush=True)
json.dump(out, open(W + "results/swrd_chain.json", "w"), indent=1)
