"""External test of the frozen RIAWELC models on SWRD tiles (protocol: swrd_protocol.md)."""
import json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
sys.path.insert(0, ".")
from models import build
K = 4
def probs(Z, lt, T=1.0):
    Z = torch.from_numpy(Z).double()
    if lt == "edl":
        a = F.softplus(Z) + 1.0
        return (a / a.sum(1, keepdim=True)).numpy(), (K / a.sum(1)).numpy()
    p = F.softmax(Z / T, 1).numpy()
    return p, 1 - p.max(1)


def ece(p, y, B=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y)
    edges = np.linspace(0, 1, B + 1); e = 0.0; rel = []
    for i in range(B):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
            rel.append((float(conf[m].mean()), float(acc[m].mean()), int(m.sum())))
    return e, rel


def aurc(conf, correct):
    o = np.argsort(-conf, kind="stable"); c = correct[o]
    risk = np.cumsum(1 - c) / np.arange(1, len(c) + 1)
    return risk.mean(), risk


def conformal(pv, yv, pt, yt, alpha):
    n = len(yv)
    s = 1 - pv[np.arange(n), yv]
    q = np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = pt >= 1 - q
    cov = sets[np.arange(len(yt)), yt].mean()
    return float(cov), float(sets.sum(1).mean()), float((sets.sum(1) == 1).mean())



torch.set_num_threads(4)
W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); XS = torch.from_numpy(S["X"]).unsqueeze(1).float() / 255.0
YS = S["y"]; IMG = S["img"]
RIA = np.load(W + "riawelc_80.npz"); YR = RIA["y"]
cfg = json.load(open(W + "paper_cfg.json"))
MODELS = list(dict.fromkeys(cfg["main"] + cfg["paug_baselines"] + [m for m, _ in cfg["ablation"]] + ["prism+paug"]))


def wpath(m, f):
    p = f"{W}results/{m}_group_f{f}_s0.pt"
    return p if os.path.exists(p) else f"{W}rerun/{m}_group_f{f}_s0.pt"


@torch.no_grad()
def run(m):
    base = m.split("+")[0]
    lt = "edl" if base.startswith("prism") and "_ce" not in base else "ce"
    out = {"P": [], "acc": [], "mf1": [], "rec": [], "ece": [], "nll": [], "cov": [], "size": []}
    for f in range(5):
        net = build(base, 80); net.load_state_dict(torch.load(wpath(m, f), map_location="cpu")); net.eval()
        Z = torch.cat([net(XS[k:k + 512]) for k in range(0, len(XS), 512)]).numpy()
        p, _ = probs(Z, lt); pred = p.argmax(1)
        V = np.load(f"{W}results/{m}_group_f{f}_s0.npz") if os.path.exists(f"{W}results/{m}_group_f{f}_s0.npz") else np.load(f"{W}rerun/{m}_group_f{f}_s0.npz")
        pv, _ = probs(V["Zva"], lt)
        cov, size, _ = conformal(pv, YR[V["va_idx"]], p, YS, 0.10)
        out["P"].append(p); out["acc"].append(float((pred == YS).mean()))
        out["mf1"].append(float(f1_score(YS, pred, average="macro")))
        out["rec"].append(recall_score(YS, pred, average=None, labels=range(4)).tolist())
        out["ece"].append(float(ece(p, YS)[0])); out["nll"].append(float(-np.log(np.clip(p[np.arange(len(YS)), YS], 1e-12, 1)).mean()))
        out["cov"].append(cov); out["size"].append(size)
    out["P"] = np.stack(out["P"])
    out["ens_acc"] = float((out["P"].mean(0).argmax(1) == YS).mean())
    out["corr"] = (out["P"].argmax(2) == YS[None]).mean(0)          # per tile, averaged over fold models
    return out


# incremental: models already in swrd_external.json keep their stored results; new ones are evaluated
OLD = json.load(open(W + "results/swrd_external.json")) if os.path.exists(W + "results/swrd_external.json") else {"models": {}, "cmp": {}}
REF = "prismr+paug"; NEW = [m for m in MODELS if m not in OLD["models"]]
ims = np.unique(IMG); ix = np.searchsorted(ims, IMG)
rng = np.random.default_rng(0); B = rng.integers(0, len(ims), (10000, len(ims)))
cnt = np.bincount(ix, minlength=len(ims)).astype(float)
R = {m: run(m) for m in ([REF] if NEW else []) + NEW}
if NEW: assert abs(np.mean(R[REF]["acc"]) - np.mean(OLD["models"][REF]["acc"])) < 1e-9 if REF in OLD["models"] else True
for m in NEW:
    r = R[m]
    print(f"{m:28s} acc {np.mean(r['acc'])*100:5.1f}±{np.std(r['acc'],ddof=1)*100:4.1f}  mF1 {np.mean(r['mf1'])*100:5.1f}  "
          f"ens {r['ens_acc']*100:5.1f}  rec {np.round(np.mean(r['rec'],0)*100,1)}  ECE {np.mean(r['ece'])*100:4.1f}  cov {np.mean(r['cov'])*100:4.1f} size {np.mean(r['size']):.2f}", flush=True)
    if m == REF: continue
    dsum = np.bincount(ix, weights=R[REF]["corr"] - R[m]["corr"], minlength=len(ims))
    d = dsum.sum() / cnt.sum(); bs = dsum[B].sum(1) / cnt[B].sum(1)
    OLD["cmp"][m] = dict(d=float(d), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    print(f"PRISM-Net - {m:26s} {d*100:+.2f} [{OLD['cmp'][m]['lo']*100:+.2f}, {OLD['cmp'][m]['hi']*100:+.2f}]")
for m in NEW: OLD["models"][m] = {k: v for k, v in R[m].items() if k not in ("P", "corr")}
OLD["models"] = {m: OLD["models"][m] for m in MODELS}
OLD.update(n_tiles=int(len(YS)), n_images=int(len(ims)), class_counts=np.bincount(YS).tolist())
json.dump(OLD, open(W + "results/swrd_external.json", "w"), indent=1)
