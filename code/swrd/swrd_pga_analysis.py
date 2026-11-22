"""Analysis of PGA (zero-label) on the SWRD test half; also picks the PGA variant on V for few-shot runs."""
import json, numpy as np
from sklearn.metrics import f1_score
W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); Y = S["y"]; IMG = S["img"]
sp = json.load(open(W + "swrd/swrd_split.json")); te = np.isin(IMG, sp["test"]); V = np.isin(IMG, sp["V"])
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
P0 = np.load(W + "swrd/swrd_probs.npz"); PA = np.load(W + "swrd/swrd_probs_adabn.npz"); PG = np.load(W + "swrd/swrd_probs_pga.npz")
ims = np.sort(np.array(sp["test"])); ix = np.searchsorted(ims, IMG[te]); cnt = np.bincount(ix).astype(float)
boot = np.random.default_rng(0).integers(0, len(ims), (10000, len(ims)))


def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def summ(P):
    return dict(acc=[float((p[te].argmax(1) == Y[te]).mean()) for p in P], mf1=[float(f1_score(Y[te], p[te].argmax(1), average="macro")) for p in P],
                ece=[float(ece(p[te], Y[te])) for p in P], corr=np.mean([p[te].argmax(1) == Y[te] for p in P], 0),
                vacc=float(np.mean([(p[V].argmax(1) == Y[V]).mean() for p in P])))


def diff(ca, cb):
    ds = np.bincount(ix, weights=ca - cb); bs = ds[boot].sum(1) / cnt[boot].sum(1)
    return float(ds.sum() / cnt.sum()), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


VARS = ["frozen", "S", "R", "bn", "S_bn", "R_bn"]
R = {}
for m in MODELS:
    R[m] = {"frozen": summ(list(P0[m])), "bn": summ(list(PA[m]))}
    for v in ("S", "R", "S_bn", "R_bn"): R[m][v] = summ(list(PG[f"{m}|{v}"]))
out = {"params": json.load(open(W + "swrd/pga_params.json")), "models": {}}
for m in MODELS:
    o = {}
    for v in VARS:
        r = R[m][v]; o[v] = dict(acc=float(np.mean(r["acc"])), std=float(np.std(r["acc"], ddof=1)), mf1=float(np.mean(r["mf1"])), ece=float(np.mean(r["ece"])), vacc=r["vacc"])
        if v != "frozen": o[v]["gain"], o[v]["gain_ci"] = diff(r["corr"], R[m]["frozen"]["corr"])
        if m != "prismr+paug": o[v]["d"], o[v]["d_ci"] = diff(R["prismr+paug"][v]["corr"], r["corr"])
    out["models"][m] = o
    print(f"{m:26s} " + " | ".join(f"{v}: {o[v]['acc']*100:5.1f}" + (f" ({o[v]['gain']*100:+.1f})" if v != 'frozen' else "") for v in VARS))
FT = ["prismr+paug", "resnet18", "resnet18+paug"]
vS = np.mean([R[m]["S"]["vacc"] for m in FT]); vR = np.mean([R[m]["R"]["vacc"] for m in FT])
out["ft_variant"] = "S" if vS >= vR else "R"; out["v_acc"] = dict(S=float(vS), R=float(vR))
print("V accuracy (frozen, 3 models): S", vS, "R", vR, "->", out["ft_variant"])
for v in VARS:
    best = max(MODELS, key=lambda m: out["models"][m][v]["acc"]); lo = min(out["models"][m][v]["d_ci"][0] for m in MODELS if m != "prismr+paug")
    print(v, "best:", best, "min lower CI of PRISM-Net lead:", round(lo * 100, 2))
json.dump(out, open(W + "results/swrd_pga.json", "w"), indent=1)
