"""Round-2 adaptation analysis on the SWRD test half (protocol: swrd_adapt2_protocol.md)."""
import json, math, numpy as np
from sklearn.metrics import f1_score
W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); Y = S["y"]; IMG = S["img"]
sp = json.load(open(W + "swrd/swrd_split.json")); te = np.isin(IMG, sp["test"])
sel = json.load(open(W + "swrd/round2_selection.json"))
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
FT = ["prismr+paug", "resnet18", "resnet18+paug"]
ims = np.sort(np.array(sp["test"])); ix = np.searchsorted(ims, IMG[te]); cnt = np.bincount(ix).astype(float)
boot = np.random.default_rng(0).integers(0, len(ims), (10000, len(ims)))


def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def summarize(Ps):
    acc = [float((p[te].argmax(1) == Y[te]).mean()) for p in Ps]
    return dict(acc=acc, mf1=[float(f1_score(Y[te], p[te].argmax(1), average="macro")) for p in Ps],
                ece=[float(ece(p[te], Y[te])) for p in Ps], corr=np.mean([p[te].argmax(1) == Y[te] for p in Ps], 0))


def diff(ca, cb):
    ds = np.bincount(ix, weights=ca - cb); bs = ds[boot].sum(1) / cnt[boot].sum(1)
    return float(ds.sum() / cnt.sum()), np.percentile(bs, [2.5, 97.5]).tolist()


OUT = {"selection": sel}
# zero-label AdaBN
PA = np.load(W + "swrd/swrd_probs_adabn.npz"); P0 = np.load(W + "swrd/swrd_probs.npz")
ZA = {m: summarize(list(PA[m])) for m in MODELS}; Z0 = {m: summarize(list(P0[m])) for m in MODELS}
OUT["adabn"] = {}
for m in MODELS:
    d, ci = diff(ZA["prismr+paug"]["corr"], ZA[m]["corr"]) if m != "prismr+paug" else (0.0, [0, 0])
    OUT["adabn"][m] = dict(acc0=np.mean(Z0[m]["acc"]), acc=np.mean(ZA[m]["acc"]), std=float(np.std(ZA[m]["acc"], ddof=1)),
                           mf1=np.mean(ZA[m]["mf1"]), ece=np.mean(ZA[m]["ece"]), ece0=np.mean(Z0[m]["ece"]), d=d, ci=ci)
    print(f"AdaBN {m:24s} {np.mean(Z0[m]['acc'])*100:5.1f} -> {np.mean(ZA[m]['acc'])*100:5.1f}±{np.std(ZA[m]['acc'],ddof=1)*100:.1f}  ECE {np.mean(Z0[m]['ece'])*100:4.1f} -> {np.mean(ZA[m]['ece'])*100:4.1f}  d {d*100:+.2f} {np.round(np.array(ci)*100,2)}")
# fine-tuning, round 1 (C0) and round 2 (selected)
OUT["ft"] = {}
for b in (5, 20):
    R = {}
    for m in FT:
        r = sel[f"{m}|{b}"]; suf = "" if r == "C0" else f"_{r}"
        R[m] = summarize([np.load(f"{W}swrd/ft/{m}_f{f}_b{b}{suf}.npy") for f in range(5)])
        R0 = summarize([np.load(f"{W}swrd/ft/{m}_f{f}_b{b}.npy") for f in range(5)])
        OUT["ft"][f"{m}|{b}"] = dict(recipe=r, acc=R[m]["acc"], mf1=R[m]["mf1"], ece=R[m]["ece"], acc_r1=R0["acc"], ece_r1=R0["ece"])
    for m in FT[1:]:
        d, ci = diff(R["prismr+paug"]["corr"], R[m]["corr"]); OUT["ft"][f"{m}|{b}"].update(d=d, ci=ci)
    print(f"budget {b}%: " + " | ".join(f"{m} [{sel[f'{m}|{b}']}] acc {np.mean(R[m]['acc'])*100:.1f}±{np.std(R[m]['acc'],ddof=1)*100:.1f} mF1 {np.mean(R[m]['mf1'])*100:.1f} ECE {np.mean(R[m]['ece'])*100:.1f}"
                                         + (f" d {OUT['ft'][f'{m}|{b}']['d']*100:+.2f} {np.round(np.array(OUT['ft'][f'{m}|{b}']['ci'])*100,2)}" if m != 'prismr+paug' else "") for m in FT))
json.dump(OUT, open(W + "results/swrd_adapt2.json", "w"), indent=1, default=float)
