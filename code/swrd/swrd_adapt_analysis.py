"""Analysis of the SWRD adaptation experiments (protocol: swrd_adapt_protocol.md)."""
import json, math, os, numpy as np
from sklearn.metrics import f1_score
W = "./"; P = "../latex/"
S = np.load(W + "swrd/swrd_tiles80.npz"); Y = S["y"]; IMG = S["img"]
sp = json.load(open(W + "swrd/swrd_split.json"))
te = np.isin(IMG, sp["test"]); pool = np.array(sp["pool"])
RIA = np.load(W + "riawelc_80.npz"); YR = RIA["y"]
PR = np.load(W + "swrd/swrd_probs.npz")
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
NAMES = dict(cfg["names"]); NAMES.update({"prismr+paug": "PRISM-Net", "resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-S",
                                         "shufflenet_v2_x1_0": "ShuffleNetV2", "efficientnet_b0": "EfficientNet-B0", "vit": "ViT", "gfnet": "GFNet"})
EPS = 0.10


def qhat(p, y):
    n = len(y); s = 1 - p[np.arange(n), y]
    return np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - EPS)) / n), method="higher")


def setstats(p, y, q):
    sets = p >= 1 - q
    return sets[np.arange(len(y)), y].mean(), sets.sum(1).mean(), (sets.sum(1) == 1).mean()


def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


# ---- A. conformal recalibration --------------------------------------------------------------
rng = np.random.default_rng(0); NC = [25, 50, 100, 200]; REP = 200
draws = {n: [rng.choice(pool, n, replace=False) for _ in range(REP)] for n in NC}
A = {}
for m in MODELS:
    Pm = PR[m]; res = {}
    cov0 = []
    for f in range(5):
        V = np.load(f"{W}results/{m}_group_f{f}_s0.npz") if os.path.exists(f"{W}results/{m}_group_f{f}_s0.npz") else np.load(f"{W}rerun/{m}_group_f{f}_s0.npz")
        Zv = V["Zva"].astype(np.float64)
        if m.split("+")[0].startswith("prism") and "_ce" not in m:
            a = np.logaddexp(0, Zv) + 1; pv = a / a.sum(1, keepdims=True)
        else:
            e = np.exp(Zv - Zv.max(1, keepdims=True)); pv = e / e.sum(1, keepdims=True)
        cov0.append(setstats(Pm[f][te], Y[te], qhat(pv, YR[V["va_idx"]])))
    res["riawelc"] = np.mean(cov0, 0).tolist()
    for n in NC:
        st = []
        for d in draws[n]:
            c = np.isin(IMG, d)
            for f in range(5):
                st.append(setstats(Pm[f][te], Y[te], qhat(Pm[f][c], Y[c])))
        st = np.array(st); res[n] = st.mean(0).tolist() + [float(np.percentile(st[:, 0], 5))]
    A[m] = res
    print(f"{m:26s} RIAWELC cov {res['riawelc'][0]*100:5.1f} size {res['riawelc'][1]:.2f} | " +
          " | ".join(f"n={n}: cov {res[n][0]*100:5.1f} (5%: {res[n][3]*100:4.1f}) size {res[n][1]:.2f} single {res[n][2]*100:4.1f}" for n in NC))

# ---- B. few-shot fine-tuning ------------------------------------------------------------------
FT = ["prismr+paug", "resnet18", "resnet18+paug"]; BUD = [0, 5, 20]
ims = np.array(sp["test"]); ix = np.searchsorted(np.sort(ims), IMG[te]); cnt = np.bincount(ix).astype(float)
boot = np.random.default_rng(0).integers(0, len(ims), (10000, len(ims)))
B = {}
for b in BUD:
    corr = {}
    for m in FT:
        Ps = [PR[m][f] if b == 0 else np.load(f"{W}swrd/ft/{m}_f{f}_b{b}.npy") for f in range(5)]
        acc = [float((p[te].argmax(1) == Y[te]).mean()) for p in Ps]
        mf1 = [float(f1_score(Y[te], p[te].argmax(1), average="macro")) for p in Ps]
        ec = [float(ece(p[te], Y[te])) for p in Ps]
        cal = np.isin(IMG, sp["adapt_5" if b == 5 else "adapt_20"]) if b else None
        cs = [setstats(p[te], Y[te], qhat(p[cal], Y[cal])) for p in Ps] if b else [(np.nan,) * 3]
        corr[m] = np.mean([(p[te].argmax(1) == Y[te]) for p in Ps], 0)
        B[(m, b)] = dict(acc=acc, mf1=mf1, ece=ec, cov=float(np.mean([c[0] for c in cs])), size=float(np.mean([c[1] for c in cs])))
    for m in FT[1:]:
        ds = np.bincount(ix, weights=corr["prismr+paug"] - corr[m]); d = ds.sum() / cnt.sum()
        bs = ds[boot].sum(1) / cnt[boot].sum(1)
        B[(m, b)]["d"] = float(d); B[(m, b)]["ci"] = np.percentile(bs, [2.5, 97.5]).tolist()
    print(f"budget {b}%: " + " | ".join(f"{m} acc {np.mean(B[(m,b)]['acc'])*100:.1f}±{np.std(B[(m,b)]['acc'],ddof=1)*100:.1f} mF1 {np.mean(B[(m,b)]['mf1'])*100:.1f} ECE {np.mean(B[(m,b)]['ece'])*100:.1f} cov {B[(m,b)]['cov']*100:.1f} size {B[(m,b)]['size']:.2f}" + (f" d {B[(m,b)]['d']*100:+.2f} {np.round(np.array(B[(m,b)]['ci'])*100,2)}" if 'd' in B[(m,b)] else "") for m in FT))
json.dump({"recal": {m: {str(k): v for k, v in r.items()} for m, r in A.items()}, "ft": {f"{m}|{b}": v for (m, b), v in B.items()},
           "n_test_imgs": len(sp["test"]), "n_test_tiles": int(te.sum())}, open(W + "results/swrd_adapt.json", "w"), indent=1)
