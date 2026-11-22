"""R1 analysis: PGA-R against generic per-radiograph standardisation (ZS-R, HM-R), alone and with AdaBN."""
import json, numpy as np
W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); Y = S["y"]; IMG = S["img"]
sp = json.load(open(W + "swrd/swrd_split.json")); te = np.isin(IMG, sp["test"])
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
P0 = np.load(W + "swrd/swrd_probs.npz"); PG = np.load(W + "swrd/swrd_probs_pga.npz"); PS = np.load(W + "swrd/swrd_probs_std.npz")
ims = np.sort(np.array(sp["test"])); ix = np.searchsorted(ims, IMG[te]); cnt = np.bincount(ix).astype(float)
boot = np.random.default_rng(0).integers(0, len(ims), (10000, len(ims)))
corr = lambda P: np.mean([p[te].argmax(1) == Y[te] for p in P], 0)


def diff(ca, cb):
    ds = np.bincount(ix, weights=ca - cb); bs = ds[boot].sum(1) / cnt[boot].sum(1)
    return float(ds.sum() / cnt.sum()), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


out = {}; C = {}
for m in MODELS:
    c = {"frozen": corr(P0[m]), "PGA": corr(PG[f"{m}|R"]), "PGA_bn": corr(PG[f"{m}|R_bn"])}
    for v in ("ZS", "HM", "ZS_bn", "HM_bn"): c[v] = corr(PS[f"{m}|{v}"])
    C[m] = c; o = {v: dict(acc=float(c[v].mean())) for v in c}
    for b in ("ZS", "HM"):
        o[b]["d"], o[b]["d_ci"] = diff(c["PGA"], c[b]); o[b + "_bn"]["d"], o[b + "_bn"]["d_ci"] = diff(c["PGA_bn"], c[b + "_bn"])
        o[b]["gain"], o[b]["gain_ci"] = diff(c[b], c["frozen"]); o[b + "_bn"]["gain"], o[b + "_bn"]["gain_ci"] = diff(c[b + "_bn"], c["frozen"])
    out[m] = o
    print(f"{m:26s} frozen {o['frozen']['acc']*100:5.1f} | ZS {o['ZS']['acc']*100:5.1f} HM {o['HM']['acc']*100:5.1f} PGA {o['PGA']['acc']*100:5.1f} | "
          f"+BN: ZS {o['ZS_bn']['acc']*100:5.1f} HM {o['HM_bn']['acc']*100:5.1f} PGA {o['PGA_bn']['acc']*100:5.1f} | "
          f"PGA-ZS {o['ZS']['d']*100:+.1f} PGA-HM {o['HM']['d']*100:+.1f} | bn: {o['ZS_bn']['d']*100:+.1f} {o['HM_bn']['d']*100:+.1f}")
summ = {}
for b in ("ZS", "HM", "ZS_bn", "HM_bn"):
    pg = "PGA_bn" if b.endswith("bn") else "PGA"
    summ[b] = dict(mean_acc=float(np.mean([out[m][b]["acc"] for m in MODELS])), mean_d=float(np.mean([out[m][b]["d"] for m in MODELS])),
                   pga_sig_better=int(sum(out[m][b]["d_ci"][0] > 0 for m in MODELS)), pga_sig_worse=int(sum(out[m][b]["d_ci"][1] < 0 for m in MODELS)),
                   n_improved=int(sum(out[m][b]["gain"] > 0 for m in MODELS)), n_sig_improved=int(sum(out[m][b]["gain_ci"][0] > 0 for m in MODELS)),
                   best_model=max(MODELS, key=lambda m: out[m][b]["acc"]))
    # all-model pooled difference (mean over models of the per-tile correctness), bootstrap over radiographs
    summ[b]["pooled_d"], summ[b]["pooled_ci"] = diff(np.mean([C[m][pg] for m in MODELS], 0), np.mean([C[m][b] for m in MODELS], 0))
for v in ("PGA", "PGA_bn"): summ[v] = dict(mean_acc=float(np.mean([out[m][v]["acc"] for m in MODELS])))
print(json.dumps(summ, indent=1)); json.dump(dict(models=out, summary=summ), open(W + "results/swrd_std.json", "w"), indent=1)
