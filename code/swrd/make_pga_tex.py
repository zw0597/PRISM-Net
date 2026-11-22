"""Tables and macros for PGA (zero-label, all models; few-shot with PGA-R)."""
import json, os, numpy as np
from sklearn.metrics import f1_score
W = "./"; P = "../latex/"
d = json.load(open(W + "results/swrd_pga.json")); M = d["models"]
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
N = dict(cfg["names"]); N.update({"prismr+paug": "PRISM-Net", "resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-S",
                                   "shufflenet_v2_x1_0": "ShuffleNetV2", "efficientnet_b0": "EfficientNet-B0", "vit": "ViT", "gfnet": "GFNet"})
COLS = ["frozen", "R", "bn", "R_bn"]
best = {v: max(M[m][v]["acc"] for m in MODELS) for v in COLS}
with open(P + "tab_pga.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            "Model & Frozen & PGA-R & AdaBN & PGA-R + AdaBN & Gain [95\\% CI]\\\\\n\\midrule\n")
    for m in MODELS:
        c = []
        for v in COLS:
            t = f"{M[m][v]['acc']*100:.1f}"; c.append(f"\\textbf{{{t}}}" if M[m][v]["acc"] == best[v] else t)
        g = M[m]["R_bn"]; f.write(f"{N[m]} & " + " & ".join(c) + f" & ${g['gain']*100:+.1f}$ $[{g['gain_ci'][0]*100:.1f}, {g['gain_ci'][1]*100:.1f}]$\\\\\n")
        if m == cfg["main"][-1]: f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
gR = [M[m]["R"]["gain"] * 100 for m in MODELS]; gRB = [M[m]["R_bn"]["gain"] * 100 for m in MODELS]; gS = [M[m]["S"]["gain"] * 100 for m in MODELS]
pr = M["prismr+paug"]; others = [m for m in MODELS if m != "prismr+paug"]
ci = lambda c: f"$[{c[0]*100:.1f}, {c[1]*100:.1f}]$"
mac = {"PgPrismR": f"{pr['R']['acc']*100:.1f}", "PgPrismRB": f"{pr['R_bn']['acc']*100:.1f}", "PgPrismGain": f"{pr['R_bn']['gain']*100:.1f}",
       "PgPrismGainCI": ci(pr['R_bn']['gain_ci']), "PgNPos": str(sum(g > 0 for g in gR)), "PgNPosRB": str(sum(g > 0 for g in gRB)),
       "PgNSig": str(sum(M[m]['R_bn']['gain_ci'][0] > 0 for m in MODELS)), "PgNModels": str(len(MODELS)),
       "PgRNRB": f"{M['resnet18']['R_bn']['acc']*100:.1f}", "PgRNGain": f"{M['resnet18']['R_bn']['gain']*100:.1f}",
       "PgMaxGain": f"{max(gRB):.1f}", "PgSLo": f"{min(gS):.1f}", "PgSHi": f"{max(gS):.1f}",
       "PgLeadRN": f"{M['resnet18']['R_bn']['d']*100:.1f}", "PgLeadRNCI": ci(M['resnet18']['R_bn']['d_ci']),
       "PgVggRB": f"{M['vgg11_bn']['R_bn']['acc']*100:.1f}", "PgLeadVgg": f"{M['vgg11_bn']['R_bn']['d']*100:+.1f}", "PgLeadVggCI": ci(M['vgg11_bn']['R_bn']['d_ci']),
       "PgMinLo": f"{min(M[m]['R_bn']['d_ci'][0] for m in others)*100:.1f}",
       "PgGS": f"{d['params']['g_S']:.2f}", "PgCS": f"{d['params']['c_S']:.2f}",
       "PgGRlo": f"{d['params']['g_R'][0]:.2f}", "PgGRmed": f"{d['params']['g_R'][1]:.2f}", "PgGRhi": f"{d['params']['g_R'][2]:.2f}"}
ch = json.load(open(W + "results/swrd_chain.json")); cc = [ch[m]["chain"] for m in MODELS]
mac.update({"ChCovLo": f"{min(c['cov'] for c in cc)*100:.1f}", "ChCovHi": f"{max(c['cov'] for c in cc)*100:.1f}",
            "ChNSmaller": str(sum(ch[m]['chain']['size'] < ch[m]['frozen']['size'] for m in MODELS)),
            "ChPrismSize": f"{ch['prismr+paug']['chain']['size']:.2f}", "ChPrismSizeFrozen": f"{ch['prismr+paug']['frozen']['size']:.2f}",
            "ChPrismSingle": f"{ch['prismr+paug']['chain']['single']*100:.1f}", "ChPrismSingleFrozen": f"{ch['prismr+paug']['frozen']['single']*100:.1f}",
            "ChOtherSingle": f"{max(ch[m]['chain']['single'] for m in MODELS if not m.startswith('prism'))*100:.1f}",
            "ChLiteSingle": f"{ch['prismx+paug']['chain']['single']*100:.1f}"})
# few-shot with PGA-R
S = np.load(W + "swrd/swrd_tiles80.npz"); Y = S["y"]; IMG = S["img"]; sp = json.load(open(W + "swrd/swrd_split.json")); te = np.isin(IMG, sp["test"])
sel = json.load(open(W + "swrd/round2_selection.json")); FT = ["prismr+paug", "resnet18", "resnet18+paug"]
ims = np.sort(np.array(sp["test"])); ix = np.searchsorted(ims, IMG[te]); cnt = np.bincount(ix).astype(float)
boot = np.random.default_rng(0).integers(0, len(ims), (10000, len(ims)))


def diff(ca, cb):
    ds = np.bincount(ix, weights=ca - cb); bs = ds[boot].sum(1) / cnt[boot].sum(1)
    return float(ds.sum() / cnt.sum()), [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


ft = {}
if all(os.path.exists(f"{W}swrd/ft/{m}_f{f}_b{b}" + ("" if sel[f'{m}|{b}'] == "C0" else f"_{sel[f'{m}|{b}']}") + "_pgaR.npy") for m in FT for f in range(5) for b in (5, 20)):
    for b in (5, 20):
        C = {}; C0 = {}
        for m in FT:
            suf = "" if sel[f"{m}|{b}"] == "C0" else f"_{sel[f'{m}|{b}']}"
            Ps = [np.load(f"{W}swrd/ft/{m}_f{f}_b{b}{suf}_pgaR.npy") for f in range(5)]; P0 = [np.load(f"{W}swrd/ft/{m}_f{f}_b{b}{suf}.npy") for f in range(5)]
            C[m] = np.mean([p[te].argmax(1) == Y[te] for p in Ps], 0); C0[m] = np.mean([p[te].argmax(1) == Y[te] for p in P0], 0)
            acc = [float((p[te].argmax(1) == Y[te]).mean()) for p in Ps]
            ft[f"{m}|{b}"] = dict(acc=float(np.mean(acc)), std=float(np.std(acc, ddof=1)), ece=float(np.mean([ece(p[te], Y[te]) for p in Ps])),
                                  acc0=float(C0[m].mean()))
            ft[f"{m}|{b}"]["gain"], ft[f"{m}|{b}"]["gain_ci"] = diff(C[m], C0[m])
        for m in FT[1:]:
            ft[f"{m}|{b}"]["d"], ft[f"{m}|{b}"]["d_ci"] = diff(C["prismr+paug"], C[m])
        print(b, {m: (round(ft[f'{m}|{b}']['acc0'] * 100, 1), round(ft[f'{m}|{b}']['acc'] * 100, 1), round(ft[f'{m}|{b}']['ece'] * 100, 1),
                      [round(x * 100, 1) for x in ft[f'{m}|{b}']['gain_ci']], ft[f'{m}|{b}'].get('d'), ft[f'{m}|{b}'].get('d_ci')) for m in FT})
    json.dump(ft, open(W + "results/swrd_pga_ft.json", "w"), indent=1)
    for b, t in ((5, "Five"), (20, "Tw")):
        for m, k in (("prismr+paug", "Prism"), ("resnet18", "RN"), ("resnet18+paug", "RNP")):
            r = ft[f"{m}|{b}"]; mac[f"PgFt{k}{t}"] = f"{r['acc']*100:.1f}"; mac[f"PgFt{k}{t}Gain"] = f"{r['gain']*100:+.1f}"
            mac[f"PgFt{k}{t}GainCI"] = ci(r["gain_ci"]); mac[f"PgFt{k}{t}Ece"] = f"{r['ece']*100:.1f}"
            if "d" in r: mac[f"PgFt{k}{t}D"] = f"{r['d']*100:+.1f}"; mac[f"PgFt{k}{t}DCI"] = ci(r["d_ci"])
with open(P + "numbers_pga.tex", "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(mac)
