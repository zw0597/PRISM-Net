"""Main-text adaptation table (round 2) and macros; round-1 table goes to the supplement."""
import json, numpy as np, shutil
W = "./"; P = "../latex/"
d = json.load(open(W + "results/swrd_adapt2.json")); A = d["adabn"]; F = d["ft"]
r1 = json.load(open(W + "results/swrd_adapt.json"))["ft"]
N = {"prismr+paug": "PRISM-Net", "resnet18": "ResNet-18", "resnet18+paug": "ResNet-18+PAug"}
FT = ["prismr+paug", "resnet18", "resnet18+paug"]
RN = {"C0": "R1", "C1": "BN+L", "C2": "BN+L$-$P", "C3": "BN+L+ST"}
ci = lambda c: f"$[{c[0]*100:.1f}, {c[1]*100:.1f}]$"
with open(P + "tab_adapt.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{1.5pt}\n\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
            "Labels & Model (recipe) & Acc & ECE & $\\Delta$ [95\\% CI]\\\\\n\\midrule\n")
    blocks = [("0 (AdaBN)", {m: dict(acc=A[m]["acc"], std=A[m]["std"], mf1=A[m]["mf1"], ece=A[m]["ece"], d=A[m]["d"], ci=A[m]["ci"], rec="") for m in FT})]
    for b, lab in ((5, "5\\% (79)"), (20, "20\\% (317)")):
        blocks.append((lab, {m: dict(acc=np.mean(F[f"{m}|{b}"]["acc"]), std=float(np.std(F[f"{m}|{b}"]["acc"], ddof=1)),
                                     mf1=np.mean(F[f"{m}|{b}"]["mf1"]), ece=np.mean(F[f"{m}|{b}"]["ece"]),
                                     d=F[f"{m}|{b}"].get("d"), ci=F[f"{m}|{b}"].get("ci"), rec=RN[F[f"{m}|{b}"]["recipe"]]) for m in FT}))
    for bi, (lab, R) in enumerate(blocks):
        ba = max(r["acc"] for r in R.values()); be = min(r["ece"] for r in R.values())
        for i, m in enumerate(FT):
            r = R[m]; acc = f"{r['acc']*100:.1f}$\\pm${r['std']*100:.1f}"; acc = f"\\textbf{{{acc}}}" if r["acc"] == ba else acc
            e = f"{r['ece']*100:.1f}"; e = f"\\textbf{{{e}}}" if r["ece"] == be else e
            name = N[m] + (f" ({r['rec']})" if r["rec"] else "")
            dl = "---" if m == "prismr+paug" else f"${r['d']*100:+.1f}$ {ci(r['ci'])}"
            f.write(f"{lab if i == 0 else ''} & {name} & {acc} & {e} & {dl}\\\\\n")
        if bi < len(blocks) - 1: f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# supplement: round-1 (protocol) table and AdaBN for all models
with open(P + "supp_tab_adapt_r1.tex", "w") as f:
    f.write("\\footnotesize\n\\begin{tabular}{@{}llcccc@{}}\n\\toprule\nLabels & Model & Acc & ECE & Cov. & $\\Delta$ [95\\% CI]\\\\\n\\midrule\n")
    for bi, b in enumerate((5, 20)):
        for i, m in enumerate(FT):
            r = r1[f"{m}|{b}"]; a = np.array(r["acc"]) * 100
            dl = "---" if m == "prismr+paug" else f"${r['d']*100:+.1f}$ {ci(r['ci'])}"
            f.write(f"{(str(b)+'%').replace('%','\\%') if i == 0 else ''} & {N[m]} & {a.mean():.1f}$\\pm${a.std(ddof=1):.1f} & {np.mean(r['ece'])*100:.1f} & {r['cov']*100:.1f} & {dl}\\\\\n")
        if bi == 0: f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
cfg = json.load(open(W + "paper_cfg.json")); MODELS = [m for m in cfg["main"] + cfg["paug_baselines"] if m in A]
NN = dict(cfg["names"]); NN.update({"prismr+paug": "PRISM-Net", "resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-S",
                                     "shufflenet_v2_x1_0": "ShuffleNetV2", "efficientnet_b0": "EfficientNet-B0", "vit": "ViT", "gfnet": "GFNet"})
with open(P + "supp_tab_adabn.tex", "w") as f:
    f.write("\\footnotesize\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nModel & Acc (frozen) & Acc (AdaBN) & ECE (frozen) & ECE (AdaBN) & $\\Delta$ [95\\% CI]\\\\\n\\midrule\n")
    for m in MODELS:
        r = A[m]; dl = "---" if m == "prismr+paug" else f"${r['d']*100:+.1f}$ {ci(r['ci'])}"
        f.write(f"{NN[m]} & {r['acc0']*100:.1f} & {r['acc']*100:.1f}$\\pm${r['std']*100:.1f} & {r['ece0']*100:.1f} & {r['ece']*100:.1f} & {dl}\\\\\n")
        if m == cfg["main"][-1]: f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
g = lambda m, b, k: np.mean(F[f"{m}|{b}"][k]) * 100
others = [m for m in MODELS if m != "prismr+paug"]
mac = {"AbPrismZero": f"{A['prismr+paug']['acc0']*100:.1f}", "AbPrism": f"{A['prismr+paug']['acc']*100:.1f}",
       "AbRN": f"{A['resnet18']['acc']*100:.1f}", "AbMinD": f"{min(A[m]['d'] for m in others)*100:.1f}",
       "AbMinLo": f"{min(A[m]['ci'][0] for m in others)*100:.1f}",
       "RtPrismFive": f"{g('prismr+paug',5,'acc'):.1f}", "RtRNFive": f"{g('resnet18',5,'acc'):.1f}", "RtRNPFive": f"{g('resnet18+paug',5,'acc'):.1f}",
       "RtDFive": f"{F['resnet18|5']['d']*100:.1f}", "RtCIFive": ci(F['resnet18|5']['ci']),
       "RtDPFive": f"{F['resnet18+paug|5']['d']*100:.1f}", "RtCIPFive": ci(F['resnet18+paug|5']['ci']),
       "RtPrismEceFive": f"{g('prismr+paug',5,'ece'):.1f}", "RtRNEceFive": f"{g('resnet18',5,'ece'):.1f}",
       "RtPrismTw": f"{g('prismr+paug',20,'acc'):.1f}", "RtRNTw": f"{g('resnet18',20,'acc'):.1f}", "RtRNPTw": f"{g('resnet18+paug',20,'acc'):.1f}",
       "RtDTw": f"{F['resnet18|20']['d']*100:.1f}", "RtCITw": ci(F['resnet18|20']['ci']),
       "RtDPTw": f"{F['resnet18+paug|20']['d']*100:.1f}", "RtCIPTw": ci(F['resnet18+paug|20']['ci']),
       "RoPrismTw": f"{np.mean(r1['prismr+paug|20']['acc'])*100:.1f}", "RoRNTw": f"{np.mean(r1['resnet18|20']['acc'])*100:.1f}",
       "RoDTw": f"{r1['resnet18|20']['d']*100:.1f}"}
with open(P + "numbers_adapt2.tex", "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(mac)
