"""Tables and macros for the SWRD adaptation experiments."""
import json, numpy as np
W = "./"; P = "../latex/"
d = json.load(open(W + "results/swrd_adapt.json")); R = d["recal"]; F = d["ft"]
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
N = dict(cfg["names"]); N.update({"prismr+paug": "PRISM-Net", "resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-S",
                                   "shufflenet_v2_x1_0": "ShuffleNetV2", "efficientnet_b0": "EfficientNet-B0", "vit": "ViT", "gfnet": "GFNet"})
FT = ["prismr+paug", "resnet18", "resnet18+paug"]
# main-text table: fine-tuning
with open(P + "tab_adapt.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{1.8pt}\n\\begin{tabular}{@{}llccccc@{}}\n\\toprule\n"
            "Budget & Model & Acc & ECE & Cov. & Size & $\\Delta$ [95\\% CI]\\\\\n\\midrule\n")
    for bi, (b, lab) in enumerate([(0, "0\\%"), (5, "5\\%"), (20, "20\\%")]):
        accs = {m: np.mean(F[f"{m}|{b}"]["acc"]) for m in FT}; best = max(accs.values())
        eces = {m: np.mean(F[f"{m}|{b}"]["ece"]) for m in FT}; beste = min(eces.values())
        for i, m in enumerate(FT):
            r = F[f"{m}|{b}"]; a = np.array(r["acc"]) * 100
            acc = f"{a.mean():.1f}$\\pm${a.std(ddof=1):.1f}"; acc = f"\\textbf{{{acc}}}" if accs[m] == best else acc
            e = f"{np.mean(r['ece'])*100:.1f}"; e = f"\\textbf{{{e}}}" if eces[m] == beste else e
            if b == 0:
                rc = R[m]["25"]; cov, size = f"{rc[0]*100:.1f}$^\\dagger$", f"{rc[1]:.2f}$^\\dagger$"
            else:
                cov, size = f"{r['cov']*100:.1f}", f"{r['size']:.2f}"
            dl = "---" if m == "prismr+paug" else f"${r['d']*100:+.1f}$ $[{r['ci'][0]*100:.1f}, {r['ci'][1]*100:.1f}]$"
            f.write(f"{lab if i == 0 else ''} & {N[m]} & {acc} & {e} & {cov} & {size} & {dl}\\\\\n")
        if bi < 2: f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# supplementary table: recalibration for all models
with open(P + "supp_tab_recal.tex", "w") as f:
    f.write("\\footnotesize\n\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n"
            " & \\multicolumn{2}{c}{RIAWELC thresholds} & \\multicolumn{3}{c}{25 SWRD radiographs} & \\multicolumn{3}{c}{200 SWRD radiographs}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
            "Model & Cov. & Size & Cov. (5th pct.) & Size & Single & Cov. (5th pct.) & Size & Single\\\\\n\\midrule\n")
    for m in MODELS:
        r = R[m]; a = r["riawelc"]; b = r["25"]; c = r["200"]
        f.write(f"{N[m]} & {a[0]*100:.1f} & {a[1]:.2f} & {b[0]*100:.1f} ({b[3]*100:.1f}) & {b[1]:.2f} & {b[2]*100:.1f} & "
                f"{c[0]*100:.1f} ({c[3]*100:.1f}) & {c[1]:.2f} & {c[2]*100:.1f}\\\\\n")
        if m == cfg["main"][-1]: f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
others = [m for m in MODELS if m != "prismr+paug"]
g = lambda m, b, k: np.mean(F[f"{m}|{b}"][k]) * 100
mac = {"AdTestImgs": str(d["n_test_imgs"]), "AdTestTiles": f"{d['n_test_tiles']:,}".replace(",", "\\,"),
       "RcCovLo": f"{min(R[m]['25'][0] for m in MODELS)*100:.1f}", "RcCovHi": f"{max(R[m]['25'][0] for m in MODELS)*100:.1f}",
       "RcSizeLo": f"{min(R[m]['25'][1] for m in MODELS):.1f}", "RcSizeHi": f"{max(R[m]['25'][1] for m in MODELS):.1f}",
       "RcSinglePrism": f"{R['prismr+paug']['25'][2]*100:.0f}", "RcSingleOther": f"{max(R[m]['25'][2] for m in others)*100:.0f}",
       "RcCovPrismTwoH": f"{R['prismr+paug']['200'][0]*100:.1f}", "RcFifthLo": f"{min(R[m]['25'][3] for m in MODELS)*100:.0f}",
       "FtPrismFive": f"{g('prismr+paug',5,'acc'):.1f}", "FtRNFive": f"{g('resnet18',5,'acc'):.1f}", "FtRNPFive": f"{g('resnet18+paug',5,'acc'):.1f}",
       "FtPrismTw": f"{g('prismr+paug',20,'acc'):.1f}", "FtRNTw": f"{g('resnet18',20,'acc'):.1f}", "FtRNPTw": f"{g('resnet18+paug',20,'acc'):.1f}",
       "FtPrismEceFive": f"{g('prismr+paug',5,'ece'):.1f}", "FtRNEceFive": f"{g('resnet18',5,'ece'):.1f}", "FtRNPEceFive": f"{g('resnet18+paug',5,'ece'):.1f}",
       "FtPrismEceTw": f"{g('prismr+paug',20,'ece'):.1f}", "FtRNEceTw": f"{g('resnet18',20,'ece'):.1f}",
       "FtDFive": f"{F['resnet18|5']['d']*100:.1f}", "FtCIFive": "$[{:.1f}, {:.1f}]$".format(*[x * 100 for x in F['resnet18|5']['ci']]),
       "FtDTw": f"{F['resnet18|20']['d']*100:.1f}", "FtCITw": "$[{:.1f}, {:.1f}]$".format(*[x * 100 for x in F['resnet18|20']['ci']]),
       "FtCovPrismFive": f"{F['prismr+paug|5']['cov']*100:.1f}", "FtCovRNFive": f"{F['resnet18|5']['cov']*100:.1f}",
       "FtCovPrismTw": f"{F['prismr+paug|20']['cov']*100:.1f}"}
with open(P + "numbers_adapt.tex", "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(mac)
