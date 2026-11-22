"""Table and macros for the SWRD external test."""
import json, numpy as np
W = "./"; P = "../latex/"
d = json.load(open(W + "results/swrd_external.json")); M = d["models"]; C = d["cmp"]
cfg = json.load(open(W + "paper_cfg.json")); N = dict(cfg["names"]); N["prismr+paug"] = "PRISM-Net"
N.update({"mobilenet_v3_small": "MobileNetV3-S", "shufflenet_v2_x1_0": "ShuffleNetV2", "efficientnet_b0": "EfficientNet-B0",
          "vit": "ViT", "gfnet": "GFNet", "resnet18": "ResNet-18"})
ROWS = cfg["main"] + ["@"] + cfg["paug_baselines"]
mean = lambda m, k: float(np.mean(M[m][k]))
best = {k: (max if k in ("acc", "mf1", "ens_acc") else min)(mean(m, k) if k != "ens_acc" else M[m][k] for m in ROWS if m != "@")
        for k in ("acc", "mf1", "ens_acc", "ece", "nll")}


def fmt(v, k, s):
    b = abs(v - best[k]) < 1e-9
    return ("\\textbf{%s}" % s) if b else s


with open(P + "tab_swrd.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            "Model & Acc & mF1 & ECE$\\downarrow$ & Cov. & $\\Delta$ [95\\% CI]\\\\\n\\midrule\n")
    for m in ROWS:
        if m == "@":
            f.write("\\midrule\n"); continue
        r = M[m]; a = np.array(r["acc"]) * 100
        cells = [fmt(a.mean() / 100, "acc", f"{a.mean():.1f}$\\pm${a.std(ddof=1):.1f}"),
                 fmt(mean(m, "mf1"), "mf1", f"{mean(m,'mf1')*100:.1f}"),
                 fmt(mean(m, "ece"), "ece", f"{mean(m,'ece')*100:.1f}"),
                 f"{mean(m,'cov')*100:.1f}"]
        dl = "---" if m == "prismr+paug" else f"${C[m]['d']*100:+.1f}$ [{C[m]['lo']*100:.1f}, {C[m]['hi']*100:.1f}]"
        f.write(f"{N[m]} & " + " & ".join(cells) + f" & {dl}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

pr = M["prismr+paug"]; rn = M["resnet18"]; rnp = M["resnet18+paug"]
others = [m for m in ROWS if m not in ("@", "prismr+paug")]
mac = {"SwTiles": f"{d['n_tiles']:,}".replace(",", "\\,"), "SwImgs": f"{d['n_images']:,}".replace(",", "\\,"),
       "SwLP": str(d["class_counts"][2]), "SwPrism": f"{mean('prismr+paug','acc')*100:.1f}",
       "SwPrismMF": f"{mean('prismr+paug','mf1')*100:.1f}", "SwPrismEns": f"{pr['ens_acc']*100:.1f}",
       "SwPrismNLL": f"{mean('prismr+paug','nll'):.2f}", "SwPrismECE": f"{mean('prismr+paug','ece')*100:.1f}", "SwRN": f"{mean('resnet18','acc')*100:.1f}",
       "SwRNP": f"{mean('resnet18+paug','acc')*100:.1f}",
       "SwDRN": f"{C['resnet18']['d']*100:.1f}", "SwCIRN": f"[{C['resnet18']['lo']*100:.1f}, {C['resnet18']['hi']*100:.1f}]",
       "SwDRNP": f"{C['resnet18+paug']['d']*100:.1f}", "SwCIRNP": f"[{C['resnet18+paug']['lo']*100:.1f}, {C['resnet18+paug']['hi']*100:.1f}]",
       "SwMinLo": f"{min(C[m]['lo'] for m in others)*100:.1f}",
       "SwWorst": f"{min(mean(m,'acc') for m in others)*100:.1f}",
       "SwCovMin": f"{min(mean(m,'cov') for m in ROWS if m!='@')*100:.0f}", "SwCovMax": f"{max(mean(m,'cov') for m in ROWS if m!='@')*100:.0f}",
       "SwNoRASG": f"{mean('prismr_norasg+paug','acc')*100:.1f}", "SwCE": f"{mean('prismr_ce+paug','acc')*100:.1f}",
       "SwNoPAug": f"{mean('prismr','acc')*100:.1f}", "SwIso": f"{mean('prismr_iso+paug','acc')*100:.1f}",
       "SwBLAN": f"{mean('prism+paug','acc')*100:.1f}",
       "SwDNoPAug": f"{C['prismr']['d']*100:.1f}", "SwCINoPAug": f"[{C['prismr']['lo']*100:.1f}, {C['prismr']['hi']*100:.1f}]",
       "SwDCE": f"{C['prismr_ce+paug']['d']*100:.1f}", "SwCICE": f"[{C['prismr_ce+paug']['lo']*100:.1f}, {C['prismr_ce+paug']['hi']*100:.1f}]",
       "SwRecPrism": ", ".join(f"{x*100:.0f}" for x in np.mean(pr["rec"], 0))}
with open(P + "numbers_swrd.tex", "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(mac)
