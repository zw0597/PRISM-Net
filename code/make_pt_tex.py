"""Table and macros for E1 (ImageNet initialisation, 112 px)."""
import json, numpy as np
W = "./"; P = "../latex/"
d = json.load(open(W + "results/pt_summary.json")); s = json.load(open(W + "summary.json"))["agg"]
MAP = [("prismrpt+paug", "prismr+paug", "PRISM-Net"), ("resnet18pt", "resnet18", "ResNet-18"), ("resnet18pt+paug", "resnet18+paug", "ResNet-18 + PAug")]
sc = lambda m: s[m]
mean = lambda m, k: float(np.mean(d[m][k]))
best = {k: (max if k in ("acc", "mf1") else min)(mean(m, k) for m, _, _ in MAP) for k in ("acc", "mf1", "ece", "nll")}
bf = lambda m, k, t: f"\\textbf{{{t}}}" if mean(m, k) == best[k] else t
with open(P + "tab_pt.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            "Model & Acc (scratch) & Acc & ECE$\\downarrow$ & NLL$\\downarrow$ & $\\Delta$ [95\\% CI]\\\\\n\\midrule\n")
    for m, m0, nm in MAP:
        a = np.array(d[m]["acc"]) * 100; a0 = sc(m0)["acc"]; a0 = a0[0] if isinstance(a0, list) else a0
        dl = "---" if m == "prismrpt+paug" else f"${d[m]['d']*100:+.1f}$ $[{d[m]['ci'][0]*100:.1f}, {d[m]['ci'][1]*100:.1f}]$"
        f.write(f"{nm} & {a0*100:.1f} & {bf(m,'acc',f'{a.mean():.1f}$\\pm${a.std(ddof=1):.1f}')} & {bf(m,'ece',f'{mean(m,chr(101)+chr(99)+chr(101))*100:.1f}')} & "
                f"{bf(m,'nll',f'{mean(m,chr(110)+chr(108)+chr(108)):.3f}')} & {dl}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
mac = {"PtPrism": f"{mean('prismrpt+paug','acc')*100:.1f}", "PtRN": f"{mean('resnet18pt','acc')*100:.1f}", "PtRNP": f"{mean('resnet18pt+paug','acc')*100:.1f}",
       "PtDRN": f"{d['resnet18pt']['d']*100:.1f}", "PtCIRN": "$[{:.1f}, {:.1f}]$".format(*[x * 100 for x in d['resnet18pt']['ci']]),
       "PtpRN": f"{d['resnet18pt']['pW']:.2f}", "PtDRNP": f"{d['resnet18pt+paug']['d']*100:.1f}",
       "PtCIRNP": "$[{:.1f}, {:.1f}]$".format(*[x * 100 for x in d['resnet18pt+paug']['ci']]), "PtpRNP": f"{d['resnet18pt+paug']['pW']:.2f}",
       "PtWinsRN": str(d['resnet18pt']['fold_wins']), "PtWinsRNP": str(d['resnet18pt+paug']['fold_wins']),
       "PtPrismNLL": f"{mean('prismrpt+paug','nll'):.3f}", "PtPrismECE": f"{mean('prismrpt+paug','ece')*100:.1f}",
       "PtRNPECE": f"{mean('resnet18pt+paug','ece')*100:.1f}"}
with open(P + "numbers_pt.tex", "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(mac); print(open(P + "tab_pt.tex").read())
