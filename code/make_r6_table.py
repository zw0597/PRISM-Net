# Presentation only: main-text Table VII from r6_summary.json (frozen r6_analysis.py) plus the GPU 1000-step accuracy of
# the three R4 models from long_summary.json (long_analysis.py), so that the R4 comparison needs no separate table.
# Usage: python make_r6_table.py <latex_dir>  -> tab_r6m.tex
import sys, os, json, numpy as np
L = sys.argv[1]; d = json.load(open("r6_summary.json")); C = d["cmp"]; G = json.load(open("long_summary.json"))
ORDER = ["prismr+paug", "prismx+paug", "resnet18", "vgg11_bn", "mobilenet_v3_small", "shufflenet_v2_x1_0", "efficientnet_b0", "vit",
         "gfnet", "convnextv2_atto", "repvit_m0_9", "mobilenetv4_conv_small", "resnet18+paug", "mobilenet_v3_small+paug", "gfnet+paug"]
mean = lambda m, k: np.mean(d[m][k])
best = {k: (min if k == "ece" else max)(mean(m, k) for m in ORDER) for k in ("acc", "mf1", "ece", "mca")}
def fp(p): return "$<\\!10^{-3}$" if p < 1e-3 else f"${p:.3f}$"
def cell(m, k, sd=True):
    s = f"{mean(m, k)*100:.1f}" + (f"\\pm{np.std(d[m][k], ddof=1)*100:.1f}" if sd else "")
    return f"$\\mathbf{{{s}}}$" if abs(mean(m, k) - best[k]) < 1e-12 else f"${s}$"
t = ["\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lccccccccccc@{}}", "\\toprule",
     "Model & Acc$_{1000}$ (\\%) & Acc (\\%) & mF1 (\\%) & ECE (\\%) & mCA (\\%) & $\\Delta$Acc (pts) & 95\\% CI & $p_{\\mathrm{W}}$ & $p_{\\mathrm{Holm}}$ & W/L\\\\", "\\midrule"]
for m in ORDER:
    if m == "resnet18+paug": t.append("\\midrule")
    a1 = f"${np.mean(G[m + '_1000']['acc'])*100:.1f}$" if f"{m}_1000" in G else "--"
    r = f"{d[m]['name']} & {a1} & {cell(m, 'acc')} & {cell(m, 'mf1')} & {cell(m, 'ece', False)} & {cell(m, 'mca', False)}"
    if m == "prismr+paug": r += " & -- & -- & -- & -- & --\\\\"
    else:
        c = C[m]; r += f" & ${c['d']*100:+.2f}$ & $[{c['lo']*100:.2f}, {c['hi']*100:.2f}]$ & {fp(c['pw'])} & {fp(c['holm'])} & {c['W']}/{c['L']}\\\\"
    t.append(r)
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "tab_r6m.tex"), "w").write("\n".join(t) + "\n")
