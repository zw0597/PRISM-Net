"""v4.5 additions: Holm column for tab_pair, R5 class-conditional coverage table, R4 longer-budget table (if available).
Usage: python make_v45_tex.py <latex_dir>"""
import json, sys, re, os, glob, numpy as np
L = sys.argv[1]
H = json.load(open("holm_pair.json"))
NAME2TAG = {"PRISM-Net-Lite": "prismx+paug", "ResNet-18": "resnet18", "VGG-11": "vgg11_bn", "MobileNetV3-S": "mobilenet_v3_small",
            "ShuffleNetV2": "shufflenet_v2_x1_0", "EfficientNet-B0": "efficientnet_b0", "ViT": "vit", "GFNet": "gfnet",
            "ConvNeXt V2-Atto": "convnextv2_atto", "RepViT-M0.9": "repvit_m0_9", "MobileNetV4-S": "mobilenetv4_conv_small",
            "ResNet-18 + PAug": "resnet18+paug", "MobileNetV3-S + PAug": "mobilenet_v3_small+paug", "GFNet + PAug": "gfnet+paug"}
def fp(p): return "$<\\!10^{-3}$" if p < 1e-3 else f"${p:.3f}$"
# ---- tab_pair with Holm column (rebuilt from the v4.4 table rows, values unchanged)
src = open(os.path.join(L, "tab_pair.tex")).read()
if "Holm" not in src:
    out = []
    for line in src.splitlines():
        if line.startswith("\\begin{tabular}"):
            line = "\\begin{tabular}{@{}lccccc@{}}"
        elif line.startswith("Baseline &"):
            line = "Baseline & $\\Delta$Acc (pts) & 95\\% CI & $p_{\\mathrm{W}}$ & $p_{\\mathrm{Holm}}$ & W/L\\\\"
        else:
            m = re.match(r"^(.+?) & (.+?) & (.+?) & (.+?) & (.+?)\\\\$", line)
            if m and m.group(1) in NAME2TAG:
                ph = H[NAME2TAG[m.group(1)]][1]
                line = f"{m.group(1)} & {m.group(2)} & {m.group(3)} & {m.group(4)} & {fp(ph)} & {m.group(5)}\\\\"
        out.append(line)
    open(os.path.join(L, "tab_pair.tex"), "w").write("\n".join(out) + "\n")
nsig = sum(v[1] < 0.05 for v in H.values())
# ---- R5 table
C = json.load(open("cp_class.json"))
rows = ["PRISM-Net", "ResNet-18", "VGG-11", "ResNet-18 + PAug"]
t = ["\\footnotesize", "\\setlength{\\tabcolsep}{2.2pt}", "\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
     "Model & Calib. & ND & PO & LP & CR & All & Size & Single\\\\", "\\midrule"]
for r in rows:
    for key, lab in (("marg", "marg."), ("mond", "class")):
        v = C[r]["0.1"][key]; cc = v["cov_cls"]
        t.append(f"{r if key == 'marg' else ''} & {lab} & " + " & ".join(f"{x*100:.1f}" for x in cc) +
                 f" & {v['cov']*100:.1f} & {v['size']:.2f} & {v['single']*100:.1f}\\\\")
    if r != rows[-1]: t.append("\\addlinespace[1pt]")
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "tab_cpclass.tex"), "w").write("\n".join(t) + "\n")
tf = ["\\footnotesize", "\\begin{tabular}{@{}lcccccccccccc@{}}", "\\toprule",
      " & \\multicolumn{6}{c}{Marginal calibration} & \\multicolumn{6}{c}{Class-conditional calibration}\\\\",
      "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
      "Model & ND & PO & LP & CR & All & Single & ND & PO & LP & CR & All & Single\\\\", "\\midrule"]
for r in C:
    cells = []
    for key in ("marg", "mond"):
        v = C[r]["0.1"][key]; cells += [f"{x*100:.1f}" for x in v["cov_cls"]] + [f"{v['cov']*100:.1f}", f"{v['single']*100:.1f}"]
    tf.append(r + " & " + " & ".join(cells) + "\\\\")
tf += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "tab_cpclass_full.tex"), "w").write("\n".join(tf) + "\n")
lp_marg = [C[m]["0.1"]["marg"]["cov_cls"][2] for m in C]
mond_all = [C[m]["0.1"]["mond"]["cov"] for m in C]
mond_min = [min(C[m]["0.1"]["mond"]["cov_cls"]) for m in C]
P = C["PRISM-Net"]["0.1"]
num = {"HolmNSig": str(nsig), "HolmRN": f"{H['resnet18'][1]:.2f}", "HolmRNP": f"{H['resnet18+paug'][1]:.2f}",
       "HolmVgg": f"{H['vgg11_bn'][1]:.2f}", "HolmGF": f"{H['gfnet'][1]:.3f}",
       "CpLPLo": f"{min(lp_marg)*100:.1f}", "CpLPHi": f"{max(lp_marg)*100:.1f}",
       "CpPrismLP": f"{P['marg']['cov_cls'][2]*100:.1f}", "CpPrismCR": f"{P['marg']['cov_cls'][3]*100:.1f}",
       "CpPrismND": f"{P['marg']['cov_cls'][0]*100:.1f}", "CpPrismPO": f"{P['marg']['cov_cls'][1]*100:.1f}",
       "CpMondLo": f"{min(mond_all)*100:.1f}", "CpMondHi": f"{max(mond_all)*100:.1f}",
       "CpPrismMond": f"{P['mond']['cov']*100:.1f}", "CpPrismMondMin": f"{min(P['mond']['cov_cls'])*100:.1f}",
       "CpPrismMondMax": f"{max(P['mond']['cov_cls'])*100:.1f}", "CpPrismMondSingle": f"{P['mond']['single']*100:.1f}",
       "CpPrismLPFive": f"{C['PRISM-Net']['0.05']['marg']['cov_cls'][2]*100:.1f}"}
# ---- R4 (longer budget), if complete
def load(d, m):
    fs = []
    for f in range(5):
        p = f"{d}/{m}_group_f{f}_s0.json"
        if not os.path.exists(p): return None
        fs.append(p)
    return fs
LONG = {}
if all(load(f"results_gpu_{it}", m) for it in (1000, 3000) for m in ("prismr+paug", "resnet18", "vgg11_bn")):
    exec(open("long_analysis.py").read())
open(os.path.join(L, "numbers_v45.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in {**num, **LONG}.items()))
print(num); print(LONG)
