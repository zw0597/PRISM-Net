"""Table and macros for R1: PGA-R against generic per-radiograph standardisation (ZS-R, HM-R)."""
import json
W = "./"; P = "../latex/"
d = json.load(open(W + "results/swrd_std.json")); M = d["models"]; S = d["summary"]
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
N = dict(cfg["names"]); N.update({"prismr+paug": "PRISM-Net", "resnet18": "ResNet-18", "mobilenet_v3_small": "MobileNetV3-S",
                                   "shufflenet_v2_x1_0": "ShuffleNetV2", "efficientnet_b0": "EfficientNet-B0", "vit": "ViT", "gfnet": "GFNet"})
COLS = ["ZS", "HM", "PGA", "ZS_bn", "HM_bn", "PGA_bn"]
with open(P + "tab_std.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Input alignment only} & \\multicolumn{3}{c}{With AdaBN}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Model & ZS-R & HM-R & PGA-R & ZS-R & HM-R & PGA-R\\\\\n\\midrule\n")
    for m in MODELS:
        c = []
        for grp in (COLS[:3], COLS[3:]):
            b = max(M[m][v]["acc"] for v in grp)
            for v in grp:
                t = f"{M[m][v]['acc']*100:.1f}"; c.append(f"\\textbf{{{t}}}" if M[m][v]["acc"] == b else t)
        f.write(f"{N[m]} & " + " & ".join(c) + "\\\\\n")
        if m == cfg["main"][-1]: f.write("\\midrule\n")
    f.write("\\midrule\nMean & " + " & ".join(f"{S[v]['mean_acc']*100:.1f}" for v in COLS) + "\\\\\n\\bottomrule\n\\end{tabular}\n")
ci = lambda c: f"$[{c[0]*100:.1f}, {c[1]*100:.1f}]$"
mac = {"StdN": str(len(MODELS))}
for v, k in (("ZS", "Zs"), ("HM", "Hm"), ("ZS_bn", "ZsB"), ("HM_bn", "HmB")):
    s = S[v]
    mac.update({f"Std{k}Mean": f"{s['mean_acc']*100:.1f}", f"Std{k}D": f"{s['pooled_d']*100:+.1f}", f"Std{k}DCI": ci(s["pooled_ci"]),
                f"Std{k}Better": str(s["pga_sig_better"]), f"Std{k}Worse": str(s["pga_sig_worse"]), f"Std{k}NImp": str(s["n_improved"]),
                f"Std{k}NSig": str(s["n_sig_improved"]), f"Std{k}Prism": f"{M['prismr+paug'][v]['acc']*100:.1f}",
                f"Std{k}PrismBest": str(int(s["best_model"] == "prismr+paug"))})
mac.update({"StdPgaMean": f"{S['PGA']['mean_acc']*100:.1f}", "StdPgaBMean": f"{S['PGA_bn']['mean_acc']*100:.1f}"})
with open(P + "numbers_std.tex", "w") as f:
    for k, v in mac.items(): f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(mac)
