"""Supplementary tables: per-fold accuracies, per-corruption-level accuracies, seed study."""
import json, glob, os
import numpy as np
P = "../latex/"
S = json.load(open("./summary.json")); A = S["agg"]
C = json.load(open("./paper_cfg.json")); NAME = dict(C["names"])
NAME.update({"prismr": "PRISM-Net w/o PAug", "prismr_ce+paug": "PRISM-Net, CE", "prismr_norasg+paug": "PRISM-Net w/o RASG",
             "prismr_iso+paug": "PRISM-Net, isotropic gate"})
MODS = C["main"] + C["paug_baselines"] + ["prismr", "prismr_ce+paug", "prismr_norasg+paug", "prismr_iso+paug", "prism+paug"]
MODS = [m for m in MODS if m in A]
with open(P + "supp_tab_folds_acc.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nModel & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5 & Mean\\\\\n\\midrule\n")
    for m in MODS:
        v = A[m]["acc_folds"]
        f.write(NAME[m] + " & " + " & ".join(f"{x*100:.1f}" for x in v) + f" & {np.mean(v)*100:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
KEYS = ["gamma_0.5", "gamma_0.7", "gamma_1.5", "gamma_2.0", "gain_0.4", "gain_0.7", "noise_0.02", "noise_0.05", "noise_0.08",
        "blur_0.8", "blur_1.5", "blur_2.5", "contrast_0.7", "contrast_0.5", "contrast_0.3"]
HEAD = ["$\\gamma$0.5", "$\\gamma$0.7", "$\\gamma$1.5", "$\\gamma$2.0", "$c$0.4", "$c$0.7", "$\\sigma_n$.02", "$\\sigma_n$.05", "$\\sigma_n$.08",
        "$\\sigma_b$0.8", "$\\sigma_b$1.5", "$\\sigma_b$2.5", "$s$0.7", "$s$0.5", "$s$0.3"]
with open(P + "supp_tab_corr.tex", "w") as f:
    f.write("\\scriptsize\n\\setlength{\\tabcolsep}{1.6pt}\n\\begin{tabular}{@{}l" + "c" * 15 + "@{}}\n\\toprule\nModel & " + " & ".join(HEAD) + "\\\\\n\\midrule\n")
    for m in C["main"] + C["paug_baselines"]:
        f.write(NAME[m] + " & " + " & ".join(f"{A[m]['rob'][k][0]*100:.1f}" for k in KEYS) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("supp tables written")
