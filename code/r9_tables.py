# Presentation: main-text table and macros from r9_post.json / r8 summaries. Usage: python r9_tables.py <latex_dir>
import sys, os, json, numpy as np
L = sys.argv[1]; O = json.load(open("r9_post.json")); T = json.load(open("r8_trained_summary.json"))
CFG = json.load(open("paper_cfg.json")); MODELS = CFG["main"] + CFG["paug_baselines"]
NAME = dict(CFG["names"]); NAME.update({"prismr+paug": "PRISM-Net", "prismx+paug": "PRISM-Net-Lite"})
f = lambda x, d=1: f"{x*100:.{d}f}"
best = max(O[m]["caa"]["acc"] for m in MODELS)
t = ["\\footnotesize", "\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
     " & \\multicolumn{4}{c}{Accuracy (\\%)} & & \\multicolumn{2}{c}{ECE (\\%)}\\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){7-8}",
     "Model & PGA-R & +AdaBN & CAA-P & CAA-L & $\\Delta$ [95\\% CI] & CAA-L & +UP\\\\", "\\midrule"]
for m in MODELS:
    o = O[m]; d = o["caa-pgaR_bn"]; a = f(o["caa"]["acc"]); a = f"\\textbf{{{a}}}" if o["caa"]["acc"] == best else a
    if m == "resnet18+paug": t.append("\\midrule")
    t.append(f"{NAME[m]} & {f(o['pgaR']['acc'])} & {f(o['pgaR_bn']['acc'])} & {f(o['caaP']['acc'])} & {a} & $+{d['d']*100:.1f}$ $[{d['lo']*100:.1f}, {d['hi']*100:.1f}]$ & {f(o['caa']['ece'])} & {f(o['caa_bn_up']['ece'])}\\\\".replace("$+-", "$-"))
t.append("\\midrule")
for g, nm in (("gpu:comp", "PRISM-Net (GPU)"), ("gpu:oc", "PRISM-Net-OC (GPU)")):
    o = O[g]; d = o["caa-pgaR_bn"]
    t.append(f"{nm} & {f(o['pgaR']['acc'])} & {f(o['pgaR_bn']['acc'])} & -- & {f(o['caa']['acc'])} & $+{d['d']*100:.1f}$ $[{d['lo']*100:.1f}, {d['hi']*100:.1f}]$ & {f(o['caa']['ece'])} & {f(o['caa_bn_up']['ece'])}\\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "tab_caa.tex"), "w").write("\n".join(t) + "\n")
D = [O[m]["caa-pgaR_bn"] for m in MODELS]
P = O["prismr+paug"]; OC = O["gpu:oc"]; CP = O["gpu:comp"]; o8 = T["out"]
num = {"CaPrism": f(P["caa"]["acc"]), "CaPrismPB": f(P["pgaR_bn"]["acc"]), "CaPrismD": f(P["caa-pgaR_bn"]["d"]), "CaPrismCI": f"[{P['caa-pgaR_bn']['lo']*100:.1f}, {P['caa-pgaR_bn']['hi']*100:.1f}]",
       "CaNSig": str(sum(d["lo"] > 0 for d in D)), "CaNPos": str(sum(d["d"] > 0 for d in D)), "CaDLo": f(min(d["d"] for d in D)), "CaDHi": f(max(d["d"] for d in D)),
       "CaMean": f(np.mean([O[m]["caa"]["acc"] for m in MODELS])), "CaPBMean": f(np.mean([O[m]["pgaR_bn"]["acc"] for m in MODELS])),
       "CaPMean": f(np.mean([O[m]["caaP"]["acc"] for m in MODELS])), "CaRMean": f(np.mean([O[m]["pgaR"]["acc"] for m in MODELS])),
       "CaValL": f(np.mean([O[m]["val_acc"]["caa"] for m in MODELS])), "CaValP": "39.3", "CaValNL": "15",
       "CaECEPrism": f(P["caa"]["ece"]), "CaECEPrismPB": f(P["pgaR_bn"]["ece"]),
       "UpECEMean": f(np.mean([O[m]["caa_bn_up"]["ece"] for m in MODELS])), "UpECEMeanNo": f(np.mean([O[m]["caa_bn"]["ece"] for m in MODELS])),
       "UpAccMean": f(np.mean([O[m]["caa_bn_up"]["acc"] for m in MODELS])), "UpAccMeanNo": f(np.mean([O[m]["caa_bn"]["acc"] for m in MODELS])),
       "CaBNMean": f(np.mean([O[m]["caa_bn"]["acc"] for m in MODELS])),
       "CaSinglePrism": f(P["recal_caa"]["single"]), "CaSinglePrismPB": f(P["recal_pgaR_bn"]["single"]), "CaCovLo": f(min(O[m]["recal_caa"]["cov"] for m in MODELS)), "CaCovHi": f(max(O[m]["recal_caa"]["cov"] for m in MODELS)),
       "OcCaa": f(OC["caa"]["acc"]), "OcCaaComp": f(CP["caa"]["acc"]),
       "OcAcc": f(o8["oc"]["acc"]), "OcCompAcc": f(o8["comp"]["acc"]), "OcVal": f(o8["oc"]["val_acc"]), "OcCompVal": f(o8["comp"]["val_acc"]),
       "OcMCA": f(o8["oc"]["mca"]), "OcCompMCA": f(o8["comp"]["mca"]), "OcECE": f(o8["oc"]["ece"]), "OcCompECE": f(o8["comp"]["ece"]),
       "OcSwFD": f(T["swrd_cmp"]["oc|frozen-comp|frozen"]["d"]), "OcSwFCI": f"[{T['swrd_cmp']['oc|frozen-comp|frozen']['lo']*100:.1f}, {T['swrd_cmp']['oc|frozen-comp|frozen']['hi']*100:.1f}]",
       "OcSwBD": f(T["swrd_cmp"]["oc|bn-comp|bn"]["d"]), "OcSwBCI": f"[{T['swrd_cmp']['oc|bn-comp|bn']['lo']*100:.1f}, {T['swrd_cmp']['oc|bn-comp|bn']['hi']*100:.1f}]",
       "CnAcc": f(o8["canon"]["acc"]), "CnVal": f(o8["canon"]["val_acc"]), "CnSw": f(o8["canon"]["swrd"]["frozen"]["test"]["acc"]),
       "CnSwD": f(T["swrd_cmp"]["canon|frozen-comp|pgaR"]["d"]), "CnSwCI": f"[{T['swrd_cmp']['canon|frozen-comp|pgaR']['lo']*100:.1f}, {T['swrd_cmp']['canon|frozen-comp|pgaR']['hi']*100:.1f}]",
       "BzAcc": f(o8["B0"]["acc"]), "BzD": f(T["cmp"]["B0"]["d"]), "BzCI": f"[{T['cmp']['B0']['lo']*100:.1f}, {T['cmp']['B0']['hi']*100:.1f}]"}
open(os.path.join(L, "numbers_r9.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()))
print(num)
