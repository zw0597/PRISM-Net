# Presentation only: supplementary table and LaTeX macros from r7_summary.json (r7_decision.py). Usage: python r7_tables.py <latex_dir>
import sys, os, json, numpy as np
L = sys.argv[1]; S = json.load(open("r7_summary.json")); R = S["riawelc"]; W = S["swrd"]
CFG = json.load(open("paper_cfg.json")); MODELS = CFG["main"] + CFG["paug_baselines"]
NAME = dict(CFG["names"]); NAME.update({"prismr+paug": "PRISM-Net", "prismx+paug": "PRISM-Net-Lite"})
P, B = "R7|0.02|0.1", "B1|0.1"
t = ["\\footnotesize", "\\begin{tabular}{@{}lcccccccccc@{}}", "\\toprule",
     " & \\multicolumn{6}{c}{Threshold for $\\alpha=0.02$ + class-conditional sets ($\\varepsilon=0.10$)} & \\multicolumn{4}{c}{Marginal sets ($\\varepsilon=0.10$)}\\\\",
     "\\cmidrule(lr){2-7}\\cmidrule(lr){8-11}",
     "Model & MDR & MDR$_{\\max}$ & Films $>\\alpha$ & AR & LP cov. & Auto & MDR & AR & LP cov. & Auto\\\\", "\\midrule"]
for m in MODELS:
    v, b = R[m][P], R[m][B]
    if m == "resnet18+paug": t.append("\\midrule")
    t.append(f"{NAME[m]} & {v['mdr']*100:.2f} & {v['film_mdr_max']*100:.1f} & {v['film_frac_above']*100:.0f} & {v['ar']*100:.1f} & {v['cov'][2]*100:.1f} & {v['auto']*100:.1f}"
             f" & {b['mdr']*100:.2f} & {b['ar']*100:.1f} & {b['cov'][2]*100:.1f} & {b['auto']*100:.1f}\\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "supp_tab_r7.tex"), "w").write("\n".join(t) + "\n")
base = [m for m in MODELS if m not in ("prismr+paug", "prismx+paug")]
f = lambda x, d=2: f"{x*100:.{d}f}"
num = {"MdBPrism": f(R["prismr+paug"][B]["mdr"]), "MdBLo": f(min(R[m][B]["mdr"] for m in MODELS)), "MdBHi": f(max(R[m][B]["mdr"] for m in MODELS)),
       "ArBPrism": f(R["prismr+paug"][B]["ar"], 1), "AutoBPrism": f(R["prismr+paug"][B]["auto"], 1), "LpBPrism": f(R["prismr+paug"][B]["cov"][2], 1),
       "MdRPrism": f(R["prismr+paug"][P]["mdr"]), "MdRLite": f(R["prismx+paug"][P]["mdr"]),
       "MdRBaseLo": f(min(R[m][P]["mdr"] for m in base)), "MdRBaseHi": f(max(R[m][P]["mdr"] for m in base)),
       "MdRNAbove": str(sum(R[m][P]["mdr"] > 0.02 for m in MODELS)),
       "MdRFilmPrism": f(R["prismr+paug"][P]["film_mdr_max"], 1), "MdRFilmBaseMax": f(max(R[m][P]["film_mdr_max"] for m in base), 0),
       "MdROnePrism": f(R["prismr+paug"]["R7|0.01|0.1"]["mdr"]), "MdRFivePrism": f(R["prismr+paug"]["R7|0.05|0.1"]["mdr"]),
       "LpRPrism": f(R["prismr+paug"][P]["cov"][2], 1), "AutoRPrism": f(R["prismr+paug"][P]["auto"], 1), "ArRPrism": f(R["prismr+paug"][P]["ar"], 1),
       "MdSwF": f(W["prismr+paug|frozen"]["mdr"]), "MdSwB": f(W["prismr+paug|pga_bn"]["mdr"]),
       "MdSwFAbove": f(W["prismr+paug|frozen"]["frac_above"], 0), "MdSwBAbove": f(W["prismr+paug|pga_bn"]["frac_above"], 0),
       "AutoSwB": f(W["prismr+paug|pga_bn"]["auto"], 0)}
open(os.path.join(L, "numbers_r7.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()))
print(num)
