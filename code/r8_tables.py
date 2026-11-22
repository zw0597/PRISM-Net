# Presentation: supplementary table of the R8 training variants and of UP on PGA-R. Usage: python r8_tables.py <latex_dir>
import sys, os, json, numpy as np
L = sys.argv[1]; T = json.load(open("r8_trained_summary.json")); o = T["out"]; I = json.load(open("r8_infer_summary.json"))
f = lambda x: f"{x*100:.1f}"
rows = [("comp", "PRISM-Net (comparator)"), ("oc", "Orbit consistency (OC)"), ("canon", "Canonical acquisition (CA)"), ("occanon", "OC + CA"), ("B0", "Comparator, test-time CA")]
t = ["\\footnotesize", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
     " & \\multicolumn{4}{c}{RIAWELC (five folds)} & \\multicolumn{3}{c}{SWRD test half, accuracy}\\\\", "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
     "Variant & Acc & Val.\\ acc & ECE & mCA & Frozen & PGA-R & PGA-R + AdaBN\\\\", "\\midrule"]
for k, nm in rows:
    v = o[k]; sw = v["swrd"]; canon = k in ("canon", "occanon", "B0")
    fr = f(sw["frozen"]["test"]["acc"]); pg = "--" if canon else f(sw["pgaR"]["test"]["acc"]); bn = f(sw["bn"]["test"]["acc"])
    t.append(f"{nm} & {f(v['acc'])} & {f(v['val_acc'])} & {f(v['ece'])} & {f(v['mca'])} & {fr} & {pg} & {bn}\\\\")
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "supp_tab_r8.tex"), "w").write("\n".join(t) + "\n")
M = list(I); num = {"UpPgaECE": f(np.mean([I[m]["R"]["test"]["ece"] for m in M])), "UpPgaECEUp": f(np.mean([I[m]["C"]["test"]["ece"] for m in M])),
     "UpPgaAcc": f(np.mean([I[m]["R"]["test"]["acc"] for m in M])), "UpPgaAccUp": f(np.mean([I[m]["C"]["test"]["acc"] for m in M])),
     "UpPgaNLL": f"{np.mean([I[m]['R']['test']['nll'] for m in M]):.2f}", "UpPgaNLLUp": f"{np.mean([I[m]['C']['test']['nll'] for m in M]):.2f}"}
open(os.path.join(L, "numbers_r8.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()))
print(num)
