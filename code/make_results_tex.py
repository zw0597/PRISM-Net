"""Result tables and macros for the paper, generated from summary.json / efficiency.json / interp.json."""
import json, os
import numpy as np

P = "../latex/"
S = json.load(open("./summary.json")); A, OFF = S["agg"], S["official"]
C = json.load(open("./paper_cfg.json"))
E = json.load(open("./efficiency.json")) if os.path.exists("./efficiency.json") else {}
FIN, NAME = C["final"], C["names"]
MAIN = [m for m in C["main"] if m in A]
PB = [m for m in C["paug_baselines"] if m in A]
CLS = ["ND", "PO", "LP", "CR"]


def pm(v, scale=100, d=1):
    return f"${v[0]*scale:.{d}f}\\pm{v[1]*scale:.{d}f}$"


def pv(p):
    if p is None:
        return "--"
    if p < 1e-3:
        return "$<\\!10^{-3}$"
    return f"${p:.3f}$"


def bold_best(rows, col_vals, higher):
    """return set of row indices whose mean is best in column"""
    arr = np.array(col_vals)
    b = arr.max() if higher else arr.min()
    return {i for i, v in enumerate(arr) if abs(v - b) < 1e-9}


def emph(s, on):
    return f"\\textbf{{{s}}}" if on else s


def bm(s, on):  # bold inside math
    return "$\\mathbf{" + s[1:-1] + "}$" if on else s


# ------------------------------------------------------------------ main comparison
mods = MAIN + PB
cols = [("acc", True), ("mf1", True), ("bacc", True), ("ece", False), ("nll", False), ("aurc", False)]
best = {k: bold_best(mods, [A[m][k][0] for m in mods], h) for k, h in cols}
with open(P + "tab_main.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.4pt}\n\\begin{tabular}{@{}lrrrcccccc@{}}\n\\toprule\n")
    f.write("Model & Par.\\,(M) & GFLOPs & Lat.\\,(ms) & Acc (\\%) & mF1 (\\%) & BAcc (\\%) & ECE (\\%) & NLL & AURC (\\%)\\\\\n\\midrule\n")
    for i, m in enumerate(mods):
        if m == PB[0] if PB else False:
            f.write("\\midrule\n")
        a = A[m]; e = E.get(m.split("+")[0], {})
        par = f"{a['params']/1e6:.2f}"; gf = f"{e['gflops']:.3f}" if e else "--"; lat = f"{e['lat_ms']:.1f}" if e else "--"
        cells = [bm(pm(a["acc"]), i in best["acc"]), bm(pm(a["mf1"]), i in best["mf1"]), bm(pm(a["bacc"]), i in best["bacc"]),
                 bm(pm(a["ece"]), i in best["ece"]), bm(pm(a["nll"], 1, 3), i in best["nll"]), bm(pm(a["aurc"]), i in best["aurc"])]
        wl = f"{a['films_ref_better']}/{a['films_ref_worse']}" if "films_ref_better" in a else "--"
        f.write(f"{NAME[m]}{' (ours)' if m == FIN else ''} & {par} & {gf} & {lat} & " + " & ".join(cells) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ paired comparison (film level)
rng = np.random.default_rng(0)
RF = A[FIN]["per_film"]; FL = sorted(RF)
nF = np.array([RF[f][1] for f in FL]); aF = np.array([RF[f][0] for f in FL])
BIDX = rng.integers(0, len(FL), (10000, len(FL)))
PAIR = {}
for m in mods:
    if m == FIN:
        continue
    b = np.array([A[m]["per_film"][f][0] for f in FL])
    d = (nF * (aF - b)).sum() / nF.sum()
    bs = (nF[BIDX] * (aF[BIDX] - b[BIDX])).sum(1) / nF[BIDX].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    PAIR[m] = (d, lo, hi, A[m].get("wilcoxon_p_vs_ref"), A[m].get("films_ref_better"), A[m].get("films_ref_worse"))
with open(P + "tab_pair.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.4pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
    f.write("Baseline & $\\Delta$Acc (pts) & 95\\% CI & $p_{\\mathrm{W}}$ & W/L\\\\\n\\midrule\n")
    for m in mods:
        if m == FIN:
            continue
        if PB and m == PB[0]:
            f.write("\\midrule\n")
        d, lo, hi, p, w, l = PAIR[m]
        f.write(f"{NAME[m]} & ${d*100:+.2f}$ & $[{lo*100:.2f},\\,{hi*100:.2f}]$ & {pv(p)} & {w}/{l}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ per-class F1
with open(P + "tab_cls.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{1.6pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nModel & ND & PO & LP & CR\\\\\n\\midrule\n")
    bc = [bold_best(mods, [A[m]["f1_cls"][k] for m in mods], True) for k in range(4)]
    for i, m in enumerate(mods):
        f.write(NAME[m] + " & " + " & ".join(bm(f"${A[m]['f1_cls'][k]*100:.1f}\\pm{A[m]['f1_cls_std'][k]*100:.1f}$", i in bc[k]) for k in range(4)) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ official vs grouped
om = [m for m in C["official_models"] if m in OFF and m in A]
infl = []
with open(P + "tab_official.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\nModel & Official split & Film-grouped & Inflation\\\\\n\\midrule\n")
    for m in om:
        d = (OFF[m]["acc"] - A[m]["acc"][0]) * 100; infl.append(d)
        f.write(f"{NAME[m]} & ${OFF[m]['acc']*100:.2f}$ & {pm(A[m]['acc'], 100, 2)} & $+{d:.2f}$\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ ablation
HELD = ["blur_0.8", "blur_1.5", "blur_2.5", "contrast_0.7", "contrast_0.5", "contrast_0.3"]
INF = ["gamma_0.5", "gamma_0.7", "gamma_1.5", "gamma_2.0", "gain_0.4", "gain_0.7", "noise_0.02", "noise_0.05", "noise_0.08"]


def mean_rob(a, keys):
    return float(np.mean([a["rob"][k][0] for k in keys]))


ab = [(m, n) for m, n in C["ablation"] if m in A]
ref = A[FIN]
with open(P + "tab_ablation.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n")
    f.write("Variant & Acc (\\%) & $\\Delta$Acc & $p_{\\mathrm{W}}$ & mF1 (\\%) & ECE (\\%) & Rob.$_{\\mathrm{in}}$ & Rob.$_{\\mathrm{out}}$\\\\\n\\midrule\n")
    for m, n in ab:
        a = A[m]; d = (a["acc"][0] - ref["acc"][0]) * 100
        ds = "--" if m == FIN else f"${d:+.1f}$"
        nf = a["n_folds"]
        f.write(f"{n}{'' if nf == 5 else f'$^{{({nf})}}$'} & {pm(a['acc'])} & {ds} & {'--' if m == FIN else pv(a.get('wilcoxon_p_vs_ref'))} & ${a['mf1'][0]*100:.1f}$ & ${a['ece'][0]*100:.1f}$ & "
                f"${mean_rob(a, INF)*100:.1f}$ & ${mean_rob(a, HELD)*100:.1f}$\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ robustness by family
FAM = [("gamma", ["gamma_0.5", "gamma_0.7", "gamma_1.5", "gamma_2.0"]), ("gain", ["gain_0.4", "gain_0.7"]),
       ("noise", ["noise_0.02", "noise_0.05", "noise_0.08"]), ("blur", ["blur_0.8", "blur_1.5", "blur_2.5"]),
       ("contrast", ["contrast_0.7", "contrast_0.5", "contrast_0.3"])]
with open(P + "tab_robust.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n")
    f.write("Model & Clean & Gamma & Gain & Noise & Blur$^{\\dagger}$ & Contr.$^{\\dagger}$ & mCA\\\\\n\\midrule\n")
    vals = {m: [A[m]["acc"][0]] + [mean_rob(A[m], ks) for _, ks in FAM] + [mean_rob(A[m], INF + HELD)] for m in mods}
    bc = [bold_best(mods, [vals[m][j] for m in mods], True) for j in range(7)]
    for i, m in enumerate(mods):
        if PB and m == PB[0]:
            f.write("\\midrule\n")
        f.write(NAME[m] + " & " + " & ".join(emph(f"{vals[m][j]*100:.1f}", i in bc[j]) for j in range(7)) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ conformal / selective
with open(P + "tab_conf.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n")
    f.write(" & \\multicolumn{3}{c}{$\\varepsilon=0.10$} & \\multicolumn{3}{c}{$\\varepsilon=0.05$} & \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
    f.write("Model & Cov.\\ (\\%) & Size & Single (\\%) & Cov.\\ (\\%) & Size & Single (\\%) & ECE$_{\\mathrm{TS}}$ (\\%)\\\\\n\\midrule\n")
    for m in mods:
        a = A[m]; c10, c5 = a["cp10"], a["cp5"]
        ts = f"{a['ece_ts'][0]*100:.2f}" if "ece_ts" in a else f"{a['ece'][0]*100:.2f}$^{{\\ast}}$"
        f.write(f"{NAME[m]} & {c10[0]*100:.1f} & {c10[1]:.2f} & {c10[2]*100:.1f} & {c5[0]*100:.1f} & {c5[1]:.2f} & {c5[2]*100:.1f} & {ts}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ macros
bb = max([m for m in MAIN if m not in (FIN, "prismx+paug")], key=lambda m: A[m]["acc"][0])
lite = A["prismx+paug"]
def d(m):  # accuracy difference to the final model (points, positive = final better)
    return f"{(ref['acc'][0] - A[m]['acc'][0])*100:.1f}"
others = [m for m in MAIN + PB if m not in (FIN, "resnet18", "vgg11_bn") and "wilcoxon_p_vs_ref" in A[m]]
pmax = max(A[m]["wilcoxon_p_vs_ref"] for m in others)
def fp(p):
    return "<10^{-3}" if p < 1e-3 else f"{p:.3f}"
I = json.load(open("./interp.json")) if os.path.exists("./interp.json") else None
eta = "--"
if I:
    aucs = [v["auc_PO_lt_oriented"] for k, v in I["layers"].items() if k in ("glob.6.rasg", "glob.7.rasg")]
    eta = f"{min(aucs):.2f}--{max(aucs):.2f}"
ef = lambda m, k, fmt: (fmt % E[m][k]) if m in E else "--"
mac = {
    "PrismAcc": f"{ref['acc'][0]*100:.1f}", "PrismAccStd": f"{ref['acc'][1]*100:.1f}", "PrismMF": f"{ref['mf1'][0]*100:.1f}",
    "PrismECE": f"{ref['ece'][0]*100:.1f}", "PrismAURC": f"{ref['aurc'][0]*100:.2f}",
    "PrismCovTen": f"{ref['cp10'][0]*100:.1f}", "PrismSizeTen": f"{ref['cp10'][1]:.2f}", "PrismSingleTen": f"{ref['cp10'][2]*100:.1f}",
    "PrismCovFive": f"{ref['cp5'][0]*100:.1f}", "PrismParams": f"{ref['params']/1e6:.1f}",
    "PrismMCA": f"{mean_rob(ref, INF + HELD)*100:.1f}", "PrismGainLow": f"{ref['rob']['gain_0.4'][0]*100:.1f}",
    "BestBase": NAME[bb], "BestBaseAcc": f"{A[bb]['acc'][0]*100:.1f}", "BestBaseMCA": f"{mean_rob(A[bb], INF + HELD)*100:.1f}",
    "RNGainLow": f"{A['resnet18']['rob']['gain_0.4'][0]*100:.1f}",
    "MarginBest": d(bb), "RNPAcc": f"{A['resnet18+paug']['acc'][0]*100:.1f}", "MarginRNP": d("resnet18+paug"),
    "pRN": fp(A["resnet18"]["wilcoxon_p_vs_ref"]), "pRNP": fp(A["resnet18+paug"]["wilcoxon_p_vs_ref"]), "pMaxOther": fp(pmax),
    "WinsRN": str(A["resnet18"]["films_ref_better"]), "CIRN": f"[{PAIR['resnet18'][1]*100:.2f}, {PAIR['resnet18'][2]*100:.2f}]", "DRN": f"{PAIR['resnet18'][0]*100:.2f}", "CIRNP": f"[{PAIR['resnet18+paug'][1]*100:.2f}, {PAIR['resnet18+paug'][2]*100:.2f}]", "DRNP": f"{PAIR['resnet18+paug'][0]*100:.2f}", "CIMinLo": f"{min(v[1] for v in PAIR.values())*100:.2f}", "LossRN": str(A["resnet18"]["films_ref_worse"]),
    "LiteAcc": f"{lite['acc'][0]*100:.1f}", "LiteParams": f"{lite['params']/1e6:.2f}",
    "AblNoPAug": d("prismr") if "prismr" in A else "--", "AblCE": d("prismr_ce+paug") if "prismr_ce+paug" in A else "--",
    "AblNoRASG": d("prismr_norasg+paug") if "prismr_norasg+paug" in A else "--", "AblIso": d("prismr_iso+paug") if "prismr_iso+paug" in A else "--",
    "AblRNP": d("resnet18+paug"),
    "AblBLAN": f"{(lite['acc'][0] - A['prism+paug']['acc'][0])*100:.1f}" if "prism+paug" in A else "--",
    "InflationRange": (f"{min(infl):.1f} and {max(infl):.1f}" if infl else "--"),
    "OffMin": (f"{min(OFF[m]['acc'] for m in om)*100:.0f}" if om else "--"),
    "EtaAUC": eta,
    "PrismLat": ef("prismr", "lat_ms", "%.1f"), "RNLat": ef("resnet18", "lat_ms", "%.1f"), "LiteLat": ef("prismx", "lat_ms", "%.1f"),
    "LiteGF": ef("prismx", "gflops", "%.3f"),
    "RasgParams": f"{(E['prismr']['params'] - E['prismr_norasg']['params'])/1e6:.2f}",
    "RasgGF": f"{E['prismr']['gflops'] - E['prismr_norasg']['gflops']:.4f}", "RasgLat": f"{E['prismr']['lat_ms'] - E['prismr_norasg']['lat_ms']:.1f}",
    "TilesPerSec": (f"{int(1000 / E['prismr']['lat_ms'] / 10) * 10}" if "prismr" in E else "--"),
    "VggAcc": f"{A['vgg11_bn']['acc'][0]*100:.1f}", "VggD": f"{PAIR['vgg11_bn'][0]*100:.2f}", "VggCI": f"[{PAIR['vgg11_bn'][1]*100:.2f}, {PAIR['vgg11_bn'][2]*100:.2f}]",
    "VggPW": fp(A["vgg11_bn"]["wilcoxon_p_vs_ref"]), "VggW": str(A["vgg11_bn"]["films_ref_better"]), "VggL": str(A["vgg11_bn"]["films_ref_worse"]),
    "VggMCA": f"{mean_rob(A['vgg11_bn'], INF + HELD)*100:.1f}", "RNMCA": f"{mean_rob(A['resnet18'], INF + HELD)*100:.1f}",
    "VggNoise": f"{mean_rob(A['vgg11_bn'], ['noise_0.02', 'noise_0.05', 'noise_0.08'])*100:.1f}",
    "PrismNoise": f"{mean_rob(ref, ['noise_0.02', 'noise_0.05', 'noise_0.08'])*100:.1f}",
    "VggECE": f"{A['vgg11_bn']['ece'][0]*100:.1f}", "VggGF": ef("vgg11_bn", "gflops", "%.2f"), "VggLat": ef("vgg11_bn", "lat_ms", "%.1f"),
    "PrismGF": ef("prismr", "gflops", "%.2f"), "VggOff": f"{OFF['vgg11_bn']['acc']*100:.1f}",
    "NCiExcl": str(sum(PAIR[m][1] > 0 for m in PAIR)), "NBase": str(len(PAIR)),
    "TrainIters": "1000", "TrainEpochs": "5", "SeedCount": "three",
}
with open(P + "numbers.tex", "w") as f:
    f.write("\\input{numbers_audit}\n")
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(json.dumps(mac, indent=0))
