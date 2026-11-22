"""Seed study: PRISM-Net vs ResNet-18 over seeds 0-2 on all five folds.
Writes paper/supp_seeds.tex (table + text) and seed macros appended to paper/numbers.tex."""
import glob, os, json
import numpy as np
from scipy.stats import wilcoxon
R = "./results/"; P = "../latex/"
D = np.load("./riawelc_80.npz"); Y = D["y"]; FILM = D["film"]
MODELS = [("prismr+paug", "PRISM-Net"), ("resnet18", "ResNet-18"), ("vgg11_bn", "VGG-11")]
SEEDS = [0, 1, 2]


def run(m, f, s):
    Z = np.load(f"{R}{m}_group_f{f}_s{s}.npz"); te = Z["te_idx"]
    return te, (Z["Zte_clean_0"].argmax(1) == Y[te])


acc = {m: np.zeros((len(SEEDS), 5)) for m, _ in MODELS}
film_acc = {m: {} for m, _ in MODELS}            # film -> list over seeds of accuracy
film_n = {}
for m, _ in MODELS:
    for si, s in enumerate(SEEDS):
        for f in range(5):
            te, c = run(m, f, s)
            acc[m][si, f] = c.mean()
            for fl in np.unique(FILM[te]):
                k = FILM[te] == fl
                film_acc[m].setdefault(fl, []).append(c[k].mean()); film_n[fl] = int(k.sum())
FL = sorted(film_n); n = np.array([film_n[f] for f in FL])
def compare(ref, oth):
    a, b = acc[ref], acc[oth]
    fa = np.array([np.mean(film_acc[ref][f]) for f in FL]); fb = np.array([np.mean(film_acc[oth][f]) for f in FL])
    pw = wilcoxon(fa, fb).pvalue
    rng = np.random.default_rng(0); idx = rng.integers(0, len(FL), (10000, len(FL)))
    d = (n * (fa - fb)).sum() / n.sum(); bs = (n[idx] * (fa[idx] - fb[idx])).sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(W=str(int((a > b).sum())), P=str(a.size), PW=f"{pw:.3f}", D=f"{d*100:.2f}", CI=f"[{lo*100:.2f}, {hi*100:.2f}]",
                FW=str(int((fa > fb).sum())), FL=str(int((fa < fb).sum())))
mean_over_folds = {m: acc[m].mean(1) for m in acc}            # per seed
c1 = compare("prismr+paug", "resnet18"); c2 = compare("prismr+paug", "vgg11_bn")
mac = {"SeedPrism": f"{mean_over_folds['prismr+paug'].mean()*100:.1f}", "SeedPrismStd": f"{mean_over_folds['prismr+paug'].std(ddof=1)*100:.1f}",
       "SeedRN": f"{mean_over_folds['resnet18'].mean()*100:.1f}", "SeedRNStd": f"{mean_over_folds['resnet18'].std(ddof=1)*100:.1f}",
       "SeedVgg": f"{mean_over_folds['vgg11_bn'].mean()*100:.1f}", "SeedVggStd": f"{mean_over_folds['vgg11_bn'].std(ddof=1)*100:.1f}",
       "SeedWins": c1["W"], "SeedPairs": c1["P"], "SeedPW": c1["PW"], "SeedD": c1["D"], "SeedCI": c1["CI"], "SeedFilmW": c1["FW"], "SeedFilmL": c1["FL"],
       "SeedVWins": c2["W"], "SeedVPW": c2["PW"], "SeedVD": c2["D"], "SeedVCI": c2["CI"], "SeedVFilmW": c2["FW"], "SeedVFilmL": c2["FL"],
       "SeedPrismMin": f"{mean_over_folds['prismr+paug'].min()*100:.1f}", "SeedRNMax": f"{mean_over_folds['resnet18'].max()*100:.1f}",
       "SeedVggMax": f"{mean_over_folds['vgg11_bn'].max()*100:.1f}", "SeedVggMin": f"{mean_over_folds['vgg11_bn'].min()*100:.1f}"}
print(json.dumps(mac, indent=0))
for m in acc: print(m, acc[m].round(3))
with open(P + "numbers_seed.tex", "w") as f:
    for k, v in mac.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))
with open(P + "tab_seeds.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\nModel & Seed & F1 & F2 & F3 & F4 & F5 & Mean\\\\\n\\midrule\n")
    for m, nm in MODELS:
        for si, s in enumerate(SEEDS):
            f.write(f"{nm if si == 0 else ''} & {s} & " + " & ".join(f"{x*100:.1f}" for x in acc[m][si]) + f" & {acc[m][si].mean()*100:.1f}\\\\\n")
        mo = mean_over_folds[m]
        f.write(f" & Mean & " + " & ".join(f"{x*100:.1f}" for x in acc[m].mean(0)) + f" & ${mo.mean()*100:.1f}\\pm{mo.std(ddof=1)*100:.1f}$\\\\\n")
        if m != MODELS[-1][0]:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
