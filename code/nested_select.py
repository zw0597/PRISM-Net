"""Nested (per-fold) model selection: removes the test-fold films from every decision.

For outer fold f the candidate network is trained on the training films of fold f, the candidate
with the highest accuracy on the validation films of fold f is selected, and only that candidate is
scored on the test films of fold f. The test films of fold f are therefore never used for training or
for selection of the model evaluated on them. The estimate is that of the *selection procedure*.

Pools
  all   : every design trained on all five folds (seed 0), both rounds, ablations and baselines
  round1: designs that existed before the second design round (baselines, PRISM-Net-Lite family, BLAN model)
Writes paper/numbers_nested.tex, paper/tab_nested.tex, results/nested.json
"""
import glob, os, json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

W = "./"; P = "../latex/"
D = np.load(W + "riawelc_80.npz"); Y = D["y"]; FILM = D["film"]
NAMES = {"prismr+paug": "PRISM", "prismr_ce+paug": "PRISM, CE", "prismr_norasg+paug": "PRISM, no RASG",
         "prismr_iso+paug": "PRISM, iso.", "prismr": "PRISM, no PAug", "resnet18": "RN18",
         "resnet18+paug": "RN18 + PAug", "prismx+paug": "Lite", "gfnet": "GFNet", "gfnet+paug": "GFNet + PAug", "vgg11_bn": "VGG-11"}
ROUND2 = ("prismr", "prismx_k7", "prismx_stem2", "prismx_l05", "prismx_w")

runs = {}
for d in ("results", "dev2"):
    for z in glob.glob(f"{W}{d}/*_group_f*_s0.npz"):
        m, f = os.path.basename(z)[:-7].split("_group_f"); runs.setdefault(m, {})[int(f)] = z
pool_all = sorted(m for m, r in runs.items() if len(r) == 5)
pool_r1 = [m for m in pool_all if not m.startswith(ROUND2)]


def load(m, f):
    Z = np.load(runs[m][f]); va, te = Z["va_idx"], Z["te_idx"]
    return (Z["Zva"].argmax(1) == Y[va]).mean(), te, Z["Zte_clean_0"].argmax(1)


def nested(pool):
    sel, correct, pred_all, te_all, accs, mf1 = [], [], [], [], [], []
    for f in range(5):
        best = max(pool, key=lambda m: load(m, f)[0])
        _, te, p = load(best, f)
        sel.append(best); accs.append((p == Y[te]).mean()); mf1.append(f1_score(Y[te], p, average="macro"))
        correct.append(p == Y[te]); te_all.append(te)
    return sel, np.array(accs), np.array(mf1), np.concatenate(te_all), np.concatenate(correct)


def fixed(m):
    te_all, c, accs = [], [], []
    for f in range(5):
        _, te, p = load(m, f); te_all.append(te); c.append(p == Y[te]); accs.append((p == Y[te]).mean())
    return np.array(accs), np.concatenate(te_all), np.concatenate(c)


def film_cmp(te, ca, cb):
    FL = np.unique(FILM[te]); fa = np.array([ca[FILM[te] == x].mean() for x in FL])
    fb = np.array([cb[FILM[te] == x].mean() for x in FL]); n = np.array([(FILM[te] == x).sum() for x in FL])
    rng = np.random.default_rng(0); idx = rng.integers(0, len(FL), (10000, len(FL)))
    d = (n * (fa - fb)).sum() / n.sum(); bs = (n[idx] * (fa[idx] - fb[idx])).sum(1) / n[idx].sum(1)
    return d, np.percentile(bs, [2.5, 97.5]), wilcoxon(fa, fb).pvalue, int((fa > fb).sum()), int((fa < fb).sum())


out = {"pool_all": pool_all, "pool_round1": pool_r1}
acc_rn, te_rn, c_rn = fixed("resnet18")
acc_pr, te_pr, c_pr = fixed("prismr+paug")
mac = {}
rows = []
for key, pool in (("all", pool_all), ("r1", pool_r1)):
    sel, accs, mf1, te, c = nested(pool)
    assert np.array_equal(te, te_rn)  # identical split -> identical tile order
    d, ci, pw, w, l = film_cmp(te_rn, c, c_rn)
    out[key] = {"selected": sel, "acc": accs.tolist(), "mF1": mf1.tolist(), "d_vs_rn": d, "ci": ci.tolist(), "pW": pw, "W": w, "L": l}
    rows.append((key, sel, accs, mf1, d, ci, pw, w, l))
    tag = "All" if key == "all" else "RI"
    mac[f"Nest{tag}Acc"] = f"{accs.mean()*100:.1f}"; mac[f"Nest{tag}Std"] = f"{accs.std(ddof=1)*100:.1f}"
    mac[f"Nest{tag}MF"] = f"{mf1.mean()*100:.1f}"; mac[f"Nest{tag}D"] = f"{d*100:+.2f}"
    mac[f"Nest{tag}CI"] = f"$[{ci[0]*100:.2f}, {ci[1]*100:.2f}]$"; mac[f"Nest{tag}PW"] = f"{pw:.3f}"
    mac[f"Nest{tag}N"] = str(len(pool)); mac[f"Nest{tag}Prism"] = str(sum(s.startswith("prismr") for s in sel))
    mac[f"Nest{tag}Gap"] = f"{(acc_pr.mean()-accs.mean())*100:.1f}"
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(W + "results/nested.json", "w"), indent=1, default=float)
with open(P + "numbers_nested.tex", "w") as fh:
    for k, v in mac.items():
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
with open(P + "tab_nested.tex", "w") as fh:
    fh.write("\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
             "Fold & \\multicolumn{2}{c}{All designs (%d)} & \\multicolumn{2}{c}{W/o round 2 (%d)}\\\\\n"
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n & Selected & Acc & Selected & Acc\\\\\n\\midrule\n" % (len(pool_all), len(pool_r1)))
    A, B = rows
    for f in range(5):
        fh.write(f"F{f+1} & {NAMES.get(A[1][f], A[1][f])} & {A[2][f]*100:.1f} & {NAMES.get(B[1][f], B[1][f])} & {B[2][f]*100:.1f}\\\\\n")
    fh.write("\\midrule\n")
    fh.write(f"Mean & & ${A[2].mean()*100:.1f}\\pm{A[2].std(ddof=1)*100:.1f}$ & & ${B[2].mean()*100:.1f}\\pm{B[2].std(ddof=1)*100:.1f}$\\\\\n")
    fh.write(f"$\\Delta$ vs.\\ RN18 & & ${A[4]*100:+.2f}$ & & ${B[4]*100:+.2f}$\\\\\n")
    fh.write(f"95\\% CI & & $[{A[5][0]*100:.2f}, {A[5][1]*100:.2f}]$ & & $[{B[5][0]*100:.2f}, {B[5][1]*100:.2f}]$\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
