# R6 analysis (reviewer_protocol.md, R6; written before any R6 result exists).
# All 15 models at 3000 steps (GPU, seed 0, five film-grouped folds): accuracy, macro-F1, ECE, mCA; PRISM-Net vs each
# baseline at the film level (tile-weighted difference, film bootstrap 1e4 rng 0, Wilcoxon over films, W/L) + Holm.
# Usage: python r6_analysis.py <latex_dir>   -> r6_summary.json, <latex_dir>/tab_r6.tex, supp_tab_r6_folds.tex, numbers_r6.tex
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
def f1_score(y, pr, average="macro", K=4):   # macro-F1 over the K classes (as sklearn, zero_division=0)
    f = []
    for c in range(K):
        tp = ((pr == c) & (y == c)).sum(); fp_ = ((pr == c) & (y != c)).sum(); fn = ((pr != c) & (y == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp_ + fn))
    return float(np.mean(f))
L = sys.argv[1]
D = np.load("./riawelc_80.npz"); Y = D["y"]; FILM = D["film"]
R = "results_gpu_3000"
REF = ("prismr+paug", "PRISM-Net")
BASE = [("prismx+paug", "PRISM-Net-Lite"), ("resnet18", "ResNet-18"), ("vgg11_bn", "VGG-11"),
        ("mobilenet_v3_small", "MobileNetV3-S"), ("shufflenet_v2_x1_0", "ShuffleNetV2"), ("efficientnet_b0", "EfficientNet-B0"),
        ("vit", "ViT"), ("gfnet", "GFNet"), ("convnextv2_atto", "ConvNeXt V2-Atto"), ("repvit_m0_9", "RepViT-M0.9"),
        ("mobilenetv4_conv_small", "MobileNetV4-S"), ("resnet18+paug", "ResNet-18 + PAug"),
        ("mobilenet_v3_small+paug", "MobileNetV3-S + PAug"), ("gfnet+paug", "GFNet + PAug")]
ALL = [REF] + BASE
if os.environ.get("R6_TEST"): BASE = [b for b in BASE if b[0] in ("resnet18", "vgg11_bn")]; ALL = [REF] + BASE
PAUG_GROUP = {"resnet18+paug", "mobilenet_v3_small+paug", "gfnet+paug"}

def probs(Z, lt):
    Z = torch.from_numpy(Z).double()
    if lt == "edl":
        a = F.softplus(Z) + 1.0; return (a / a.sum(1, keepdim=True)).numpy()
    return F.softmax(Z, 1).numpy()

def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.sum(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

S = {}
for m, nm in ALL:
    acc, mf1, ec, mca, film, trl, tra = [], [], [], [], {}, [], []
    for f in range(5):
        J = json.load(open(f"{R}/{m}_group_f{f}_s0.json")); lt = J["loss_type"]
        Z = np.load(f"{R}/{m}_group_f{f}_s0.npz"); te = Z["te_idx"]; yt = Y[te]
        p = probs(Z["Zte_clean_0"], lt); pr = p.argmax(1); c = pr == yt
        acc.append(c.mean()); mf1.append(f1_score(yt, pr, average="macro")); ec.append(ece(p, yt))
        ks = [k for k in Z.files if k.startswith("Zte_") and k != "Zte_clean_0"]; assert len(ks) == 15, (m, f, len(ks))
        mca.append(np.mean([(Z[k].argmax(1) == yt).mean() for k in ks]))
        for fl in np.unique(FILM[te]):
            k = FILM[te] == fl; film[fl] = (c[k].mean(), int(k.sum()))
        trl.append(J["log"][-1]["loss"]); tra.append(J["log"][-1]["acc"])
    S[m] = dict(name=nm, acc=np.array(acc), mf1=np.array(mf1), ece=np.array(ec), mca=np.array(mca), film=film,
                tr_loss=np.array(trl), tr_acc=np.array(tra))

def cmp(a, b):
    fa_, fb_ = S[a]["film"], S[b]["film"]; FL = sorted(fa_)
    n = np.array([fa_[f][1] for f in FL]); fa = np.array([fa_[f][0] for f in FL]); fb = np.array([fb_[f][0] for f in FL])
    pw = wilcoxon(fa, fb).pvalue
    rng = np.random.default_rng(0); idx = rng.integers(0, len(FL), (10000, len(FL)))
    d = (n * (fa - fb)).sum() / n.sum(); bs = (n[idx] * (fa[idx] - fb[idx])).sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(d=float(d), lo=float(lo), hi=float(hi), pw=float(pw), W=int((fa > fb).sum()), L=int((fa < fb).sum()))

C = {m: cmp(REF[0], m) for m, _ in BASE}
order = sorted(C, key=lambda m: C[m]["pw"]); k = len(order); run = 0.0
for i, m in enumerate(order):
    run = max(run, min(1.0, (k - i) * C[m]["pw"])); C[m]["holm"] = run

def fp(p): return "$<\\!10^{-3}$" if p < 1e-3 else f"${p:.3f}$"
best = lambda key, hi=True: (max if hi else min)(S[m][key].mean() for m, _ in ALL)
t = ["\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcccccccccc@{}}", "\\toprule",
     "Model & Acc (\\%) & mF1 (\\%) & ECE (\\%) & mCA (\\%) & $\\Delta$Acc (pts) & 95\\% CI & $p_{\\mathrm{W}}$ & $p_{\\mathrm{Holm}}$ & W/L\\\\", "\\midrule"]
for m, nm in ALL:
    v = S[m]
    def cell(key, hi=True, sd=True):
        x = v[key].mean() * 100; s = f"{x:.1f}" + (f"\\pm{v[key].std(ddof=1)*100:.1f}" if sd else "")
        return f"$\\mathbf{{{s}}}$" if abs(v[key].mean() - best(key, hi)) < 1e-12 else f"${s}$"
    row = f"{nm} & {cell('acc')} & {cell('mf1')} & {cell('ece', False)} & {cell('mca', True, False)}"
    if m == REF[0]: row += " & -- & -- & -- & -- & --\\\\"
    else:
        c = C[m]; row += f" & ${c['d']*100:+.2f}$ & $[{c['lo']*100:.2f}, {c['hi']*100:.2f}]$ & {fp(c['pw'])} & {fp(c['holm'])} & {c['W']}/{c['L']}\\\\"
    if m == "resnet18+paug": t.append("\\midrule")
    t.append(row)
t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "tab_r6.tex"), "w").write("\n".join(t) + "\n")

s = ["\\footnotesize", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
     "Model & F1 & F2 & F3 & F4 & F5 & Train loss & Train acc (\\%)\\\\", "\\midrule"]
for m, nm in ALL:
    v = S[m]
    s.append(f"{nm} & " + " & ".join(f"{x*100:.1f}" for x in v["acc"]) + f" & {v['tr_loss'].mean():.3f} & {v['tr_acc'].mean()*100:.1f}\\\\")
s += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "supp_tab_r6_folds.tex"), "w").write("\n".join(s) + "\n")

rank = sorted(ALL, key=lambda x: -S[x[0]]["acc"].mean())
bb = [x for x in rank if x[0] != REF[0]][0]
num = {"RsPrism": f"{S[REF[0]]['acc'].mean()*100:.1f}", "RsPrismStd": f"{S[REF[0]]['acc'].std(ddof=1)*100:.1f}",
       "RsPrismRank": str([x[0] for x in rank].index(REF[0]) + 1),
       "RsBest": rank[0][1], "RsBestAcc": f"{S[rank[0][0]]['acc'].mean()*100:.1f}",
       "RsBestBase": bb[1], "RsBestBaseAcc": f"{S[bb[0]]['acc'].mean()*100:.1f}",
       "RsNCi": str(sum(C[m]["lo"] > 0 for m in C)), "RsNCiNeg": str(sum(C[m]["hi"] < 0 for m in C)),
       "RsNHolm": str(sum(C[m]["holm"] < 0.05 for m in C)),
       "RsNHolmNames": ", ".join(S[m]["name"] for m, _ in BASE if C[m]["holm"] >= 0.05),
       "RsMinAcc": f"{min(S[m]['acc'].mean() for m, _ in ALL)*100:.1f}",
       "RsPrismECE": f"{S[REF[0]]['ece'].mean()*100:.1f}", "RsPrismMCA": f"{S[REF[0]]['mca'].mean()*100:.1f}",
       "RsECERank": str(sorted(ALL, key=lambda x: S[x[0]]["ece"].mean()).index(REF) + 1),
       "RsMCARank": str(sorted(ALL, key=lambda x: -S[x[0]]["mca"].mean()).index(REF) + 1)}
KEY = {"prismr+paug": "Prism", "prismx+paug": "Lite", "resnet18": "RN", "vgg11_bn": "Vgg", "mobilenet_v3_small": "MBThree",
       "shufflenet_v2_x1_0": "Shuffle", "efficientnet_b0": "Eff", "vit": "ViT", "gfnet": "GF", "convnextv2_atto": "ConvNeXt",
       "repvit_m0_9": "RepViT", "mobilenetv4_conv_small": "MBFour", "resnet18+paug": "RNP", "mobilenet_v3_small+paug": "MBThreeP",
       "gfnet+paug": "GFP"}
for m, nm in ALL:
    num[f"RsAcc{KEY[m]}"] = f"{S[m]['acc'].mean()*100:.1f}"
    if m != REF[0]:
        c = C[m]; num[f"RsD{KEY[m]}"] = f"{c['d']*100:+.2f}"; num[f"RsCI{KEY[m]}"] = f"[{c['lo']*100:.2f}, {c['hi']*100:.2f}]"
        num[f"RsPW{KEY[m]}"] = f"{c['pw']:.3f}"; num[f"RsHolm{KEY[m]}"] = f"{c['holm']:.3f}"
open(os.path.join(L, "numbers_r6.tex"), "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()))
json.dump({m: dict(name=v["name"], **{k: v[k].tolist() for k in ("acc", "mf1", "ece", "mca", "tr_loss", "tr_acc")}) for m, v in S.items()}
          | {"cmp": C}, open("r6_summary.json", "w"), indent=1)
for m, nm in rank:
    v = S[m]; c = C.get(m)
    print(f"{nm:22s} acc {v['acc'].mean()*100:5.1f}±{v['acc'].std(ddof=1)*100:3.1f} mf1 {v['mf1'].mean()*100:5.1f} ece {v['ece'].mean()*100:4.1f} "
          f"mca {v['mca'].mean()*100:5.1f} trloss {v['tr_loss'].mean():.3f}" +
          (f"  d {c['d']*100:+.2f} [{c['lo']*100:.2f},{c['hi']*100:.2f}] pW {c['pw']:.3f} holm {c['holm']:.3f} {c['W']}/{c['L']}" if c else ""))
print(num)
