# R4 analysis (exec'd by make_v45_tex.py; defines LONG and writes tab_long.tex). Protocol: reviewer_protocol.md, R4.
import numpy as np, json, math
import torch, torch.nn.functional as F
from scipy.stats import wilcoxon
_D = np.load("./riawelc_80.npz"); _Y = _D["y"]; _FILM = _D["film"]
_M = [("prismr+paug", "PRISM-Net", "edl"), ("resnet18", "ResNet-18", "ce"), ("vgg11_bn", "VGG-11", "ce")]
def _probs(Z, lt):
    Z = torch.from_numpy(Z).double()
    if lt == "edl":
        a = F.softplus(Z) + 1.0; return (a / a.sum(1, keepdim=True)).numpy()
    return F.softmax(Z, 1).numpy()
def _ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.sum(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e
_R = {}
for it in (1000, 3000):
    for m, nm, lt in _M:
        accs, eces, mcas, film = [], [], [], {}
        for f in range(5):
            Z = np.load(f"results_gpu_{it}/{m}_group_f{f}_s0.npz"); te = Z["te_idx"]; yt = _Y[te]
            p = _probs(Z["Zte_clean_0"], lt); c = p.argmax(1) == yt
            accs.append(c.mean()); eces.append(_ece(p, yt))
            mcas.append(np.mean([(_probs(Z[k], lt).argmax(1) == yt).mean() for k in Z.files if k.startswith("Zte_") and k != "Zte_clean_0"]))
            for fl in np.unique(_FILM[te]):
                k = _FILM[te] == fl; film[fl] = (c[k].mean(), int(k.sum()))
        _R[(m, it)] = dict(acc=np.array(accs), ece=np.array(eces), mca=np.array(mcas), film=film)
def _cmp(a, b, it=3000):
    fa_, fb_ = _R[(a, it)]["film"], _R[(b, it)]["film"]; FL = sorted(fa_)
    n = np.array([fa_[f][1] for f in FL]); fa = np.array([fa_[f][0] for f in FL]); fb = np.array([fb_[f][0] for f in FL])
    pw = wilcoxon(fa, fb).pvalue
    rng = np.random.default_rng(0); idx = rng.integers(0, len(FL), (10000, len(FL)))
    d = (n * (fa - fb)).sum() / n.sum(); bs = (n[idx] * (fa[idx] - fb[idx])).sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(D=f"{d*100:+.2f}", CI=f"[{lo*100:.2f}, {hi*100:.2f}]", PW=f"{pw:.3f}", W=int((fa > fb).sum()), L=int((fa < fb).sum()))
_c1 = _cmp("prismr+paug", "resnet18"); _c2 = _cmp("prismr+paug", "vgg11_bn")
_c1k = _cmp("prismr+paug", "resnet18", 1000); _c2k = _cmp("prismr+paug", "vgg11_bn", 1000)
json.dump({f"{m}_{it}": dict(acc=v["acc"].tolist(), ece=v["ece"].tolist(), mca=v["mca"].tolist()) for (m, it), v in _R.items()} |
          dict(cmp3000_rn=_c1, cmp3000_vgg=_c2, cmp1000_rn=_c1k, cmp1000_vgg=_c2k), open("long_summary.json", "w"), indent=1)
_t = ["\\footnotesize", "\\setlength{\\tabcolsep}{2.4pt}", "\\begin{tabular}{@{}llccccccc@{}}", "\\toprule",
      "Model & Steps & F1 & F2 & F3 & F4 & F5 & Acc & ECE / mCA\\\\", "\\midrule"]
for m, nm, lt in _M:
    for it in (1000, 3000):
        v = _R[(m, it)]
        _t.append(f"{nm if it == 1000 else ''} & {it} & " + " & ".join(f"{x*100:.1f}" for x in v["acc"]) +
                  f" & ${v['acc'].mean()*100:.1f}\\pm{v['acc'].std(ddof=1)*100:.1f}$ & {v['ece'].mean()*100:.1f} / {v['mca'].mean()*100:.1f}\\\\")
    if m != _M[-1][0]: _t.append("\\midrule")
_t += ["\\bottomrule", "\\end{tabular}"]
open(os.path.join(L, "tab_long.tex"), "w").write("\n".join(_t) + "\n")
for m, nm, lt in _M:
    k = {"prismr+paug": "Prism", "resnet18": "RN", "vgg11_bn": "Vgg"}[m]
    for it, s in ((1000, "K"), (3000, "T")):
        v = _R[(m, it)]
        LONG[f"Lg{k}{s}"] = f"{v['acc'].mean()*100:.1f}"; LONG[f"Lg{k}{s}ECE"] = f"{v['ece'].mean()*100:.1f}"; LONG[f"Lg{k}{s}MCA"] = f"{v['mca'].mean()*100:.1f}"
for nm, c in (("RN", _c1), ("Vgg", _c2), ("RNK", _c1k), ("VggK", _c2k)):
    LONG[f"LgD{nm}"] = c["D"]; LONG[f"LgCI{nm}"] = c["CI"]; LONG[f"LgPW{nm}"] = c["PW"]; LONG[f"LgW{nm}"] = str(c["W"]); LONG[f"LgL{nm}"] = str(c["L"])
