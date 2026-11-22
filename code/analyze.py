"""Compute every number reported in the paper from saved logits -> summary.json."""
import glob, json, math, os
import numpy as np
import torch, torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, recall_score, confusion_matrix, balanced_accuracy_score

R = "./results"
D = np.load("./riawelc_80.npz")
Y = D["y"]; FILM = D["film"]
K = 4
CLS = ["ND", "PO", "LP", "CR"]


def probs(Z, lt, T=1.0):
    Z = torch.from_numpy(Z).double()
    if lt == "edl":
        a = F.softplus(Z) + 1.0
        return (a / a.sum(1, keepdim=True)).numpy(), (K / a.sum(1)).numpy()
    p = F.softmax(Z / T, 1).numpy()
    return p, 1 - p.max(1)


def ece(p, y, B=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y)
    edges = np.linspace(0, 1, B + 1); e = 0.0; rel = []
    for i in range(B):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
            rel.append((float(conf[m].mean()), float(acc[m].mean()), int(m.sum())))
    return e, rel


def aurc(conf, correct):
    o = np.argsort(-conf, kind="stable"); c = correct[o]
    risk = np.cumsum(1 - c) / np.arange(1, len(c) + 1)
    return risk.mean(), risk


def conformal(pv, yv, pt, yt, alpha):
    n = len(yv)
    s = 1 - pv[np.arange(n), yv]
    q = np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = pt >= 1 - q
    cov = sets[np.arange(len(yt)), yt].mean()
    return float(cov), float(sets.sum(1).mean()), float((sets.sum(1) == 1).mean())


def fit_T(Zv, yv):
    Z = torch.from_numpy(Zv).double(); yy = torch.from_numpy(yv)
    lt = torch.zeros(1, dtype=torch.double, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=200)

    def cl():
        opt.zero_grad(); l = F.cross_entropy(Z / lt.exp(), yy); l.backward(); return l
    opt.step(cl)
    return float(lt.exp())


def metrics(p, y):
    pred = p.argmax(1)
    e, rel = ece(p, y)
    return dict(acc=float((pred == y).mean()), mf1=float(f1_score(y, pred, average="macro")),
                bacc=float(balanced_accuracy_score(y, pred)),
                f1_cls=f1_score(y, pred, average=None, labels=range(K)).tolist(),
                rec_cls=recall_score(y, pred, average=None, labels=range(K)).tolist(),
                ece=float(e), nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
                brier=float(((p - np.eye(K)[y]) ** 2).sum(1).mean()))


runs = {}
for jf in sorted(glob.glob(f"{R}/*.json")):
    J = json.load(open(jf))
    if "args" not in J: continue
    a = J["args"]
    tag = os.path.basename(jf)[:-5]
    Z = np.load(jf[:-5] + ".npz")
    lt = J["loss_type"]; te = Z["te_idx"]; va = Z["va_idx"]
    yt, yv = Y[te], Y[va]
    pt, ut = probs(Z["Zte_clean_0"], lt); pv, _ = probs(Z["Zva"], lt)
    m = metrics(pt, yt)
    m["cm"] = confusion_matrix(yt, pt.argmax(1), labels=range(K)).tolist()
    m["aurc"], risk = aurc(pt.max(1), (pt.argmax(1) == yt).astype(float))
    m["aurc"] = float(m["aurc"])
    if lt == "edl":
        m["aurc_u"] = float(aurc(-ut, (pt.argmax(1) == yt).astype(float))[0])
    _, m["reliability"] = ece(pt, yt)
    for al in (0.05, 0.10):
        m[f"cp{int(al*100)}"] = conformal(pv, yv, pt, yt, al)
    if lt == "ce":
        T = fit_T(Z["Zva"], yv); pts, _ = probs(Z["Zte_clean_0"], lt, T)
        m["T"] = T; m["ece_ts"] = float(ece(pts, yt)[0])
        m["nll_ts"] = float(-np.log(np.clip(pts[np.arange(len(yt)), yt], 1e-12, 1)).mean())
    rob = {}
    for k in Z.files:
        if k.startswith("Zte_") and k != "Zte_clean_0":
            pc, _ = probs(Z[k], lt)
            rob[k[4:]] = float((pc.argmax(1) == yt).mean())
    m["rob"] = rob
    # per-film accuracy
    films = FILM[te]; pf = {}
    for f in np.unique(films):
        mm = films == f
        pf[str(f)] = [float((pt.argmax(1)[mm] == yt[mm]).mean()), int(mm.sum())]
    m["per_film"] = pf
    m["params"] = J["params"]; m["train_time"] = J["train_time"]
    m["risk_curve"] = risk[np.linspace(0, len(risk) - 1, 200).astype(int)].tolist()
    m["log"] = J["log"]
    runs[tag] = m

# aggregate over folds
agg = {}
models = sorted({t.split("_group_")[0] for t in runs if "_group_" in t})
for mdl in models:
    fs = [runs[f"{mdl}_group_f{f}_s0"] for f in range(5) if f"{mdl}_group_f{f}_s0" in runs]
    if not fs:
        continue
    A = {"n_folds": len(fs), "params": fs[0]["params"]}
    for key in ["acc", "mf1", "bacc", "ece", "nll", "brier", "aurc", "ece_ts", "nll_ts", "aurc_u", "train_time"]:
        v = [f[key] for f in fs if key in f]
        if v:
            A[key] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
    A["f1_cls"] = np.mean([f["f1_cls"] for f in fs], 0).tolist()
    A["f1_cls_std"] = np.std([f["f1_cls"] for f in fs], 0, ddof=1).tolist() if len(fs) > 1 else [0] * K
    A["rec_cls"] = np.mean([f["rec_cls"] for f in fs], 0).tolist()
    A["cm"] = np.sum([f["cm"] for f in fs], 0).tolist()
    for c in ("cp5", "cp10"):
        A[c] = np.mean([f[c] for f in fs], 0).tolist()
        A[c + "_std"] = np.std([f[c] for f in fs], 0, ddof=1).tolist() if len(fs) > 1 else [0, 0, 0]
    A["rob"] = {k: [float(np.mean([f["rob"][k] for f in fs])), float(np.std([f["rob"][k] for f in fs], ddof=1)) if len(fs) > 1 else 0.0]
                for k in fs[0]["rob"]}
    A["acc_folds"] = [f["acc"] for f in fs]
    A["mf1_folds"] = [f["mf1"] for f in fs]
    pf = {}
    for f in fs:
        pf.update(f["per_film"])
    A["per_film"] = pf
    # pooled reliability
    A["risk_curve"] = np.mean([f["risk_curve"] for f in fs], 0).tolist()
    agg[mdl] = A

# Wilcoxon signed-rank over films: final model (REF) vs every other model
REF = "prismr+paug"
if REF in agg:
    base = agg[REF]["per_film"]
    for mdl, A in agg.items():
        if mdl == REF or len(A["per_film"]) != len(base):
            continue
        fl = sorted(base)
        a = np.array([base[f][0] for f in fl]); b = np.array([A["per_film"][f][0] for f in fl])
        if np.allclose(a, b):
            continue
        st = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        A["wilcoxon_p_vs_ref"] = float(st.pvalue)
        A["films_ref_better"] = int((a > b).sum()); A["films_ref_worse"] = int((a < b).sum())

off = {}
for t, m in runs.items():
    if "_official_" in t:
        off[t.split("_official_")[0]] = {k: m[k] for k in ("acc", "mf1", "bacc", "ece", "nll")}

json.dump(dict(runs=runs, agg=agg, official=off), open(f"{R}/../summary.json", "w"))
print("models:", {m: agg[m]["n_folds"] for m in agg}, "official:", list(off))
for mdl in agg:
    A = agg[mdl]
    print(f"{mdl:22s} acc {A['acc'][0]*100:6.2f}±{A['acc'][1]*100:4.2f}  mF1 {A['mf1'][0]*100:6.2f}  ECE {A['ece'][0]*100:5.2f}  "
          f"AURC {A['aurc'][0]*100:5.2f} cov90 {A['cp10'][0]:.3f}/{A['cp10'][1]:.2f}  p={A.get('wilcoxon_p_vs_ref','-')}")
for mdl, o in off.items():
    print("OFFICIAL", mdl, round(o["acc"] * 100, 2))
