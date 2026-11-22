"""E1: ImageNet-initialised models at 112 px (risk_protocol.md)."""
import json, numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
W = "./"; D = np.load(W + "riawelc_80.npz"); Y = D["y"]; FILM = D["film"]
MODELS = ["prismrpt+paug", "resnet18pt", "resnet18pt+paug"]


def probs(Z, edl):
    Z = torch.from_numpy(Z).double()
    if edl:
        a = F.softplus(Z) + 1; return (a / a.sum(1, keepdim=True)).numpy()
    return F.softmax(Z, 1).numpy()


def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


R = {}
for m in MODELS:
    acc, mf1, ec, nll, C, TE = [], [], [], [], [], []
    for f in range(5):
        Z = np.load(f"{W}results_pt/{m}_group_f{f}_s0.npz"); te = Z["te_idx"]; p = probs(Z["Zte_clean_0"], m.startswith("prism"))
        y = Y[te]; pr = p.argmax(1)
        acc.append((pr == y).mean()); mf1.append(f1_score(y, pr, average="macro")); ec.append(ece(p, y))
        nll.append(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()); C.append(pr == y); TE.append(te)
    R[m] = dict(acc=acc, mf1=mf1, ece=ec, nll=nll, c=np.concatenate(C), te=np.concatenate(TE))
te = R[MODELS[0]]["te"]; FL = np.unique(FILM[te]); n = np.array([(FILM[te] == x).sum() for x in FL])
rng = np.random.default_rng(0); idx = rng.integers(0, len(FL), (10000, len(FL)))
out = {}
for m in MODELS:
    r = R[m]; out[m] = {k: [float(x) for x in r[k]] for k in ("acc", "mf1", "ece", "nll")}
    print(f"{m:18s} acc {np.mean(r['acc'])*100:.2f}±{np.std(r['acc'],ddof=1)*100:.2f} mF1 {np.mean(r['mf1'])*100:.2f} ECE {np.mean(r['ece'])*100:.2f} NLL {np.mean(r['nll']):.3f} folds {np.round(np.array(r['acc'])*100,1)}")
for m in MODELS[1:]:
    assert np.array_equal(R[m]["te"], te)
    fa = np.array([R[MODELS[0]]["c"][FILM[te] == x].mean() for x in FL]); fb = np.array([R[m]["c"][FILM[te] == x].mean() for x in FL])
    d = (n * (fa - fb)).sum() / n.sum(); bs = (n[idx] * (fa[idx] - fb[idx])).sum(1) / n[idx].sum(1)
    pw = wilcoxon(fa, fb).pvalue; lo, hi = np.percentile(bs, [2.5, 97.5])
    wins = int((np.array(R[MODELS[0]]["acc"]) > np.array(R[m]["acc"])).sum())
    out[m].update(d=float(d), ci=[float(lo), float(hi)], pW=float(pw), fold_wins=wins, film_w=int((fa > fb).sum()), film_l=int((fa < fb).sum()))
    print(f"PRISM-pt - {m}: d {d*100:+.2f} [{lo*100:.2f}, {hi*100:.2f}] pW {pw:.3f} fold wins {wins}/5 films {int((fa>fb).sum())}/{int((fa<fb).sum())}")
json.dump(out, open(W + "results/pt_summary.json", "w"), indent=1)
