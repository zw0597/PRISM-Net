# R7 (reviewer_protocol.md): two-stage decision layer with a controlled missed-defect rate.
# Stage 1: accept as sound iff p_ND > tau, tau calibrated on validation DEFECT tiles (finite-sample MDR <= alpha).
# Stage 2: Mondrian split conformal (score 1-p_k) on the non-accepted tiles. No retraining; existing predictions only.
# Output: r7_summary.json (all numbers used in the paper). Usage: python r7_decision.py
import json, math
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon

D = np.load("riawelc_80.npz"); Y = D["y"]; FILM = D["film"]
CFG = json.load(open("paper_cfg.json")); MODELS = CFG["main"] + CFG["paug_baselines"]
NAME = dict(CFG["names"]); NAME.update({"prismr+paug": "PRISM-Net", "prismx+paug": "PRISM-Net-Lite"})
ND, K = 0, 4
ALPHAS, EPSS = (0.01, 0.02, 0.05), (0.10, 0.05)
PRIMARY = (0.02, 0.10)

def probs(z, lt):
    z = torch.from_numpy(z).double()
    if lt == "edl":
        a = F.softplus(z) + 1.0; return (a / a.sum(1, keepdim=True)).numpy(), ((a - 1) / a.sum(1, keepdim=True)).numpy()
    p = F.softmax(z, 1).numpy(); return p, p

def kth(scores, level):
    """Conformal order statistic: the ceil((n+1)*level)-th smallest score, +inf if it does not exist."""
    n = len(scores); k = math.ceil((n + 1) * level)
    return np.inf if (n == 0 or k > n) else np.sort(scores)[k - 1]

def calibrate(p, y, alpha, eps, s=None):
    s = p[:, ND] if s is None else s
    tau = kth(s[y != ND], 1 - alpha)
    na = ~(s > tau); q = np.array([kth(1 - p[na & (y == k), k], 1 - eps) for k in range(K)])
    return tau, q

def decide(p, tau, q, s=None):
    s = p[:, ND] if s is None else s
    acc = s > tau
    C = (1 - p) <= q[None, :]; C[acc] = False; C[acc, ND] = True
    single = C.sum(1) == 1
    return acc, C, single

def marginal_q(p, y, eps):
    return kth(1 - p[np.arange(len(y)), y], 1 - eps)

def mondrian_q(p, y, eps):
    return np.array([kth(1 - p[y == k, k], 1 - eps) for k in range(K)])

def metrics(p, y, acc, C, single, film=None):
    dfc = y != ND
    out = dict(mdr=float(acc[dfc].mean()), mdr_cls=[float(acc[y == k].mean()) for k in range(1, K)],
               ar=float(acc[y == ND].mean()),
               cov=[float(C[y == k, k].mean()) for k in range(K)],
               cov_na=[float(C[(~acc) & (y == k), k].mean()) if ((~acc) & (y == k)).any() else float("nan") for k in range(K)],
               auto=float((acc | single).mean()), refer=float((~acc & ~single).mean()),
               size=float(C[~acc].sum(1).mean()) if (~acc).any() else float("nan"),
               auto_err=float(((acc & dfc) | (~acc & single & ~C[np.arange(len(y)), y])).mean()))
    return out

# ------------------------------------------------------------------ RIAWELC
R = {}
for m in MODELS:
    per = {}
    for f in range(5):
        J = json.load(open(f"results/{m}_group_f{f}_s0.json")); lt = J["loss_type"]
        Z = np.load(f"results/{m}_group_f{f}_s0.npz"); va, te = Z["va_idx"], Z["te_idx"]
        pv, bv = probs(Z["Zva"], lt); pt, bt = probs(Z["Zte_clean_0"], lt); yv, yt = Y[va], Y[te]; ft = FILM[te]
        for a in ALPHAS:
            for e in EPSS:
                tau, q = calibrate(pv, yv, a, e); acc, C, sg = decide(pt, tau, q)
                r = metrics(pt, yt, acc, C, sg)
                r["tau_inf"] = bool(np.isinf(tau))
                r["film"] = {fl: (float(acc[(ft == fl) & (yt != ND)].mean()) if ((ft == fl) & (yt != ND)).any() else None,
                                  int(((ft == fl) & (yt != ND)).sum()),
                                  float(acc[(ft == fl) & (yt == ND)].mean()) if ((ft == fl) & (yt == ND)).any() else None,
                                  int(((ft == fl) & (yt == ND)).sum()),
                                  float((acc | sg)[ft == fl].mean()), int((ft == fl).sum())) for fl in np.unique(ft)}
                per.setdefault(f"R7|{a}|{e}", []).append(r)
                if lt == "edl" and e == 0.10:
                    tau_b, q_b = calibrate(pv, yv, a, e, s=bv[:, ND]); acc_b, C_b, sg_b = decide(pt, tau_b, q_b, s=bt[:, ND])
                    per.setdefault(f"R7belief|{a}|{e}", []).append(metrics(pt, yt, acc_b, C_b, sg_b))
        for e in (0.10, 0.05, 0.02, 0.01):
            qm = marginal_q(pv, yv, e); C = (1 - pt) <= qm; acc = (C.sum(1) == 1) & C[:, ND]
            sg = C.sum(1) == 1; C2 = C.copy()
            per.setdefault(f"B1|{e}", []).append(metrics(pt, yt, acc, C2, sg))
        qd = mondrian_q(pv, yv, 0.10); C = (1 - pt) <= qd[None, :]; acc = (C.sum(1) == 1) & C[:, ND]
        per.setdefault("B2|0.1", []).append(metrics(pt, yt, acc, C, C.sum(1) == 1))
    R[m] = per

def agg(rs, key):
    v = np.array([r[key] for r in rs], dtype=float); return v.mean(0).tolist()

summary = {"riawelc": {}}
for m in MODELS:
    summary["riawelc"][m] = {}
    for k, rs in R[m].items():
        summary["riawelc"][m][k] = {key: agg(rs, key) for key in ("mdr", "mdr_cls", "ar", "cov", "cov_na", "auto", "refer", "size", "auto_err")}
        if k.startswith("R7|"):
            films = {}
            for r in rs: films.update(r["film"])
            mdrs = [v[0] for v in films.values() if v[0] is not None]
            summary["riawelc"][m][k]["film_mdr"] = {fl: v for fl, v in films.items()}
            summary["riawelc"][m][k]["film_mdr_max"] = float(max(mdrs)); summary["riawelc"][m][k]["film_frac_above"] = float(np.mean(np.array(mdrs) > float(k.split("|")[1])))
            summary["riawelc"][m][k]["tau_inf_folds"] = int(sum(r["tau_inf"] for r in rs))

def film_cmp(a, b, key):           # key: 2 -> AR (ND tiles, idx 2/3), 4 -> automatic fraction (all tiles, idx 4/5)
    A = summary["riawelc"][a][f"R7|{PRIMARY[0]}|{PRIMARY[1]}"]["film_mdr"]; B = summary["riawelc"][b][f"R7|{PRIMARY[0]}|{PRIMARY[1]}"]["film_mdr"]
    FL = sorted(fl for fl in A if A[fl][key] is not None and B[fl][key] is not None)
    n = np.array([A[fl][key + 1] for fl in FL], float); fa = np.array([A[fl][key] for fl in FL]); fb = np.array([B[fl][key] for fl in FL])
    pw = float(wilcoxon(fa, fb).pvalue) if np.any(fa != fb) else 1.0
    rng = np.random.default_rng(0); idx = rng.integers(0, len(FL), (10000, len(FL)))
    d = (n * (fa - fb)).sum() / n.sum(); bs = (n[idx] * (fa[idx] - fb[idx])).sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(d=float(d), lo=float(lo), hi=float(hi), pw=pw, W=int((fa > fb).sum()), L=int((fa < fb).sum()), films=len(FL))
summary["cmp"] = {f"{b}|{nm}": film_cmp("prismr+paug", b, key) for b in ("resnet18", "vgg11_bn") for key, nm in ((2, "ar"), (4, "auto"))}

# ------------------------------------------------------------------ SWRD
S = np.load("swrd/swrd_tiles80.npz"); YS, IMG = S["y"], S["img"]; sp = json.load(open("swrd/swrd_split.json"))
te = np.isin(IMG, sp["test"]); pool = np.array(sp["pool"])
rng = np.random.default_rng(0); NC = [25, 50, 100, 200]; REP = 200
draws = {n: [rng.choice(pool, n, replace=False) for _ in range(REP)] for n in NC}
P0 = np.load("swrd/swrd_probs.npz"); PB = np.load("swrd/swrd_probs_pga.npz")
summary["swrd"] = {}
for m in MODELS:
    for cond, Pm in (("frozen", P0[m]), ("pga_bn", PB[f"{m}|R_bn"])):
        rows = []
        for d in draws[25]:
            c = np.isin(IMG, d)
            for f in range(5):
                tau, q = calibrate(Pm[f][c], YS[c], *PRIMARY); acc, C, sg = decide(Pm[f][te], tau, q)
                r = metrics(Pm[f][te], YS[te], acc, C, sg); r["tau_inf"] = float(np.isinf(tau)); r["n_def_cal"] = int((YS[c] != ND).sum())
                rows.append(r)
        mdr = np.array([r["mdr"] for r in rows])
        summary["swrd"][f"{m}|{cond}"] = dict(mdr=float(mdr.mean()), mdr_p90=float(np.percentile(mdr, 90)), frac_above=float((mdr > PRIMARY[0]).mean()),
            ar=float(np.mean([r["ar"] for r in rows])), auto=float(np.mean([r["auto"] for r in rows])),
            refer=float(np.mean([r["refer"] for r in rows])), cov_na=np.nanmean([r["cov_na"] for r in rows], 0).tolist(),
            tau_inf=float(np.mean([r["tau_inf"] for r in rows])), n_def_cal=float(np.mean([r["n_def_cal"] for r in rows])))
json.dump(summary, open("r7_summary.json", "w"), indent=1, default=str)

a, e = PRIMARY
print(f"RIAWELC alpha={a} eps={e}: model | MDR | film max | films>alpha | AR | cov ND PO LP CR | auto | refer || B1(0.10) MDR AR LP-cov | B1(alpha) MDR AR")
for m in MODELS:
    v = summary["riawelc"][m][f"R7|{a}|{e}"]; b = summary["riawelc"][m]["B1|0.1"]; b2 = summary["riawelc"][m][f"B1|{a}"]
    print(f"{NAME[m]:22s} {v['mdr']*100:5.2f} {v['film_mdr_max']*100:5.1f} {v['film_frac_above']:.2f} {v['ar']*100:5.1f} | "
          + " ".join(f"{x*100:5.1f}" for x in v["cov"]) + f" | {v['auto']*100:5.1f} {v['refer']*100:5.1f} || "
          f"{b['mdr']*100:5.2f} {b['ar']*100:5.1f} {b['cov'][2]*100:5.1f} | {b2['mdr']*100:5.2f} {b2['ar']*100:5.1f}")
print(summary["cmp"])
for k in ("prismr+paug|frozen", "prismr+paug|pga_bn", "resnet18|frozen", "resnet18|pga_bn", "vgg11_bn|pga_bn"):
    print("SWRD", k, {kk: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for kk, v in summary["swrd"][k].items()})
