# R9 (reviewer_protocol.md): final label-free chain on SWRD. Written before any R9 quantity was computed.
# CAA (content-aware alignment, linear form) -> AdaBN -> acquisition-uncertainty propagation (UP) -> conformal recalibration (25 radiographs).
import sys, os, json, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from models import build
torch.set_num_threads(4)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CFG = json.load(open("paper_cfg.json")); MODELS = CFG["main"] + CFG["paug_baselines"]
RW = np.load("riawelc_80.npz"); YR = RW["y"]
SW = np.load("swrd/swrd_tiles80.npz"); XS8, YS, IMG = SW["X"], SW["y"], SW["img"]
SP = json.load(open("swrd/swrd_split.json")); TE = np.isin(IMG, SP["test"]); UU = np.isin(IMG, SP["U"]); POOL = np.array(SP["pool"])
EDL = lambda m: m.split("+")[0].startswith("prism") and "_ce" not in m.split("+")[0]
ND_MEAN = float(RW["X"][::3][YR[::3] == 0].astype(np.float32).mean() / 255); ND_STD = float(RW["X"][::3][YR[::3] == 0].astype(np.float32).std() / 255)
XR = torch.from_numpy(np.load("swrd/swrd_pga_inputs.npz")["XR"].astype(np.float32) / 255.0).unsqueeze(1)
PB = np.load("swrd/swrd_probs_pga.npz")
IMS = np.unique(IMG); KIM = {im: np.where(IMG == im)[0] for im in IMS}
FF = XS8.astype(np.float32) / 255.0
MC, NB, EPS = 32, 200, 0.10


def to_p(Z, edl):
    Z = Z.double()
    if edl:
        a = F.softplus(Z) + 1; return (a / a.sum(1, keepdim=True)).numpy()
    return F.softmax(Z, 1).numpy()


def ece(p, y, B=15):
    conf = p.max(1); acc = p.argmax(1) == y; e = 0.0; ed = np.linspace(0, 1, B + 1)
    for i in range(B):
        m = (conf > ed[i]) & (conf <= ed[i + 1])
        if m.sum(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def scores(p, y):
    return dict(acc=float((p.argmax(1) == y).mean()), ece=ece(p, y), nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()))


@torch.no_grad()
def predict(net, X, bs=512):
    net.eval(); return torch.cat([net(X[k:k + bs].to(DEV)).cpu() for k in range(0, len(X), bs)])


@torch.no_grad()
def adabn(net, X):
    bns = [x for x in net.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
    for b in bns: b.reset_running_stats(); b.momentum = None
    net.train(); iu = np.where(UU)[0]
    for k in range(0, len(iu), 256): net(X[iu[k:k + 256]].to(DEV))
    net.eval()


def caa(sel, mu_sd=None):
    """linear content-aware alignment; sel: dict radiograph -> selected tile indices; mu_sd: optional dict of (mean, std)."""
    x = np.empty_like(FF)
    for im, k in KIM.items():
        mu, sd = mu_sd[im] if mu_sd is not None else (FF[sel[im]].mean(), FF[sel[im]].std())
        x[k] = np.clip((FF[k] - mu) / max(sd, 1e-3) * ND_STD + ND_MEAN, 0, 1)
    return torch.from_numpy(x).unsqueeze(1)


def qhat(p, y):
    n = len(y); s = 1 - p[np.arange(n), y]
    return np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - EPS)) / n), method="higher")


def setstats(p, y, q):
    sets = p >= 1 - q
    return sets[np.arange(len(y)), y].mean(), sets.sum(1).mean(), (sets.sum(1) == 1).mean()


rng = np.random.default_rng(0); NC = [25, 50, 100, 200]; REP = 200
DRAWS = {n: [rng.choice(POOL, n, replace=False) for _ in range(REP)] for n in NC}


def recal(Ps):
    st = np.array([setstats(P[TE], YS[TE], qhat(P[c], YS[c])) for d in DRAWS[25] for c in [np.isin(IMG, d)] for P in Ps])
    return dict(cov=float(st[:, 0].mean()), size=float(st[:, 1].mean()), single=float(st[:, 2].mean()))


def rad_boot_diff(ca, cb, B=10000):
    ims = np.unique(IMG[TE]); idx = [np.where(TE & (IMG == i))[0] for i in ims]
    sa = np.array([ca[k].sum() for k in idx], float); sb = np.array([cb[k].sum() for k in idx], float); n = np.array([len(k) for k in idx], float)
    d = (sa - sb).sum() / n.sum(); r = np.random.default_rng(0).integers(0, len(ims), (B, len(ims)))
    bs = (sa[r] - sb[r]).sum(1) / n[r].sum(1); lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(d=float(d), lo=float(lo), hi=float(hi))


def chain(load_net, edl, pr_R=None, pr_Rbn=None):
    """returns dict stage -> probs (all SWRD tiles) for one fold model."""
    net = load_net(); gpu = pr_R is None
    if gpu:
        pr_R = to_p(predict(net, XR), edl)
    sel = {im: (k[pr_R[k, 0] >= 0.9] if (pr_R[k, 0] >= 0.9).sum() >= 3 else k) for im, k in KIM.items()}
    out = {"caa": to_p(predict(net, caa(sel)), edl)}
    X0 = caa(sel); adabn(net, X0); out["caa_bn"] = to_p(predict(net, X0), edl)
    r = np.random.default_rng(0); boot = {im: [(lambda kk: (FF[kk].mean(), FF[kk].std()))(r.choice(s, len(s), replace=True)) for _ in range(NB)] for im, s in sel.items()}
    pick = np.random.default_rng(0).choice(NB, MC, replace=False)
    out["caa_bn_up"] = np.mean([to_p(predict(net, caa(sel, {im: boot[im][j] for im in IMS})), edl) for j in pick], 0)
    if gpu:
        net2 = load_net(); adabn(net2, XR); out["pgaR_bn"] = to_p(predict(net2, XR), edl)
    else:
        out["pgaR_bn"] = pr_Rbn
    out["pgaR"] = pr_R
    return out


def load_cpu(m, f):
    def fn():
        p = f"results/{m}_group_f{f}_s0.pt"; p = p if os.path.exists(p) else f"rerun/{m}_group_f{f}_s0.pt"
        net = build(m.split("+")[0], 80); net.load_state_dict(torch.load(p, map_location="cpu")); return net.to(DEV)
    return fn


def load_dir(d, f):
    def fn():
        net = build("prismr", 80); net.load_state_dict(torch.load(f"{d}/prismr+paug_group_f{f}_s0.pt", map_location="cpu")); return net.to(DEV)
    return fn


if __name__ == "__main__":
    STAGES = ["pgaR", "pgaR_bn", "caa", "caa_bn", "caa_bn_up"]
    groups = [(m, [load_cpu(m, f) for f in range(5)], EDL(m), [PB[f"{m}|R"][f] for f in range(5)], [PB[f"{m}|R_bn"][f] for f in range(5)]) for m in MODELS] + \
             [(f"gpu:{v}", [load_dir(d, f) for f in range(5)], True, [None] * 5, [None] * 5) for v, d in (("comp", "results_gpu_1000"), ("oc", "results_r8_oc"))]
    probs, S = {}, {}
    for name, loaders, edl, prR, prRbn in groups:
        res = {s: [] for s in STAGES}
        for f in range(5):
            o = chain(loaders[f], edl, prR[f], prRbn[f])
            for s in STAGES: res[s].append(o[s].astype(np.float32))
        for s in STAGES: probs[f"{name}|{s}"] = np.stack(res[s])
        S[name] = {s: {k: float(np.mean([scores(P[TE], YS[TE])[k] for P in res[s]])) for k in ("acc", "ece", "nll")} for s in STAGES}
        corr = {s: np.mean([(P.argmax(1) == YS) for P in res[s]], 0) for s in STAGES}
        S[name]["caa_bn-pgaR_bn"] = rad_boot_diff(corr["caa_bn"], corr["pgaR_bn"])
        S[name]["caa-pgaR"] = rad_boot_diff(corr["caa"], corr["pgaR"])
        S[name]["up-caa_bn"] = rad_boot_diff(corr["caa_bn_up"], corr["caa_bn"])
        S[name]["recal"] = {s: recal(res[s]) for s in ("pgaR_bn", "caa_bn", "caa_bn_up")}
        s_ = S[name]; print(f"{name:26s} acc pgaR {s_['pgaR']['acc']*100:5.1f} pgaR+bn {s_['pgaR_bn']['acc']*100:5.1f} caa {s_['caa']['acc']*100:5.1f} caa+bn {s_['caa_bn']['acc']*100:5.1f} +up {s_['caa_bn_up']['acc']*100:5.1f} | "
                            f"ECE {s_['pgaR_bn']['ece']*100:4.1f} {s_['caa_bn']['ece']*100:4.1f} {s_['caa_bn_up']['ece']*100:4.1f} | d(caa_bn-pgaR_bn) {s_['caa_bn-pgaR_bn']['d']*100:+.1f} [{s_['caa_bn-pgaR_bn']['lo']*100:.1f},{s_['caa_bn-pgaR_bn']['hi']*100:.1f}] | "
                            f"recal single {s_['recal']['pgaR_bn']['single']*100:4.1f} -> {s_['recal']['caa_bn_up']['single']*100:4.1f} cov {s_['recal']['caa_bn_up']['cov']*100:4.1f}", flush=True)
    np.savez_compressed("swrd/swrd_probs_r9.npz", **probs)
    json.dump(S, open("r9_summary.json", "w"), indent=1)
