# R8 evaluation (reviewer_protocol.md, R8). Written before any R8 result.
#   python r8_eval.py infer     -> (C) acquisition-uncertainty propagation and (D) content-robust PGA for the 15 CPU models on SWRD
#   python r8_eval.py trained   -> (A) oc, (B) canon, (A+B) occanon and (B0) vs the R4 GPU-1000 PRISM-Net on RIAWELC and SWRD
import sys, os, json, math
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import wilcoxon
sys.path.insert(0, "."); from models import build
torch.set_num_threads(4)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CFG = json.load(open("paper_cfg.json")); MODELS = CFG["main"] + CFG["paug_baselines"]
NAME = dict(CFG["names"]); NAME.update({"prismr+paug": "PRISM-Net", "prismx+paug": "PRISM-Net-Lite"})
RW = np.load("riawelc_80.npz"); YR, FILM = RW["y"], RW["film"]
SW = np.load("swrd/swrd_tiles80.npz"); XS8, YS, IMG = SW["X"], SW["y"], SW["img"]
SP = json.load(open("swrd/swrd_split.json")); TE = np.isin(IMG, SP["test"]); VV = np.isin(IMG, SP["V"]); UU = np.isin(IMG, SP["U"])
EDL = lambda m: m.split("+")[0].startswith("prism") and "_ce" not in m.split("+")[0]


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


def load(path, base):
    net = build(base, 80); net.load_state_dict(torch.load(path, map_location="cpu")); return net.to(DEV)


def lstats(x8):
    a = np.log(np.maximum(x8.astype(np.float32), 1.0) / 255.0); m = np.median(a); return float(m), float(1.4826 * np.median(np.abs(a - m)))


def pga_align(x8, mt, st, ref):
    ms, ss = ref; g = ss / max(st, 1e-3); lc = ms - g * mt
    return np.clip(np.exp(lc + g * np.log(np.maximum(x8.astype(np.float32), 1.0) / 255.0)), 0, 1).astype(np.float32)


def rad_boot_diff(ca, cb, mask, B=10000):
    """tile-weighted accuracy difference with a radiograph bootstrap over the radiographs in mask."""
    ims = np.unique(IMG[mask]); idx = [np.where(mask & (IMG == i))[0] for i in ims]
    sa = np.array([ca[k].sum() for k in idx], float); sb = np.array([cb[k].sum() for k in idx], float); n = np.array([len(k) for k in idx], float)
    d = (sa - sb).sum() / n.sum(); r = np.random.default_rng(0).integers(0, len(ims), (B, len(ims)))
    bs = (sa[r] - sb[r]).sum(1) / n[r].sum(1); lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(d=float(d), lo=float(lo), hi=float(hi))


REF_ALL = lstats(RW["X"][::3])                       # paper's PGA reference (swrd_pga.py)
REF_ND = lstats(RW["X"][::3][YR[::3] == 0])          # (D) reference from sound RIAWELC tiles only
RMEAN, RSTD = float(RW["X"][::3].astype(np.float32).mean() / 255), float(RW["X"][::3].astype(np.float32).std() / 255)
ND_MEAN, ND_STD = float(RW["X"][::3][YR[::3] == 0].astype(np.float32).mean() / 255), float(RW["X"][::3][YR[::3] == 0].astype(np.float32).std() / 255)
XR = np.load("swrd/swrd_pga_inputs.npz")["XR"].astype(np.float32) / 255.0      # paper's PGA-R inputs

if sys.argv[1] == "infer":
    PB = np.load("swrd/swrd_probs_pga.npz"); out = {}
    # (C) bootstrap PGA parameters per radiograph: 200 resamples, 32 used
    rng = np.random.default_rng(0); ims = np.unique(IMG); MC = 32
    boot = {}
    for im in ims:
        k = np.where(IMG == im)[0]; pars = []
        for _ in range(200):
            kk = rng.choice(k, len(k), replace=True); pars.append(lstats(XS8[kk]))
        boot[im] = pars
    sel = np.random.default_rng(0).choice(200, MC, replace=False)
    XC = []
    for j in sel:
        x = np.empty(XS8.shape, np.float32)
        for im in ims:
            k = IMG == im; mt, st = boot[im][j]; x[k] = pga_align(XS8[k], mt, st, REF_ALL)
        XC.append(torch.from_numpy((x * 255).round().astype(np.uint8)).unsqueeze(1))
    for m in MODELS:
        base = m.split("+")[0]; e = EDL(m); PC, PD, PZ = [], [], []
        for f in range(5):
            p = f"results/{m}_group_f{f}_s0.pt"; p = p if os.path.exists(p) else f"rerun/{m}_group_f{f}_s0.pt"
            net = load(p, base)
            PC.append(np.mean([to_p(predict(net, X.float() / 255), e) for X in XC], 0))
            pr = PB[f"{m}|R"][f]; xd = np.empty(XS8.shape, np.float32); xz = np.empty(XS8.shape, np.float32)
            for im in ims:
                k = np.where(IMG == im)[0]; s = k[pr[k, 0] >= 0.9]; s = s if len(s) >= 3 else k
                mt, st = lstats(XS8[s]); xd[k] = pga_align(XS8[k], mt, st, REF_ND)
                fsel = XS8[s].astype(np.float32) / 255; ff = XS8[k].astype(np.float32) / 255
                xz[k] = np.clip((ff - fsel.mean()) / max(fsel.std(), 1e-3) * ND_STD + ND_MEAN, 0, 1)
            PD.append(to_p(predict(net, torch.from_numpy(xd).unsqueeze(1)), e)); PZ.append(to_p(predict(net, torch.from_numpy(xz).unsqueeze(1)), e))
        out[f"{m}|C"] = np.stack(PC).astype(np.float32); out[f"{m}|D"] = np.stack(PD).astype(np.float32); out[f"{m}|ZSCR"] = np.stack(PZ).astype(np.float32)
        print(m, flush=True)
    np.savez_compressed("swrd/swrd_probs_r8.npz", **out)
    S = {}
    for m in MODELS:
        S[m] = {}
        for v, P in (("R", PB[f"{m}|R"]), ("C", out[f"{m}|C"]), ("D", out[f"{m}|D"]), ("ZSCR", out[f"{m}|ZSCR"])):
            S[m][v] = {part: {k: float(np.mean([scores(P[f][msk], YS[msk])[k] for f in range(5)])) for k in ("acc", "ece", "nll")}
                       for part, msk in (("test", TE), ("val", VV))}
            S[m][v]["corr_test"] = np.mean([(P[f][TE].argmax(1) == YS[TE]) for f in range(5)], 0).tolist()
        for v in ("C", "D"):
            for w in ("R", "ZSCR") if v == "D" else ("R",):
                ca = np.array(S[m][v]["corr_test"]); cb = np.array(S[m][w]["corr_test"])
                full_a = np.zeros(len(YS)); full_b = np.zeros(len(YS)); full_a[TE] = ca; full_b[TE] = cb
                S[m][f"{v}-{w}"] = rad_boot_diff(full_a, full_b, TE)
        for v in ("R", "C", "D", "ZSCR"): S[m][v].pop("corr_test")
    json.dump(S, open("r8_infer_summary.json", "w"), indent=1)
    for m in MODELS:
        s = S[m]; print(f"{NAME[m]:22s} test acc R {s['R']['test']['acc']*100:5.1f} C {s['C']['test']['acc']*100:5.1f} D {s['D']['test']['acc']*100:5.1f} ZSCR {s['ZSCR']['test']['acc']*100:5.1f} | "
                        f"ECE R {s['R']['test']['ece']*100:5.1f} C {s['C']['test']['ece']*100:5.1f} D {s['D']['test']['ece']*100:5.1f} | val acc R {s['R']['val']['acc']*100:5.1f} C {s['C']['val']['acc']*100:5.1f} D {s['D']['val']['acc']*100:5.1f} | "
                        f"C-R {s['C-R']['d']*100:+.1f} [{s['C-R']['lo']*100:.1f},{s['C-R']['hi']*100:.1f}] D-R {s['D-R']['d']*100:+.1f} [{s['D-R']['lo']*100:.1f},{s['D-R']['hi']*100:.1f}] D-ZSCR {s['D-ZSCR']['d']*100:+.1f}")

if sys.argv[1] == "trained":
    sys.path.insert(0, ".")
    CORR = [("clean", 0)] + [("gamma", s) for s in (0.5, 0.7, 1.5, 2.0)] + [("gain", s) for s in (0.4, 0.7)] + \
           [("noise", s) for s in (0.02, 0.05, 0.08)] + [("blur", s) for s in (0.8, 1.5, 2.5)] + [("contrast", s) for s in (0.7, 0.5, 0.3)]
    RID = np.array([f"{a}|{b}" for a, b in zip(RW["film"], RW["view"])])
    V = {"comp": "results_gpu_1000", "oc": "results_r8_oc", "canon": "results_r8_canon", "occanon": "results_r8_occanon", "B0": None}
    def corrupt(x, kind, s, g):                              # identical to train.py
        if kind == "clean": return x
        if kind == "gamma": return x.clamp_min(1e-4) ** s
        if kind == "gain": return (x * s).clamp(0, 1)
        if kind == "noise": return (x + s * torch.randn(x.shape, generator=g)).clamp(0, 1)
        if kind == "blur":
            r = int(math.ceil(3 * s)); t = torch.arange(-r, r + 1).float(); k1 = torch.exp(-t ** 2 / (2 * s * s)); k1 /= k1.sum()
            xx = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1)); return F.conv2d(F.pad(xx, (0, 0, r, r), mode="reflect"), k1.view(1, 1, -1, 1))
        if kind == "contrast":
            mm = x.mean((2, 3), keepdim=True); return (mm + s * (x - mm)).clamp(0, 1)
    def canon_t(x, rid, ref):                                 # identical to train_r8.py
        ms, ss = ref; out = torch.empty_like(x); a = torch.log(x.clamp_min(1.0 / 255.0))
        for r in np.unique(rid):
            k = torch.from_numpy(np.where(rid == r)[0]); ar = a[k]; m = ar.median(); s = 1.4826 * (ar - m).abs().median()
            g = ss / max(float(s), 1e-3); lc = ms - g * float(m); out[k] = torch.exp(lc + g * ar).clamp(0, 1)
        return out
    X8 = torch.from_numpy(RW["X"]).unsqueeze(1)
    R = {v: [] for v in V}; SWR = {v: {} for v in V}
    for f in range(5):
        Sf = json.load(open("splits.json"))[str(f)]; tr, va, te = np.array(Sf["train"]), np.array(Sf["val"]), np.array(Sf["test"])
        a = torch.log((X8[tr[::3]].float() / 255).clamp_min(1.0 / 255.0)).flatten(); mref = a.median(); REFf = (float(mref), float(1.4826 * (a - mref).abs().median()))
        for v, d in V.items():
            if v == "B0":
                net = load(f"results_gpu_1000/prismr+paug_group_f{f}_s0.pt", "prismr")
                g = torch.Generator().manual_seed(1234); Xte = X8[te].float() / 255
                Zte = {f"{k}_{s}": predict(net, canon_t(corrupt(Xte, k, s, g), RID[te], REFf)) for k, s in CORR}
                Zva = predict(net, canon_t(X8[va].float() / 255, RID[va], REFf)); canon = True
            else:
                Z = np.load(f"{d}/prismr+paug_group_f{f}_s0.npz"); Zte = {k[4:]: torch.from_numpy(Z[k]) for k in Z.files if k.startswith("Zte_")}
                Zva = torch.from_numpy(Z["Zva"]); canon = v in ("canon", "occanon")
                net = load(f"{d}/prismr+paug_group_f{f}_s0.pt", "prismr")
            pt = to_p(Zte["clean_0"], True); pv = to_p(Zva, True); yt = YR[te]
            c = pt.argmax(1) == yt
            mf1 = np.mean([2 * ((pt.argmax(1) == k) & (yt == k)).sum() / max(((pt.argmax(1) == k).sum() + (yt == k).sum()), 1) for k in range(4)])
            film = {fl: (float(c[FILM[te] == fl].mean()), int((FILM[te] == fl).sum())) for fl in np.unique(FILM[te])}
            R[v].append(dict(acc=float(c.mean()), mf1=float(mf1), ece=ece(pt, yt), val_acc=float((pv.argmax(1) == YR[va]).mean()),
                             mca=float(np.mean([(to_p(Zte[k], True).argmax(1) == yt).mean() for k in Zte if k != "clean_0"])), film=film))
            # SWRD: frozen (raw for non-canonical models, own canonicalization for canonical ones), PGA-R, PGA-R + AdaBN
            if canon:
                xr = np.empty(XS8.shape, np.float32)
                for im in np.unique(IMG):
                    k = IMG == im; mt, st = lstats(XS8[k]); xr[k] = pga_align(XS8[k], mt, st, REFf)
                XIN = {"frozen": torch.from_numpy(xr).unsqueeze(1)}
            else:
                XIN = {"frozen": torch.from_numpy(XS8.astype(np.float32) / 255).unsqueeze(1), "pgaR": torch.from_numpy(XR).unsqueeze(1)}
            for nm, X in list(XIN.items()):
                SWR[v].setdefault(nm, []).append(to_p(predict(net, X), True))
            Xb = XIN["frozen"] if canon else XIN["pgaR"]; adabn(net, Xb); SWR[v].setdefault("bn", []).append(to_p(predict(net, Xb), True))
        print("fold", f, {v: round(R[v][-1]["acc"], 4) for v in V}, flush=True)
    out = {}
    for v in V:
        out[v] = {k: float(np.mean([r[k] for r in R[v]])) for k in ("acc", "mf1", "ece", "val_acc", "mca")}
        fa = {}; [fa.update(r["film"]) for r in R[v]]; out[v]["film"] = fa
        out[v]["swrd"] = {nm: {part: {k: float(np.mean([scores(P[msk], YS[msk])[k] for P in Ps])) for k in ("acc", "ece")} for part, msk in (("test", TE), ("val", VV))}
                          for nm, Ps in SWR[v].items()}
        out[v]["swrd_corr"] = {nm: np.mean([(P[TE].argmax(1) == YS[TE]) for P in Ps], 0) for nm, Ps in SWR[v].items()}
    def film_cmp(a, b):
        FL = sorted(out[a]["film"]); n = np.array([out[a]["film"][x][1] for x in FL], float)
        fa = np.array([out[a]["film"][x][0] for x in FL]); fb = np.array([out[b]["film"][x][0] for x in FL])
        r = np.random.default_rng(0).integers(0, len(FL), (10000, len(FL))); d = (n * (fa - fb)).sum() / n.sum(); bs = (n[r] * (fa[r] - fb[r])).sum(1) / n[r].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5]); return dict(d=float(d), lo=float(lo), hi=float(hi), pw=float(wilcoxon(fa, fb).pvalue) if np.any(fa != fb) else 1.0, W=int((fa > fb).sum()), L=int((fa < fb).sum()))
    cmp = {v: film_cmp(v, "comp") for v in V if v != "comp"}
    sw = {}
    for v in V:
        if v == "comp": continue
        for nm_v, nm_c in (("frozen", "frozen"), ("frozen", "pgaR") if v in ("canon", "occanon", "B0") else ("pgaR", "pgaR"), ("bn", "bn")):
            if nm_v in out[v]["swrd_corr"] and nm_c in out["comp"]["swrd_corr"]:
                a = np.zeros(len(YS)); b = np.zeros(len(YS)); a[TE] = out[v]["swrd_corr"][nm_v]; b[TE] = out["comp"]["swrd_corr"][nm_c]
                sw[f"{v}|{nm_v}-comp|{nm_c}"] = rad_boot_diff(a, b, TE)
    for v in V: out[v].pop("swrd_corr")
    json.dump(dict(out=out, cmp=cmp, swrd_cmp=sw), open("r8_trained_summary.json", "w"), indent=1)
    for v in V:
        o = out[v]; print(f"{v:8s} acc {o['acc']*100:5.1f} mF1 {o['mf1']*100:5.1f} ECE {o['ece']*100:4.1f} mCA {o['mca']*100:5.1f} val {o['val_acc']*100:5.1f} | SWRD "
                          + " ".join(f"{nm} {d['test']['acc']*100:5.1f}/{d['test']['ece']*100:4.1f}" for nm, d in o["swrd"].items()))
    print(json.dumps(cmp, indent=0)); print(json.dumps(sw, indent=0))
