"""Physics-guided acquisition alignment (PGA) on SWRD: zero-label evaluation of all models (swrd_pga_protocol.md)."""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from models import build
torch.set_num_threads(2); W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); X8 = S["X"]; IMG = S["img"]
sp = json.load(open(W + "swrd/swrd_split.json")); U = np.isin(IMG, sp["U"])
R = np.load(W + "riawelc_80.npz")["X"]


def stats(x8):
    a = np.log(np.maximum(x8.astype(np.float32), 1.0) / 255.0); m = np.median(a)
    return float(m), float(1.4826 * np.median(np.abs(a - m)))


ms, ss = stats(R[::3])


def align(x8, mt, st):
    g = ss / max(st, 1e-3); lc = ms - g * mt
    a = np.log(np.maximum(x8.astype(np.float32), 1.0) / 255.0)
    return np.clip(np.exp(lc + g * a), 0, 1).astype(np.float32), g, float(np.exp(lc))


mU, sU = stats(X8[U]); XS, gS, cS = align(X8, mU, sU)
XR = np.empty_like(XS); par = {}
for im in np.unique(IMG):
    k = IMG == im; mt, st = stats(X8[k]); XR[k], g, c = align(X8[k], mt, st); par[im] = (g, c)
gs = np.array([v[0] for v in par.values()]); cs = np.array([v[1] for v in par.values()])
info = dict(source=[ms, ss], target=[mU, sU], g_S=gs_ if (gs_ := float(gS)) else 0, c_S=cS,
            g_R=[float(np.percentile(gs, q)) for q in (10, 50, 90)], c_R=[float(np.percentile(cs, q)) for q in (10, 50, 90)])
print(info, flush=True)
np.savez_compressed(W + "swrd/swrd_pga_inputs.npz", XS=(XS * 255).round().astype(np.uint8), XR=(XR * 255).round().astype(np.uint8))
json.dump(info, open(W + "swrd/pga_params.json", "w"))
VAR = {"S": torch.from_numpy(XS).unsqueeze(1), "R": torch.from_numpy(XR).unsqueeze(1)}
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
out = dict(np.load(W + "swrd/swrd_probs_pga.npz")) if os.path.exists(W + "swrd/swrd_probs_pga.npz") else {}
for m in MODELS:
    if f"{m}|R_bn" in out: continue
    base = m.split("+")[0]; edl = base.startswith("prism") and "_ce" not in base
    res = {k: [] for k in ("S", "R", "S_bn", "R_bn")}
    for f in range(5):
        p = f"{W}results/{m}_group_f{f}_s0.pt"; p = p if os.path.exists(p) else f"{W}rerun/{m}_group_f{f}_s0.pt"
        sd = torch.load(p, map_location="cpu")
        for v, X in VAR.items():
            for bn in (False, True):
                net = build(base, 80); net.load_state_dict(sd)
                with torch.no_grad():
                    if bn:
                        bns = [x for x in net.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
                        for b in bns: b.reset_running_stats(); b.momentum = None
                        net.train(); iu = np.where(U)[0]
                        for k in range(0, len(iu), 256): net(X[iu[k:k + 256]])
                    net.eval(); Z = torch.cat([net(X[k:k + 512]) for k in range(0, len(X), 512)]).double()
                P = ((F.softplus(Z) + 1) / (F.softplus(Z) + 1).sum(1, keepdim=True) if edl else F.softmax(Z, 1)).numpy()
                res[v + ("_bn" if bn else "")].append(P.astype(np.float32))
    for k, v in res.items(): out[f"{m}|{k}"] = np.stack(v)
    print(m, flush=True)
np.savez_compressed(W + "swrd/swrd_probs_pga.npz", **out)
