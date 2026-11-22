"""R1 (reviewer_protocol.md): generic per-radiograph input standardisation (ZS-R, HM-R) as baselines for PGA-R.
Usage: python3 swrd_std.py [model ...]   (default: all models of the paper); results are merged into swrd_probs_std.npz."""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from models import build
torch.set_num_threads(1); W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); X8 = S["X"]; IMG = S["img"]
sp = json.load(open(W + "swrd/swrd_split.json")); U = np.isin(IMG, sp["U"])
R = np.load(W + "riawelc_80.npz")["X"][::3]                       # same source subset as PGA
ms, ss = float(R.mean() / 255), float(R.std() / 255)
cs = np.cumsum(np.bincount(R.ravel(), minlength=256)) / R.size   # source CDF on the 256 grey levels
INP = W + "swrd/swrd_std_inputs.npz"
if os.path.exists(INP):
    Z = np.load(INP); XZ, XH = Z["XZ"], Z["XH"]
else:
    XZ = np.empty_like(X8); XH = np.empty_like(X8)
    for im in np.unique(IMG):
        k = IMG == im; x = X8[k]; f = x.astype(np.float32) / 255
        XZ[k] = (np.clip((f - f.mean()) / max(f.std(), 1e-3) * ss + ms, 0, 1) * 255).round().astype(np.uint8)
        ct = np.cumsum(np.bincount(x.ravel(), minlength=256)) / x.size
        lut = np.searchsorted(cs, ct, side="left").clip(0, 255).astype(np.uint8)   # F_s^{-1}(F_t(v))
        XH[k] = lut[x]
    np.savez_compressed(INP, XZ=XZ, XH=XH)
VAR = {"ZS": torch.from_numpy(XZ.astype(np.float32) / 255).unsqueeze(1), "HM": torch.from_numpy(XH.astype(np.float32) / 255).unsqueeze(1)}
cfg = json.load(open(W + "paper_cfg.json")); MODELS = sys.argv[1:] or cfg["main"] + cfg["paug_baselines"]
OUT = W + "swrd/swrd_probs_std.npz"; out = dict(np.load(OUT)) if os.path.exists(OUT) else {}
for m in MODELS:
    if f"{m}|HM_bn" in out: continue
    base = m.split("+")[0]; edl = base.startswith("prism") and "_ce" not in base
    res = {k: [] for k in ("ZS", "HM", "ZS_bn", "HM_bn")}
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
    np.savez_compressed(OUT, **out); print(m, flush=True)
