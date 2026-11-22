"""Zero-label adaptation: AdaBN on unlabeled SWRD tiles (U radiographs), all architectures, five fold models."""
import json, os, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from models import build
torch.set_num_threads(2); W = "./"
S = np.load(W + "swrd/swrd_tiles80.npz"); X = torch.from_numpy(S["X"]).unsqueeze(1).float() / 255.0
sp = json.load(open(W + "swrd/swrd_split.json")); U = np.where(np.isin(S["img"], sp["U"]))[0]
cfg = json.load(open(W + "paper_cfg.json")); MODELS = cfg["main"] + cfg["paug_baselines"]
out = dict(np.load(W + "swrd/swrd_probs_adabn.npz")) if os.path.exists(W + "swrd/swrd_probs_adabn.npz") else {}
for m in MODELS:
    if m in out: continue
    base = m.split("+")[0]; edl = base.startswith("prism") and "_ce" not in base; P = []
    for f in range(5):
        p = f"{W}results/{m}_group_f{f}_s0.pt"; p = p if os.path.exists(p) else f"{W}rerun/{m}_group_f{f}_s0.pt"
        net = build(base, 80); net.load_state_dict(torch.load(p, map_location="cpu"))
        bns = [x for x in net.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
        with torch.no_grad():
            for b in bns: b.reset_running_stats(); b.momentum = None
            net.train()
            for k in range(0, len(U), 256): net(X[U[k:k + 256]])
            net.eval(); Z = torch.cat([net(X[k:k + 512]) for k in range(0, len(X), 512)]).double()
        P.append(((F.softplus(Z) + 1) / (F.softplus(Z) + 1).sum(1, keepdim=True) if edl else F.softmax(Z, 1)).numpy())
        if not bns: print(m, "has no BatchNorm: AdaBN leaves it unchanged")
    out[m] = np.stack(P).astype(np.float32); print(m, flush=True)
np.savez_compressed(W + "swrd/swrd_probs_adabn.npz", **out)
