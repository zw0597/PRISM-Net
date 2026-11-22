"""Few-shot fine-tuning of frozen RIAWELC models on SWRD (protocol: swrd_adapt_protocol.md)."""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from models import build
p = argparse.ArgumentParser(); p.add_argument("--model"); p.add_argument("--fold", type=int); p.add_argument("--budget", type=float); p.add_argument("--recipe", default="C0"); p.add_argument("--pga", default="")
args = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
W = "./"; K = 4
S = np.load(W + "swrd/swrd_tiles80.npz"); X8 = torch.from_numpy(S["X"]).unsqueeze(1)
if args.pga: X8 = torch.from_numpy(np.load(W + "swrd/swrd_pga_inputs.npz")["X" + args.pga]).unsqueeze(1); Y = torch.from_numpy(S["y"]).long()
sp = json.load(open(W + "swrd/swrd_split.json"))
adapt_imgs = set(sp["adapt_5" if args.budget == 0.05 else "adapt_20"])
tr = np.where(np.isin(S["img"], list(adapt_imgs)))[0]; te = np.where(np.isin(S["img"], sp["test"]))[0]
base = args.model.split("+")[0]; use_paug = "+paug" in args.model
edl_mode = base.startswith("prism") and "_ce" not in base
wp = f"{W}results/{args.model}_group_f{args.fold}_s0.pt"; wp = wp if os.path.exists(wp) else f"{W}rerun/{args.model}_group_f{args.fold}_s0.pt"
net = build(base, 80); net.load_state_dict(torch.load(wp, map_location="cpu"))
def augment(xb):
    B, _, H, W = xb.shape
    pad = 8
    xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (B,)); j = torch.randint(0, 2 * pad + 1, (B,))
    out = torch.stack([xp[b, :, i[b]:i[b] + H, j[b]:j[b] + W] for b in range(B)])
    fh = torch.rand(B) < 0.5; fv = torch.rand(B) < 0.5
    out[fh] = out[fh].flip(-1); out[fv] = out[fv].flip(-2)
    return out


def paug(xb, p=0.8):
    """Physics-consistent acquisition augmentation (Beer-Lambert / detector model).
    exposure gain c, detector response exponent g, Poisson-Gaussian quantum + read noise.
    Blur and contrast compression are deliberately NOT used (held-out test corruptions)."""
    B = xb.shape[0]
    on = (torch.rand(B, 1, 1, 1) < p).float()
    c = torch.empty(B, 1, 1, 1).uniform_(0.6, 1.15)
    g = torch.exp(torch.empty(B, 1, 1, 1).uniform_(math.log(0.67), math.log(1.5)))
    x = (c * xb.clamp_min(1e-4) ** g).clamp(0, 1)
    ka = torch.empty(B, 1, 1, 1).uniform_(0.0, 4e-3); kb = torch.empty(B, 1, 1, 1).uniform_(0.0, 1e-3)
    x = (x + torch.sqrt(ka * x + kb) * torch.randn_like(x)).clamp(0, 1)
    return on * x + (1 - on) * xb


def edl_loss(z, t, lam):
    alpha = F.softplus(z) + 1.0
    S = alpha.sum(1, keepdim=True)
    yoh = F.one_hot(t, K).float()
    risk = (yoh * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
    at = yoh + (1 - yoh) * alpha                       # remove target evidence
    St = at.sum(1)
    kl = (torch.lgamma(St) - math.lgamma(K) - torch.lgamma(at).sum(1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St)[:, None])).sum(1))
    return (risk + lam * kl).mean()



R = args.recipe
U = np.where(np.isin(S["img"], sp["U"]))[0]

@torch.no_grad()
def adabn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.reset_running_stats(); m.momentum = None
    model.train()
    for k in range(0, len(U), 256): model(X8[U[k:k + 256]].float() / 255.0)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.momentum = 0.1

if R != "C0": adabn(net)
if R == "C2": use_paug = False

def train(model, idx_pool, ylab, T, lr):
    warm = max(1, T // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / T))))
    model.train(); rng = np.random.default_rng(0)
    for step in range(T):
        j = rng.choice(len(idx_pool), 64, replace=len(idx_pool) < 64); idx = torch.from_numpy(idx_pool[j])
        xb = augment(X8[idx].float() / 255.0)
        if use_paug: xb = paug(xb)
        z = model(xb); yb = torch.from_numpy(ylab[j]).long()
        loss = edl_loss(z, yb, 0.2) if edl_mode else F.cross_entropy(z, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    model.eval()

def predict(model):
    with torch.no_grad():
        Z = torch.cat([model(X8[k:k + 512].float() / 255.0) for k in range(0, len(X8), 512)]).double()
    return ((F.softplus(Z) + 1) / (F.softplus(Z) + 1).sum(1, keepdim=True) if edl_mode else F.softmax(Z, 1)).numpy().astype(np.float32)

if R == "C0": train(net, tr, S["y"][tr], 200, 5e-4)
else: train(net, tr, S["y"][tr], 400, 1e-3)
if R == "C3":
    P1 = predict(net); keep = U[P1[U].max(1) >= 0.9]
    idx_all = np.concatenate([tr, keep]); lab = np.concatenate([S["y"][tr], P1[keep].argmax(1)])
    print("pseudo-labeled", len(keep), "of", len(U), "pseudo acc", float((P1[keep].argmax(1) == S["y"][keep]).mean()))
    train(net, idx_all, lab, 200, 5e-4)
if False:
    T = 200; warm = 20
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / T))))
    net.train(); rng = np.random.default_rng(0)
    for step in range(T):
        idx = torch.from_numpy(rng.choice(tr, 64, replace=len(tr) < 64))
        xb = augment(X8[idx].float() / 255.0)
        if use_paug: xb = paug(xb)
        z = net(xb)
        loss = edl_loss(z, Y[idx], 0.2) if edl_mode else F.cross_entropy(z, Y[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
P = predict(net)
os.makedirs(W + "swrd/ft", exist_ok=True)
tag = f"{args.model}_f{args.fold}_b{int(args.budget*100)}" + ("" if R == "C0" else f"_{R}") + (f"_pga{args.pga}" if args.pga else "")
np.save(W + f"swrd/ft/{tag}.npy", P)
V = np.isin(S["img"], sp["V"]); print(tag, "V acc", float((P[V].argmax(1) == S["y"][V]).mean()), "n_adapt_tiles", len(tr))
