"""Few-shot fine-tuning of frozen RIAWELC models on SWRD (protocol: swrd_adapt_protocol.md)."""
import argparse, json, math, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); from models import build
p = argparse.ArgumentParser(); p.add_argument("--model"); p.add_argument("--fold", type=int); p.add_argument("--budget", type=float)
args = p.parse_args(); torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
W = "./"; K = 4
S = np.load(W + "swrd/swrd_tiles80.npz"); X8 = torch.from_numpy(S["X"]).unsqueeze(1); Y = torch.from_numpy(S["y"]).long()
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
net.eval()
with torch.no_grad():
    Z = torch.cat([net(X8[k:k + 512].float() / 255.0) for k in range(0, len(X8), 512)]).double()
P = ((F.softplus(Z) + 1) / (F.softplus(Z) + 1).sum(1, keepdim=True) if edl_mode else F.softmax(Z, 1)).numpy().astype(np.float32)
os.makedirs(W + "swrd/ft", exist_ok=True)
tag = f"{args.model}_f{args.fold}_b{int(args.budget*100)}"
np.save(W + f"swrd/ft/{tag}.npy", P)
print(tag, "test acc", float((P[te].argmax(1) == S["y"][te]).mean()), "n_adapt_tiles", len(tr))
