import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import build

p = argparse.ArgumentParser()
p.add_argument("--model", default="prism")
p.add_argument("--protocol", default="group", choices=["group", "official"])
p.add_argument("--fold", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--loss", default="auto", choices=["auto", "ce", "edl"])
p.add_argument("--lam", type=float, default=0.2, help="max KL weight for EDL")
p.add_argument("--res", type=int, default=112)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--out", default="results")
p.add_argument("--max_steps", type=int, default=0, help="debug: stop after n steps")
p.add_argument("--iters", type=int, default=1000, help="total optimisation steps (0: use epochs)")
p.add_argument("--save_model", action="store_true")
p.add_argument("--no_corrupt", action="store_true")
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); np.random.seed(args.seed)
base_name = args.model.split("+")[0]
use_paug = "+paug" in args.model
loss_type = args.loss if args.loss != "auto" else ("edl" if base_name.startswith("prism") and "_ce" not in base_name else "ce")
model_name = base_name
if "_l05" in base_name:
    args.lam = 0.05
K = 4

D = np.load(f"./riawelc_{args.res}.npz")
X8 = torch.from_numpy(D["X"]).unsqueeze(1)          # uint8; converted per batch to save memory


def getx(i):
    return X8[i].float().div_(255.0)
y = torch.from_numpy(D["y"])
if args.protocol == "official":
    tr = np.where(D["split"] == 0)[0]; va = np.where(D["split"] == 1)[0]; te = np.where(D["split"] == 2)[0]
else:
    S = json.load(open("./splits.json"))[str(args.fold)]
    tr, va, te = np.array(S["train"]), np.array(S["val"]), np.array(S["test"])

net = build(model_name, args.res)
if hasattr(net, "norm") and hasattr(net.norm, "tau"):          # BLAN scale from training films only
    a = -torch.log(getx(tr).clamp_min(1.0 / 255.0)).flatten(1)
    med = a.median(1).values
    net.norm.tau.fill_(float((1.4826 * (a - med[:, None]).abs().median(1).values).median()))
    del a
nparam = sum(p.numel() for p in net.parameters())
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = math.ceil(len(tr) / args.bs)
total = args.iters if args.iters else steps_per_epoch * args.epochs
args.epochs = math.ceil(total / steps_per_epoch)
warm = max(50, total // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))


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


def to_prob(z):
    if loss_type == "edl":
        a = F.softplus(z) + 1.0
        return a / a.sum(1, keepdim=True), K / a.sum(1)
    pr = F.softmax(z, 1)
    return pr, 1 - pr.max(1).values


log = []
t0 = time.time(); step = 0
CK = os.path.join(args.out, f"{args.model}_{args.protocol}_f{args.fold}_s{args.seed}.ckpt")
RES = os.path.exists(CK)   # training finished earlier, evaluation was interrupted: reuse the trained weights
if RES:
    ck = torch.load(CK, map_location="cpu"); net.load_state_dict(ck["sd"]); log = ck["log"]
for ep in range(0 if RES else args.epochs):
    net.train()
    perm = torch.from_numpy(tr[np.random.permutation(len(tr))])
    tl, tc, tn = 0.0, 0, 0
    for k in range(steps_per_epoch):
        idx = perm[k * args.bs:(k + 1) * args.bs]
        xb, yb = augment(getx(idx)), y[idx]
        if use_paug:
            xb = paug(xb)
        z = net(xb)
        if step >= total:
            break
        if loss_type == "edl":
            lam = args.lam * min(1.0, step / (0.5 * total))
            loss = edl_loss(z, yb, lam)
        else:
            loss = F.cross_entropy(z, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        tl += loss.item() * len(idx); tc += (z.argmax(1) == yb).sum().item(); tn += len(idx)
        step += 1
        if args.max_steps and step >= args.max_steps:
            break
    log.append(dict(epoch=ep, loss=tl / tn, acc=tc / tn, time=time.time() - t0))
    print(f"ep {ep} loss {tl/tn:.4f} acc {tc/tn:.4f} t {time.time()-t0:.0f}s", flush=True)
    if (args.max_steps and step >= args.max_steps) or step >= total:
        break
train_time = ck["train_time"] if RES else time.time() - t0
if not RES:
    os.makedirs(args.out, exist_ok=True); torch.save(dict(sd=net.state_dict(), log=log, train_time=train_time), CK)


@torch.no_grad()
def predict(Xs, bs=256):
    net.eval(); Z = []
    for k in range(0, len(Xs), bs):
        Z.append(net(Xs[k:k + bs]))
    return torch.cat(Z)


def corrupt(x, kind, s, g):
    if kind == "clean":
        return x
    if kind == "gamma":
        return x.clamp_min(1e-4) ** s
    if kind == "gain":
        return (x * s).clamp(0, 1)
    if kind == "noise":
        return (x + s * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if kind == "blur":
        r = int(math.ceil(3 * s)); t = torch.arange(-r, r + 1).float()
        k1 = torch.exp(-t ** 2 / (2 * s * s)); k1 /= k1.sum()
        xx = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1))
        return F.conv2d(F.pad(xx, (0, 0, r, r), mode="reflect"), k1.view(1, 1, -1, 1))
    if kind == "contrast":
        m = x.mean((2, 3), keepdim=True)
        return (m + s * (x - m)).clamp(0, 1)
    raise ValueError(kind)


CORR = [("clean", 0)] + [("gamma", s) for s in (0.5, 0.7, 1.5, 2.0)] + [("gain", s) for s in (0.4, 0.7)] + \
       [("noise", s) for s in (0.02, 0.05, 0.08)] + [("blur", s) for s in (0.8, 1.5, 2.5)] + \
       [("contrast", s) for s in (0.7, 0.5, 0.3)]

res = dict(args=vars(args), loss_type=loss_type, params=nparam, train_time=train_time, log=log)
Zva = predict(getx(va)); Zte = {}
g = torch.Generator().manual_seed(1234)
Xte = getx(te)
for kind, s in (CORR[:1] if args.no_corrupt else CORR):
    Zte[f"{kind}_{s}"] = predict(corrupt(Xte, kind, s, g))
os.makedirs(args.out, exist_ok=True)
tag = f"{args.model}_{args.protocol}_f{args.fold}_s{args.seed}"
np.savez_compressed(os.path.join(args.out, tag + ".npz"), va_idx=va, te_idx=te,
                    Zva=Zva.numpy(), **{"Zte_" + k: v.numpy() for k, v in Zte.items()})
pr, _ = to_prob(Zte["clean_0"])
res["test_acc"] = (pr.argmax(1) == y[te]).float().mean().item()
json.dump(res, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
if True:
    torch.save(net.state_dict(), os.path.join(args.out, tag + ".pt"))
print("TEST ACC", res["test_acc"], "params", nparam, "time", round(train_time))
