"""RASG interpretability: anisotropy index eta per class/stage and class-mean gates.
Uses the trained PRISM-Net (+PAug) weights of every fold, evaluated on that fold's test films.
Writes interp.json (numbers) and paper/figs/fig_interp.{pdf,png}."""
import json, math, sys
import numpy as np
import torch
from scipy.stats import mannwhitneyu
sys.path.insert(0, ".")
from models import build, RASG

torch.set_num_threads(1)
M = sys.argv[1] if len(sys.argv) > 1 else "prism+paug"
folds = [int(f) for f in (sys.argv[2].split(",") if len(sys.argv) > 2 else range(5))]
D = np.load("./riawelc_80.npz")
X = torch.from_numpy(D["X"]).float().div_(255.0).unsqueeze(1)
Y = D["y"]
S = json.load(open("./splits.json"))
CLS = ["ND", "PO", "LP", "CR"]

eta = {}      # layer name -> list of per-sample eta (all folds)
HS = {}       # layer name -> spatial size
lab = []
wmean = {}    # (layer, class) -> sum of w, count
for f in folds:
    net = build(M.split("+")[0], 80)
    net.load_state_dict(torch.load(f"./results/{M}_group_f{f}_s0.pt", map_location="cpu"))
    net.eval()
    layers = [(n, m) for n, m in net.named_modules() if isinstance(m, RASG)]
    for n, m in layers:
        HS[n] = m.basis.shape[1]
    te = np.array(S[str(f)]["test"])
    store = {n: [] for n, _ in layers}
    with torch.no_grad():
        for k in range(0, len(te), 256):
            net(X[te[k:k + 256]])
            for n, m in layers:
                store[n].append(m.last_w.reshape(m.last_w.shape[0], m.G, m.R, m.A).clone())
    lab.append(Y[te])
    for n, m in layers:
        w = torch.cat(store[n])                                   # (N,G,R,A)
        wt = w - w.mean(-1, keepdim=True)
        e = (wt.flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1).clamp_min(1e-12)).numpy()
        eta.setdefault(n, []).append(e)
        for c in range(4):
            s = w[torch.from_numpy(Y[te] == c)].sum(0)
            a, b = wmean.get((n, c), (0, 0))
            wmean[(n, c)] = (a + s, b + int((Y[te] == c).sum()))
lab = np.concatenate(lab)
names = list(eta)
out = {"model": M, "folds": folds, "layers": {}}
for n in names:
    e = np.concatenate(eta[n])
    r = {CLS[c]: [float(np.median(e[lab == c])), float(np.mean(e[lab == c])), float(np.std(e[lab == c]))] for c in range(4)}
    oriented = e[(lab == 2) | (lab == 3)]
    r["p_PO_vs_LPCR"] = float(mannwhitneyu(e[lab == 1], oriented, alternative="less").pvalue)
    r["auc_PO_lt_oriented"] = float(mannwhitneyu(oriented, e[lab == 1]).statistic / (len(oriented) * (lab == 1).sum()))
    out["layers"][n] = r
    print(n, {k: (round(v[0], 3) if isinstance(v, list) else v) for k, v in r.items()})
json.dump(out, open("./interp.json", "w"), indent=1)

if "--fig" not in sys.argv:
    sys.exit()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "mathtext.fontset": "dejavuserif"})
CC = ["#7f7f7f", "#2a78d6", "#1baf7a", "#e34948"]


def full_basis(n, R=6, A=8, kappa=2.0):
    fy = torch.fft.fftshift(torch.fft.fftfreq(n)); fx = fy
    v, u = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(u ** 2 + v ** 2) / (0.5 * math.sqrt(2.0)); th = torch.atan2(v, u)
    mu = torch.linspace(0, 1, R); sig = 1.0 / (R - 1)
    phi = torch.exp(-(rho[None] - mu[:, None, None]) ** 2 / (2 * sig ** 2))
    tha = torch.arange(A) * math.pi / A
    psi = torch.exp(kappa * torch.cos(2 * (th[None] - tha[:, None, None]))); psi = psi / psi.sum(0, keepdim=True)
    psi[:, rho == 0] = 1.0 / A
    return (phi[:, None] * psi[None]).reshape(R * A, n, n)


Bf = full_basis(81)
# one representative layer per stage: the last RASG of stages 2-4
stage_of = {}
for n in names:                       # last RASG layer of each resolution (= stage)
    stage_of[HS[n]] = n
reps = [stage_of[h] for h in sorted(stage_of, reverse=True)]
fig = plt.figure(figsize=(7.16, 2.55))
gs = fig.add_gridspec(2, 6, width_ratios=[1.55, 0.06, 1, 1, 1, 1], wspace=0.08, hspace=0.32)
ax = fig.add_subplot(gs[:, 0])
pos = 0; ticks = []
for si, n in enumerate(reps):
    e = np.concatenate(eta[n])
    for c in range(4):
        bp = ax.boxplot(e[lab == c], positions=[pos], widths=0.7, showfliers=False, whis=(5, 95), patch_artist=True,
                        medianprops=dict(color="black", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                        boxprops=dict(lw=0.6))
        bp["boxes"][0].set_facecolor(CC[c]); bp["boxes"][0].set_alpha(0.8)
        pos += 1
    ticks.append(pos - 2.5); pos += 1
ax.set_xticks(ticks); ax.set_xticklabels([f"stage {si + 2}" for si in range(len(reps))])
ax.set_ylabel(r"anisotropy index $\eta$")
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=CC[c], alpha=0.8) for c in range(4)], CLS, ncol=4, frameon=False,
          loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5, handlelength=1.0, columnspacing=0.8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_title("(a)", fontsize=7.5, loc="left", pad=14)
# (b) class-mean gate deviation of the two deeper representative layers
lays = reps[-2:]
vmax = 0
maps = {}
for n in lays:
    for c in range(4):
        s, cnt = wmean[(n, c)]
        w = (s / cnt).mean(0).reshape(-1)                               # mean over groups -> (R*A,)
        g = torch.tanh(torch.einsum("k,khw->hw", w, Bf)).numpy()
        maps[(n, c)] = g; vmax = max(vmax, np.abs(g).max())
for li, n in enumerate(lays):
    for c in range(4):
        a2 = fig.add_subplot(gs[li, 2 + c])
        im = a2.imshow(maps[(n, c)], cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-0.5, 0.5, -0.5, 0.5])
        a2.set_xticks([]); a2.set_yticks([])
        Hn = HS[n]
        fq = np.fft.fftshift(np.fft.fftfreq(Hn)); uu, vv = np.meshgrid(fq, fq)
        a2.scatter(uu.ravel(), vv.ravel(), s=3, c="k", marker="o", linewidths=0)
        a2.set_xlim(-0.5, 0.5); a2.set_ylim(-0.5, 0.5)
        if li == 0:
            a2.set_title(CLS[c], fontsize=7.5)
        if c == 0:
            a2.set_ylabel(f"stage {len(reps) - len(lays) + li + 2}", fontsize=7)
fig.text(0.375, 0.93, "(b)", fontsize=7.5)
cax = fig.add_axes([0.915, 0.15, 0.008, 0.68])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6); cb.set_label(r"$G-1$", fontsize=7)
fig.savefig("../latex/figs/fig_interp.pdf", bbox_inches="tight")
fig.savefig("../latex/figs/fig_interp.png", bbox_inches="tight", dpi=200)
print("fig written")
