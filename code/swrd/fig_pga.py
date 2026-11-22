"""Figure for physics-guided acquisition alignment (PGA): parameters, intensity statistics, examples (per-model gains are in Table IX)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.5, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False})
import os
W = "./"; OUT = os.environ.get("FIG_OUT", "../latex/figs/")
S = np.load(W + "swrd/swrd_tiles80.npz"); X8 = S["X"]; IMG = S["img"]; Y = S["y"]
XR = np.load(W + "swrd/swrd_pga_inputs.npz")["XR"]; R = np.load(W + "riawelc_80.npz")["X"][::3]
sp = json.load(open(W + "swrd/swrd_split.json")); te = np.isin(IMG, sp["test"])
par = json.load(open(W + "swrd/pga_params.json")); ms, ss = par["source"]
d = json.load(open(W + "results/swrd_pga.json"))["models"]; cfg = json.load(open(W + "paper_cfg.json"))
MODELS = cfg["main"] + cfg["paug_baselines"]
N = dict(cfg["names"]); N.update({"prismr+paug": "PRISM-Net"})
lg = lambda x: np.log(np.maximum(x.astype(np.float32), 1.0) / 255.0)
g = []
for im in np.unique(IMG):
    a = lg(X8[IMG == im]); m = np.median(a); g.append(ss / max(1.4826 * np.median(np.abs(a - m)), 1e-3))
g = np.array(g)

fig = plt.figure(figsize=(7.16, 2.3))
gs = fig.add_gridspec(2, 7, width_ratios=[2.4, 0.75, 2.4, 0.45, 1.05, 1.05, 1.05], wspace=0.06, hspace=0.06,
                      left=0.07, right=0.99, top=0.95, bottom=0.24)
lab = lambda ax, t, x=0.5: ax.text(x, -0.27, t, transform=ax.transAxes, ha="center", va="top", fontsize=8)
# (a) per-radiograph response exponent
ax = fig.add_subplot(gs[:, 0])
ax.hist(np.log10(g), bins=40, color="#86b6ef", edgecolor="white", linewidth=0.3)
ax.axvline(np.log10(par["g_S"]), color="#eb6834", lw=1.0, ls="--", label=r"system $\hat g$")
ax.axvline(0, color="0.3", lw=0.7, ls=":", label=r"$\hat g=1$")
ax.set_xticks(np.log10([0.25, 1, 4, 16])); ax.set_xticklabels(["0.25", "1", "4", "16"])
ax.set_xlim(np.log10(0.2), np.log10(20)); ax.set_ylim(0, 175); ax.set_xlabel(r"exponent $\hat g$ per radiograph"); ax.set_ylabel("radiographs")
ax.legend(frameon=False, loc="upper right", handlelength=1.2, borderpad=0, bbox_to_anchor=(1.04, 1.03)); lab(ax, "(a)")
# (b) log-intensity statistics
ax = fig.add_subplot(gs[:, 2]); bins = np.linspace(-4.5, 0, 90)
for x, c, l, ls in ((lg(R), "0.2", "RIAWELC", "-"), (lg(X8[te]), "#eb6834", "SWRD", "--"), (lg(XR[te]), "#2a78d6", "SWRD + PGA-R", "-")):
    h, _ = np.histogram(x.ravel(), bins=bins, density=True); ax.plot(0.5 * (bins[1:] + bins[:-1]), h, color=c, lw=1.0, ls=ls, label=l)
ax.set_xlabel("log-intensity"); ax.set_ylabel("density"); ax.set_ylim(0, 2.1)
ax.legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0, bbox_to_anchor=(-0.02, 1.03)); lab(ax, "(b)")
# (c) examples: raw (top) and aligned (bottom), one tile per defect class from radiographs with extreme exponents
# deterministic choice: the test radiograph closest to the 10th / 50th / 90th percentile of the exponent, first defect tile
ims = np.unique(IMG); pick = []; tims = np.isin(ims, sp["test"])
for q in (10, 50, 90):
    k = np.where(tims)[0][np.argmin(np.abs(np.log(g[tims]) - np.log(np.percentile(g, q))))]
    cand = np.where((IMG == ims[k]) & (Y > 0))[0]; cand = cand if len(cand) else np.where(IMG == ims[k])[0]; pick.append(int(cand[0]))
for j, i in enumerate(pick):
    for r, (x, rl) in enumerate(((X8, "SWRD"), (XR, "PGA-R"))):
        a = fig.add_subplot(gs[r, 4 + j]); a.imshow(x[i], cmap="gray", vmin=0, vmax=255, aspect="auto"); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.4)
        if j == 0: a.set_ylabel(rl, fontsize=7)
        if r == 1:
            a.set_xlabel(["ND", "PO", "LP", "CR"][Y[i]] + rf", $\hat g$={g[np.searchsorted(ims, IMG[i])]:.2f}", fontsize=5.6, labelpad=2)
            if j == 1: a.text(0.5, -0.36, "(c)", transform=a.transAxes, ha="center", va="top", fontsize=8)
fig.savefig(OUT + "fig_pga.pdf"); fig.savefig(OUT + "fig_pga.png", dpi=200)
print("g pct", np.percentile(g, [10, 50, 90]), [(int(i), float(g[np.searchsorted(ims, IMG[i])])) for i in pick])
