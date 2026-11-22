"""Figures that do not depend on training: data audit, BLAN invariance, polar basis."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, ".")
from models import BLAN, polar_basis

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif"})
OUT = "../latex/figs/"
import os; os.makedirs(OUT, exist_ok=True)
L = json.load(open("./leak_stats.json"))
S = json.load(open("./data_stats.json"))
D = np.load("./riawelc_112.npz")
CLS = ["ND", "PO", "LP", "CR"]

# (data-audit figure: see fig_audit.py)

# ------------------------------------------------------------- Fig: BLAN invariance
idx = [int(np.where(D["y"] == c)[0][37 * (c + 3)]) for c in (3, 1, 2)]   # CR, PO, LP examples
trans = [("original", lambda I: I), (r"$0.5\,I$", lambda I: 0.5 * I), (r"$I^{0.6}$", lambda I: I ** 0.6),
         (r"$0.7\,I^{1.8}$", lambda I: 0.7 * I ** 1.8)]
blan = BLAN()
fig, axs = plt.subplots(3, 8, figsize=(7.16, 2.85), gridspec_kw=dict(wspace=0.05, hspace=0.1))
for r, i in enumerate(idx):
    I0 = D["X"][i].astype(np.float64) / 255.0
    I0 = np.clip(I0, 1 / 255, 1)
    z0 = blan(torch.from_numpy(I0).float()[None, None])[0, 0].numpy()
    for c, (nm, f) in enumerate(trans):
        I = f(I0)
        axs[r, c].imshow(I, cmap="gray", vmin=0, vmax=1)
        z = blan(torch.from_numpy(I).float()[None, None])[0, 0].numpy()
        axs[r, 4 + c].imshow(z, cmap="gray", vmin=-4, vmax=4)
        if r == 0:
            axs[r, c].set_title(nm, fontsize=7.5)
            axs[r, 4 + c].set_title(r"$\mathcal{B}($" + nm.replace("$", "") .join(["$", "$"]) + r"$)$" if False else "BLAN(" + nm + ")", fontsize=7)
        dev = np.abs(z - z0).max()
        axs[r, 4 + c].text(0.03, 0.04, f"max|Δz|={dev:.1e}", transform=axs[r, 4 + c].transAxes, fontsize=5.4,
                           color="black", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    axs[r, 0].set_ylabel(["CR", "PO", "LP"][r], fontsize=8)
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.text(0.31, 0.03, "Raw intensity under exposure / detector-response changes", ha="center", fontsize=7.5)
fig.text(0.715, 0.03, "Beer–Lambert attenuation normalisation output", ha="center", fontsize=7.5)
fig.savefig(OUT + "fig_blan.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_blan.png", bbox_inches="tight", dpi=200)
plt.close(fig)

# ------------------------------------------------------------- Fig: polar basis
H = W = 20
B = polar_basis(H, W, 6, 8, 2.0).numpy().reshape(6, 8, H, W // 2 + 1)


def full(h):  # mirror rFFT half-plane to the centred full plane for display
    Fh = np.zeros((H, W))
    for v in range(H):
        for u in range(W):
            uu, vv = (u, v) if u <= W // 2 else (W - u, (H - v) % H)
            Fh[v, u] = h[vv, uu]
    return np.fft.fftshift(Fh)[1:, 1:]


fig, axs = plt.subplots(2, 6, figsize=(7.16, 2.55), gridspec_kw=dict(wspace=0.08, hspace=0.32))
for r in range(6):
    axs[0, r].imshow(full(B[r].sum(0)), cmap="Blues", origin="lower")
    axs[0, r].set_title(rf"$\phi_{{{r+1}}}(\rho)$", fontsize=7.5)
for j, a in enumerate([0, 2, 4, 6]):
    axs[1, j].imshow(full(B[:, a].sum(0)), cmap="Oranges", origin="lower")
    axs[1, j].set_title(rf"$\psi_{{{a+1}}}(\theta)$, $\theta_{{{a+1}}}={a*180//8}^\circ$", fontsize=7)
rng = np.random.default_rng(3)
w_iso = np.repeat(np.array([0.2, 1.0, 0.6, -0.4, -0.8, -1.0])[:, None], 8, 1)
w_an = w_iso.copy(); w_an[:, [3, 4, 5]] += 1.2; w_an[:, [0, 1, 7]] -= 0.8
for j, (w, nm) in enumerate([(w_iso, "isotropic gate"), (w_an, "anisotropic gate")]):
    g = 1 + np.tanh((w[:, :, None, None] * B).sum((0, 1)))
    axs[1, 4 + j].imshow(full(g), cmap="RdBu_r", vmin=0, vmax=2, origin="lower")
    axs[1, 4 + j].set_title(nm, fontsize=7)
for a in axs.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.savefig(OUT + "fig_basis.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_basis.png", bbox_inches="tight", dpi=200)
print("done")
