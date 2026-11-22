"""Fig. audit: film x class tile counts grouped by test fold, and class proportions per fold
(the leakage fractions are given in Table I)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif"})
OUT = os.environ.get("FIG_OUT", "../latex/figs/")
L = json.load(open("./leak_stats.json"))
CLS = ["ND", "PO", "LP", "CR"]

# ------------------------------------------------------------- Fig: data audit
fig = plt.figure(figsize=(7.16, 2.55))
gs = fig.add_gridspec(1, 2, width_ratios=[2.7, 0.95], wspace=0.48)
ax = fig.add_subplot(gs[0])
films = sorted(L["film_table"], key=lambda f: (L["fold_of_film"][f], f))
M = np.array([L["film_table"][f] for f in films]).T.astype(float)
im = ax.imshow(np.log10(M + 1), aspect="auto", cmap="Blues", vmin=0, vmax=3.3)
ax.set_yticks(range(4)); ax.set_yticklabels(CLS)
ax.set_xticks(range(len(films))); ax.set_xticklabels([f.replace("RRT-", "") for f in films], rotation=90, fontsize=5.6)
fo = [L["fold_of_film"][f] for f in films]
for i in range(1, len(films)):
    if fo[i] != fo[i - 1]:
        ax.axvline(i - 0.5, color="#e34948", lw=1.0)
starts = [0] + [i for i in range(1, len(films)) if fo[i] != fo[i - 1]] + [len(films)]
for k in range(len(starts) - 1):
    ax.text((starts[k] + starts[k + 1] - 1) / 2, -0.85, f"Fold {fo[starts[k]]+1}", ha="center", va="bottom", fontsize=6.5)
ax.set_ylim(3.5, -0.5)
ax.set_xlabel("Radiographic film (grouped by test fold)")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
cb.set_label(r"$\log_{10}(1+n)$", fontsize=7); cb.ax.tick_params(labelsize=6)
ax.text(-0.05, 1.17, "(a)", transform=ax.transAxes, fontsize=8)

ax = fig.add_subplot(gs[1])
fc = np.array(L["fold_class_counts"], float)
frac = fc / fc.sum(1, keepdims=True)
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
left = np.zeros(5)
for c in range(4):
    ax.barh(np.arange(5), frac[:, c], left=left, color=cols[c], edgecolor="white", linewidth=1.0, label=CLS[c])
    left += frac[:, c]
ax.set_yticks(range(5)); ax.set_yticklabels([f"F{k+1}\n(n={int(fc[k].sum())})" for k in range(5)], fontsize=6)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Class proportion")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.24), ncol=4, frameon=False, handlelength=0.8, columnspacing=0.6, fontsize=6.3)
ax.spines[["top", "right"]].set_visible(False)
ax.text(-0.5, 1.17, "(b)", transform=ax.transAxes, fontsize=8)
fig.savefig(OUT + "fig_audit.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_audit.png", bbox_inches="tight", dpi=200)
plt.close(fig)

