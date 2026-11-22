"""Result figures, all read from summary.json (single source of truth)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.linewidth": 0.6, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.8, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = "../latex/figs/"
S = json.load(open("./summary.json"))
A, OFF = S["agg"], S["official"]
CFG = json.load(open("./paper_cfg.json"))
MAIN, NAME, FINAL = CFG.get("fig_models", CFG["main"]), CFG["names"], CFG["final"]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#8a5a2b"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
COL = {m: PAL[i] for i, m in enumerate(MAIN)}
MRK = {m: MK[i] for i, m in enumerate(MAIN)}
CLS = ["ND", "PO", "LP", "CR"]


def save(fig, n):
    fig.savefig(OUT + n + ".pdf", bbox_inches="tight"); fig.savefig(OUT + n + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- leakage inflation
mods = [m for m in CFG["official_models"] if m in OFF and m in A]
fig, ax = plt.subplots(figsize=(3.5, 2.25))
x = np.arange(len(mods)); w = 0.38
o = [OFF[m]["acc"] * 100 for m in mods]; g = [A[m]["acc"][0] * 100 for m in mods]; gs = [A[m]["acc"][1] * 100 for m in mods]
ax.bar(x - w / 2, o, w, color="#86b6ef", label="official split (leaky)")
ax.bar(x + w / 2, g, w, yerr=gs, color="#2a78d6", label="film-grouped 5-fold", error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6))
for i in range(len(mods)):
    ax.text(x[i], max(o[i], g[i] + gs[i]) + 1.2, f"$-${o[i]-g[i]:.1f}", ha="center", va="bottom", fontsize=5.8)
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in mods], rotation=30, ha="right")
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(50, 110); ax.set_yticks([50, 60, 70, 80, 90, 100])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, handlelength=1.0)
save(fig, "fig_inflation")

# ---------------------------------------------------------------- robustness curves
fam = [("gamma", [0.5, 0.7, 1.0, 1.5, 2.0], r"exponent $g$ in $I^{g}$"),
       ("gain", [0.4, 0.7, 1.0], r"gain $c$ in $cI$"),
       ("noise", [0.0, 0.02, 0.05, 0.08], r"noise std. $\sigma_n$"),
       ("blur", [0.0, 0.8, 1.5, 2.5], r"blur std. $\sigma_b$ (px)"),
       ("contrast", [1.0, 0.7, 0.5, 0.3], r"contrast factor $s$")]
fig, axs = plt.subplots(1, 5, figsize=(7.16, 1.95), gridspec_kw=dict(wspace=0.32))
for ax, (fk, lv, xl) in zip(axs, fam):
    for m in MAIN:
        ys = []
        for v in lv:
            clean = (fk in ("gamma", "gain", "contrast") and v == 1.0) or (fk in ("noise", "blur") and v == 0.0)
            ys.append(A[m]["acc"][0] * 100 if clean else A[m]["rob"][f"{fk}_{v}"][0] * 100)
        ax.plot(lv, ys, color=COL[m], marker=MRK[m], ms=3.2, lw=1.1 if m != FINAL else 1.8, label=NAME[m],
                zorder=3 if m == FINAL else 2)
    if fk == "gamma":
        ax.set_xscale("log"); ax.minorticks_off()
    ax.set_xlabel(xl, fontsize=7); ax.set_xticks(lv)
    ax.set_xticklabels([("%g" % v) for v in lv], rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=5.8, pad=1)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    tag = "held-out" if fk in ("blur", "contrast") else "in-family"
    ax.set_title(f"{fk} ({tag})", fontsize=7.2)
    if fk == "contrast":
        ax.invert_xaxis()
axs[0].set_ylabel("Accuracy (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(MAIN), frameon=False, handlelength=1.6, columnspacing=1.0)
save(fig, "fig_robust")

# ---------------------------------------------------------------- reliability + risk-coverage
best = CFG["best_baseline"]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw=dict(wspace=0.38))
for ax, m in zip(axs[:2], [FINAL, best]):
    rel = []
    for f in range(5):
        rel += S["runs"][f"{m}_group_f{f}_s0"]["reliability"]
    rel = np.array(rel)
    edges = np.linspace(0, 1, 11); cs, as_, ns = [], [], []
    for i in range(10):
        mm = (rel[:, 0] > edges[i]) & (rel[:, 0] <= edges[i + 1])
        if mm.sum():
            n = rel[mm, 2]; cs.append((rel[mm, 0] * n).sum() / n.sum()); as_.append((rel[mm, 1] * n).sum() / n.sum()); ns.append(n.sum())
    cs, as_, ns = map(np.array, (cs, as_, ns))
    ax.bar(cs, as_, width=0.085, color=COL[m], alpha=0.85, edgecolor="white", linewidth=0.6, label="accuracy")
    ax.plot([0, 1], [0, 1], ls="--", color="#555555", lw=0.8, label="perfect calibration")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{NAME[m]}: ECE {A[m]['ece'][0]*100:.2f}%", fontsize=7.2)
ax = axs[2]
for m in MAIN:
    rc = np.array(A[m]["risk_curve"]) * 100
    cov = np.linspace(0, 1, len(rc)) * 100
    ax.plot(cov[4:], rc[4:], color=COL[m], lw=1.1 if m != FINAL else 1.8, label=NAME[m], zorder=3 if m == FINAL else 2)
ax.set_xlabel("Coverage (%)"); ax.set_ylabel("Selective risk (%)")
ax.set_title("Risk\u2013coverage", fontsize=7.2)
ax.grid(lw=0.3, color="#dddddd")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=5.6, handlelength=1.2, ncol=1)
save(fig, "fig_reliab")

# ---------------------------------------------------------------- confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.85), gridspec_kw=dict(wspace=0.45))
for ax, m in zip(axs, [FINAL, best]):
    cm = np.array(A[m]["cm"], float); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLS); ax.set_yticks(range(4)); ax.set_yticklabels(CLS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(NAME[m], fontsize=7.2)
    for s in ax.spines.values():
        s.set_visible(False)
save(fig, "fig_cm")

# ---------------------------------------------------------------- per-film accuracy
films = sorted(A[FINAL]["per_film"], key=lambda f: -A[FINAL]["per_film"][f][1])
films = [f for f in films if A[FINAL]["per_film"][f][1] >= 50]
fig, ax = plt.subplots(figsize=(7.16, 1.9))
xs = np.arange(len(films))
off = np.linspace(-0.3, 0.3, len(MAIN))
for k, m in enumerate(MAIN):
    v = [A[m]["per_film"][f][0] * 100 for f in films]
    ax.scatter(xs + off[k], v, s=7 if m != FINAL else 14, color=COL[m], marker=MRK[m], label=NAME[m],
               zorder=3 if m == FINAL else 2, linewidths=0)
ax.set_xticks(xs); ax.set_xticklabels([f.replace("RRT-", "") + f" ({A[FINAL]['per_film'][f][1]})" for f in films], fontsize=5.4, rotation=90)
ax.set_ylabel("Film accuracy (%)"); ax.set_ylim(0, 105)
for xx in xs[:-1]:
    ax.axvline(xx + 0.5, color="#eeeeee", lw=0.5, zorder=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=len(MAIN), frameon=False, handletextpad=0.2, columnspacing=1.0)
save(fig, "fig_films")

# ---------------------------------------------------------------- efficiency
E = json.load(open("./efficiency.json"))
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for m in MAIN:
    ax.errorbar(E[m.split("+")[0]]["params"] / 1e6, A[m]["acc"][0] * 100, yerr=A[m]["acc"][1] * 100, fmt=MRK[m], color=COL[m],
                ms=5 if m != FINAL else 8, capsize=1.5, lw=0.6, label=NAME[m])
ax.set_xscale("log"); ax.set_xlabel("Parameters (M, log scale)"); ax.set_ylabel("Accuracy (%)")
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 12])); ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "4", "8", "12"]))
ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(0.9, 16); ax.set_ylim(58, 98)
ax.grid(lw=0.3, color="#dddddd", which="both")
ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0), frameon=False, fontsize=5.6, ncol=2, handletextpad=0.3, columnspacing=0.8)
save(fig, "fig_eff")
print("figures written")
