"""Fig. results2: (a) accuracy vs. GFLOPs, (b) SWRD adaptation ladder (the three-seed values are in Table V).
All values read from the result files used for the tables."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5,
                     "axes.linewidth": 0.6, "axes.labelsize": 7.5, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
                     "legend.fontsize": 6.2, "savefig.dpi": 300, "mathtext.fontset": "dejavuserif",
                     "axes.spines.top": False, "axes.spines.right": False})
import os
W = "./"; OUT = os.environ.get("FIG_OUT", "../latex/figs/")
S = json.load(open(W + "summary.json"))["agg"]
E = json.load(open(W + "efficiency.json"))
CFG = json.load(open(W + "paper_cfg.json")); NAME = CFG["names"]
BLUE, VGGC, RNC, RNPC, GREY = "#2a78d6", "#8a5a2b", "#1baf7a", "#e34948", "#9a9a9a"

fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.15), gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.22))

# (a) accuracy vs GFLOPs (fold means; fold std. in Table III) ------------------------------
ax = axs[0]
SHORT = {"prismr+paug": "PRISM-Net", "prismx+paug": "PRISM-Net-Lite", "resnet18": "ResNet-18", "vgg11_bn": "VGG-11",
         "mobilenet_v3_small": "MNetV3-S", "shufflenet_v2_x1_0": "ShuffleV2", "efficientnet_b0": "EffNet-B0", "vit": "ViT",
         "gfnet": "GFNet", "convnextv2_atto": "ConvNeXtV2", "repvit_m0_9": "RepViT", "mobilenetv4_conv_small": "MNetV4-S"}
# label anchor in data coordinates (x, y, ha, va); a leader line is drawn when the label is far from its point
LAB = {"prismr+paug": (0.52, 93.4, "center", "center"), "prismx+paug": (0.115, 80.6, "left", "center"),
       "resnet18": (0.66, 86.4, "left", "center"), "vgg11_bn": (1.91, 91.7, "center", "center"),
       "mobilenet_v3_small": (0.0125, 76.6, "left", "center"), "shufflenet_v2_x1_0": (0.045, 80.6, "center", "center"),
       "efficientnet_b0": (0.098, 66.1, "right", "center"), "vit": (0.62, 72.7, "left", "center"),
       "gfnet": (0.03, 91.0, "center", "center"), "convnextv2_atto": (0.16, 70.5, "left", "center"),
       "repvit_m0_9": (0.217, 89.4, "center", "center"), "mobilenetv4_conv_small": (0.064, 60.7, "left", "center")}
for m in CFG["main"]:
    e = E[m.split("+")[0]]; acc = S[m]["acc"][0] * 100
    hl = m in ("prismr+paug", "prismx+paug")
    c = BLUE if hl else (VGGC if m == "vgg11_bn" else (RNC if m == "resnet18" else GREY))
    ax.plot(e["gflops"], acc, "o" if hl else "s", ms=5.5 if hl else 3.6, color=c, mfc=c if hl else "white", mew=0.9,
            zorder=4 if hl else 3)
    x0, y0, ha, va = LAB[m]
    tc = c if c != GREY else "#555555"
    if m in ("prismx+paug", "gfnet"):
        ax.annotate(SHORT[m], (e["gflops"], acc), xytext=(x0, y0), fontsize=5.6, color=tc, ha=ha, va=va, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=c, shrinkA=1, shrinkB=3), zorder=5) if m == "prismx+paug" else \
            ax.annotate(SHORT[m], (e["gflops"], acc), xytext=(x0, y0), fontsize=5.6, color=tc, ha=ha, va=va,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY, shrinkA=1, shrinkB=2), zorder=5)
    else:
        ax.text(x0, y0, SHORT[m], fontsize=5.6, color=tc, ha=ha, va=va, fontweight="bold" if hl else "normal", zorder=5)
ax.set_xscale("log"); ax.set_xlabel("GFLOPs per tile (log scale)"); ax.set_ylabel("Accuracy (%)")
ax.set_xticks([0.02, 0.1, 0.5, 2]); ax.set_xticklabels(["0.02", "0.1", "0.5", "2"])
ax.minorticks_off(); ax.set_xlim(0.011, 3.6); ax.set_ylim(57, 96); ax.set_yticks([60, 70, 80, 90])
ax.grid(lw=0.3, color="#e4e4e4")
ax.set_title("(a) Accuracy\u2013cost trade-off", fontsize=7.5, loc="left")

# (b) SWRD adaptation ladder --------------------------------------------------
ax = axs[1]
P = json.load(open(W + "results/swrd_pga.json"))["models"]; F = json.load(open(W + "results/swrd_adapt2.json"))["ft"]
CA = json.load(open(W + "r9_post.json"))   # content-aware alignment (linear form), R9
st = ["Frozen", "PGA-R", "+AdaBN", "CAA", "79 labels", "317 labels"]
for m, nm, c, mk, ls in [("prismr+paug", "PRISM-Net", BLUE, "o", "-"), ("resnet18", "ResNet-18", RNC, "^", "-"),
                          ("resnet18+paug", "ResNet-18+PAug", RNPC, "v", "-"), ("vgg11_bn", "VGG-11", VGGC, "h", "--")]:
    v = [P[m]["frozen"]["acc"], P[m]["R"]["acc"], P[m]["R_bn"]["acc"], CA[m]["caa"]["acc"]]
    if f"{m}|5" in F:
        v += [np.mean(F[f"{m}|5"]["acc"]), np.mean(F[f"{m}|20"]["acc"])]
    v = np.array(v) * 100
    ax.plot(range(len(v)), v, ls, color=c, marker=mk, ms=4 if m == "prismr+paug" else 3, lw=1.3 if m == "prismr+paug" else 0.8,
            label=nm, zorder=3 if m == "prismr+paug" else 2)
pv = [P["prismr+paug"][k]["acc"] * 100 for k in ("frozen", "R", "R_bn")] + [CA["prismr+paug"]["caa"]["acc"] * 100] + [np.mean(F[f"prismr+paug|{b}"]["acc"]) * 100 for b in (5, 20)]
for i, y in enumerate(pv):
    ax.text(i - {2: 0.32, 3: 0.3}.get(i, 0), y + {2: 2.2, 3: 3.4}.get(i, 2.6), f"{y:.1f}", ha="center", fontsize=5.6, color=BLUE)
ax.axvspan(-0.3, 3.3, color="#f2f6fc", zorder=0); ax.text(1.5, 63, "no target\nlabels", ha="center", fontsize=5.8, color="#4a6a96")
ax.set_xticks(range(6)); ax.set_xticklabels(st, fontsize=5.8, rotation=30, ha="right", rotation_mode="anchor"); ax.set_xlim(-0.3, 5.3)
ax.set_ylabel("Accuracy on SWRD (%)"); ax.set_ylim(40, 90); ax.grid(axis="y", lw=0.3, color="#e4e4e4")
ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.1)
ax.set_title("(b) Transfer to a new DR system", fontsize=7.5, loc="left")

fig.savefig(OUT + "fig_results2.pdf", bbox_inches="tight"); fig.savefig(OUT + "fig_results2.png", bbox_inches="tight", dpi=220)
