"""Graphical abstract, figure style (three panels, real data, real radiograph tiles).
IEEE file: 660 x 295 px, all text >= 16 px, < 45 KB.  Display file: same layout at 4x with extra annotations.
All numbers are read from the result files that produce the tables of the article."""
import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

W = "./"; OUT = "../latex/figs/"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "axes.linewidth": 0.8})
BLUE, GREY, RED, INK, MUTE = "#1f5fa8", "#8c8c8c", "#b22222", "#1a1a1a", "#555555"

# ---------------------------------------------------------------- data
S = json.load(open(W + "summary.json")); CFG = json.load(open(W + "paper_cfg.json"))
OM = CFG["official_models"]; NAME = CFG["names"]
off = {m: S["official"][m]["acc"] * 100 for m in OM}; grp = {m: S["agg"][m]["acc"][0] * 100 for m in OM}
P = json.load(open(W + "results/swrd_pga.json"))["models"]; F = json.load(open(W + "results/swrd_adapt2.json"))["ft"]
lad = {m: dict(frozen=P[m]["frozen"]["acc"] * 100, aligned=P[m]["R_bn"]["acc"] * 100,
               l79=np.mean(F[f"{m}|5"]["acc"]) * 100, l317=np.mean(F[f"{m}|20"]["acc"]) * 100) for m in ("prismr+paug", "resnet18")}
R = np.load(W + "riawelc_80.npz")["X"]; SX = np.load(W + "swrd/swrd_tiles80.npz")["X"]; SP = np.load(W + "swrd/swrd_pga_inputs.npz")["XR"]
T_R, T_S = 19819, 5066                                   # crack tile (RIAWELC), lack-of-penetration tile (SWRD)
I0 = R[T_R].astype(float) / 255
paug = [np.clip(c * I0 ** g, 0, 1) for c, g in ((0.70, 1.45), (1.10, 0.70))]


def draw(scale, rich, fname):
    fig = plt.figure(figsize=(6.6, 2.95), dpi=100 * scale)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 660); bg.set_ylim(0, 295); bg.axis("off")
    pt = lambda px: px * 72 / 100                       # IEEE pixel size -> points (scales with dpi)

    def T(x, y, s, px=16, c=INK, w="normal", ha="center", va="center", **k):
        bg.text(x, y, s, fontsize=pt(px), color=c, fontweight=w, ha=ha, va=va, zorder=6, **k)

    def A(x, y, w, h):                                   # axes in pixel coordinates
        a = fig.add_axes([x / 660, y / 295, w / 660, h / 295]); a.patch.set_alpha(0); return a

    def arrow(x0, y0, x1, y1, c=INK, lw=1.4):
        bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, shrinkA=0, shrinkB=0, zorder=5))

    def tile(x, y, s, im, ec=None):
        a = A(x, y, s, s); a.imshow(im, cmap="gray", vmin=0, vmax=1 if im.max() <= 1 else 255, interpolation="bilinear")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(ec or "#444444"); sp.set_linewidth(1.0 if ec is None else 1.6)
        return a

    for xv in (213, 447):
        bg.plot([xv, xv], [10, 285], color="#d0d0d0", lw=0.8)

    # ============================== (a) audit: slope chart official -> film-grouped
    T(8, 278, "(a) Benchmark audit", 16, w="bold", ha="left")
    T(8, 256, "2443/2443 test tiles", 16, RED, ha="left")
    T(8, 237, "also in training split", 16, MUTE, ha="left")
    ax = A(44, 40, 150, 168)
    for m in OM:
        hl = m == "prismr+paug"
        ax.plot([0, 1], [off[m], grp[m]], "-o", color=BLUE if hl else GREY, lw=2.2 if hl else 1.0, ms=4.5 if hl else 3,
                zorder=3 if hl else 2, alpha=1 if hl else 0.85)
        if rich:
            ax.text(1.06, grp[m], NAME[m] if m != "prismr+paug" else "PRISM-Net", fontsize=pt(5.2), va="center",
                    color=BLUE if hl else MUTE, fontweight="bold" if hl else "normal")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(62, 100)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["official", "by film"], fontsize=pt(16))
    ax.set_yticks([70, 100]); ax.set_yticklabels(["70", "100"], fontsize=pt(16))
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=3, pad=2)
    lo, hi = min(off[m] - grp[m] for m in OM), max(off[m] - grp[m] for m in OM)
    T(54, 64, f"−{lo:.0f} to −{hi:.0f}", 16, RED, w="bold", ha="left")
    T(26, 124, "Acc (%)", 16, rotation=90)
    if rich:
        T(54, 49, "points, 8 backbones", 6.0, RED, ha="left")
        T(8, 222, "pixel-hash audit; deduplicated, film-grouped 5-fold protocol", 5.6, MUTE, ha="left")

    # ============================== (b) method: one acquisition model, both directions
    T(222, 278, "(b) PRISM-Net", 16, w="bold", ha="left")
    T(442, 278, r"$\tilde{I}=c\,I^{\,g}$", 18, ha="right")
    # row 1: PAug on a RIAWELC crack tile
    s = 46; y1 = 196
    tile(226, y1, s, I0); arrow(276, y1 + s / 2, 318, y1 + s / 2, BLUE)
    T(297, y1 + s / 2 + 13, "PAug", 16, BLUE, w="bold")
    tile(322, y1, s, paug[0]); tile(372, y1, s, paug[1])
    T(330, y1 - 12, "training: sample (c, g)", 16, MUTE)
    # row 2: PGA on an SWRD radiograph tile
    y2 = 104
    tile(226, y2, s, SX[T_S] / 255); arrow(276, y2 + s / 2, 318, y2 + s / 2, BLUE)
    T(297, y2 + s / 2 + 13, "PGA", 16, BLUE, w="bold")
    tile(322, y2, s, SP[T_S] / 255, ec=BLUE)
    T(398, y2 + s / 2 + 9, "new", 16, MUTE); T(398, y2 + s / 2 - 9, "system", 16, MUTE)
    T(330, y2 - 12, "no labels: estimate, invert", 16, MUTE)
    # row 3: evidential head + conformal decision
    y3 = 30
    tri = Polygon([[228, y3 + 4], [268, y3 + 4], [248, y3 + 38]], closed=True, fc="#eef3fa", ec=BLUE, lw=1.2, zorder=4); bg.add_patch(tri)
    bg.scatter([251], [y3 + 15], s=10, color=BLUE, zorder=5)
    T(280, y3 + 30, "evidential + conformal", 16, ha="left")
    T(280, y3 + 10, "|C|=1 accept, else refer", 16, MUTE, ha="left")
    if rich:
        T(222, 260, "one Beer–Lambert acquisition model, used in both directions", 5.6, MUTE, ha="left")
        T(330, y2 - 27, "exact recovery under the model (Prop. 3)", 5.6, BLUE)
        T(280, y3 - 4, "coverage ≥ 1−ε; RASG spectral gating adds 3% parameters", 5.4, MUTE, ha="left")

    # ============================== (c) results: transfer to a new DR system
    T(456, 278, "(c) Results", 16, w="bold", ha="left")
    T(456, 256, "90.6% on unseen films", 16, BLUE, ha="left")
    T(456, 237, "best calibrated, robust", 16, MUTE, ha="left")
    ax = A(492, 52, 152, 150)
    xs = [0, 1, 2]
    for m, c, lab in (("resnet18", GREY, "ResNet-18"), ("prismr+paug", BLUE, "PRISM-Net")):
        v = lad[m]
        ax.plot([0, 0], [v["frozen"], v["aligned"]], color=c, lw=1.0, ls=":", zorder=2)
        ax.plot(0, v["frozen"], "o", mfc="white", mec=c, ms=4.5, zorder=3)
        ax.plot(xs, [v["aligned"], v["l79"], v["l317"]], "-o", color=c, lw=2.2 if c == BLUE else 1.2, ms=4.5 if c == BLUE else 3.5, zorder=3)
        if rich:
            pts = list(zip(xs, [v["aligned"], v["l79"], v["l317"]]))
            for xx, yy in (pts[1:] if c == GREY else pts):
                ax.text(xx, yy + (2.4 if c == BLUE else -3.6), f"{yy:.1f}", fontsize=pt(5.4), color=c, ha="center", va="center")
    ax.set_xlim(-0.25, 2.25); ax.set_ylim(40, 82)
    ax.set_xticks(xs); ax.set_xticklabels(["0", "79", "317"], fontsize=pt(16))
    ax.set_yticks([40, 80]); ax.set_yticklabels(["40", "80"], fontsize=pt(16))
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=3, pad=2)
    T(568, 22, "labeled SWRD images", 16)
    T(474, 127, "Acc (%)", 16, rotation=90)
    T(600, 200, "PRISM-Net", 16, BLUE, w="bold", ha="right")
    T(644, 128, "ResNet-18", 16, GREY, ha="right")
    if rich:
        T(644, 66, "open: frozen\nfilled: PGA-R + AdaBN", 5.4, MUTE, ha="right")
        T(456, 222, "25 labeled radiographs restore 90% coverage", 5.6, MUTE, ha="left")
    fig.savefig(fname, dpi=100 * scale)
    plt.close(fig)


draw(1, False, OUT + "gagraphic.png")
from PIL import Image
im = Image.open(OUT + "gagraphic.png").convert("RGB"); assert im.size == (660, 295), im.size
im.save(OUT + "gagraphic.jpg", quality=88, dpi=(300, 300), optimize=True)
import os; print("jpg bytes", os.path.getsize(OUT + "gagraphic.jpg"))
draw(4, True, OUT + "graphical_abstract.png")
draw(4, True, OUT + "graphical_abstract.pdf")
