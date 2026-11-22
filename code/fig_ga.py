"""Graphical abstract (IEEE format, aspect 660:295). All numbers are taken from the tables of the article."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Wedge, Circle

W = "./"; OUT = "../latex/figs/"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "font.size": 9, "axes.linewidth": 0.7})
NAVY, BLUE, TEAL, ORG, RED, GREY, LIGHT = "#12355b", "#2f6fb0", "#12806a", "#c8621b", "#b3261e", "#5f6b76", "#eef2f6"
FW, FH = 13.2, 5.9
fig = plt.figure(figsize=(FW, FH)); fig.patch.set_facecolor("white")
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, FW); bg.set_ylim(0, FH); bg.axis("off")


def box(x, y, w, h, fc, ec="none", lw=0.8, r=0.06, z=1, ax=bg):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z); ax.add_patch(p); return p


def arrow(x0, y0, x1, y1, c=GREY, lw=1.3, ms=9, z=3, style="-|>", cs="arc3,rad=0"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw, zorder=z,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))


def T(x, y, s, size=9, c="black", w="normal", ha="center", va="center", **k):
    return bg.text(x, y, s, fontsize=size, color=c, fontweight=w, ha=ha, va=va, zorder=5, **k)


# ------------------------------------------------------------------ frame: three columns
X = [0.10, 4.02, 9.02]; CW = [3.82, 4.90, 4.08]; TOP = FH - 0.10; HB = 0.50
HEAD = [("1  Benchmark audit", "official split is contaminated", RED),
        ("2  PRISM-Net", "one acquisition model, used in both directions", NAVY),
        ("3  Evidence", "film-disjoint folds  +  independent system", TEAL)]
for x, w, (h, sub, c) in zip(X, CW, HEAD):
    box(x, 0.10, w, FH - 0.20, "white", ec="#c9d1d9", lw=0.9, r=0.10, z=0)
    box(x, TOP - HB, w, HB, c, r=0.10, z=1); bg.add_patch(Rectangle((x, TOP - HB), w, 0.14, fc=c, ec="none", zorder=1))
    T(x + 0.14, TOP - HB / 2, h, 11.5, "white", "bold", ha="left")
    T(x + w - 0.14, TOP - HB / 2, sub, 8.6, "white", ha="right", style="italic")

# ================================================================== column 1: audit
x0 = X[0]
R = np.load(W + "riawelc_80.npz"); XR_, YR, FILM = R["X"], R["y"], R["film"]
# film strip -> tiles -> both splits
fy = 4.42
box(x0 + 0.18, fy, 1.30, 0.50, "#3a4047", r=0.03)
rs = np.random.default_rng(3)
for k in range(5):
    j = rs.choice(np.where(YR == [0, 1, 2, 3, 1][k])[0])
    a = fig.add_axes([(x0 + 0.22 + k * 0.25) / FW, (fy + 0.13) / FH, 0.23 / FW, 0.23 / FH]); a.imshow(XR_[j], cmap="gray", vmin=0, vmax=255); a.axis("off")
T(x0 + 0.83, fy + 0.60, "one radiographic film", 8.2, GREY)
T(x0 + 0.83, fy - 0.12, "cut into tiles", 8.2, GREY)
box(x0 + 2.20, fy + 0.30, 1.44, 0.30, "#fde9e7", ec=RED, lw=0.9); T(x0 + 2.92, fy + 0.45, "official TRAIN", 8.6, RED, "bold")
box(x0 + 2.20, fy - 0.12, 1.44, 0.30, "#fde9e7", ec=RED, lw=0.9); T(x0 + 2.92, fy + 0.03, "official TEST", 8.6, RED, "bold")
arrow(x0 + 1.52, fy + 0.30, x0 + 2.16, fy + 0.45, RED); arrow(x0 + 1.52, fy + 0.20, x0 + 2.16, fy + 0.03, RED)
box(x0 + 0.18, 3.62, 3.46, 0.50, LIGHT, r=0.05)
T(x0 + 0.30, 3.87, "2 443 / 2 443", 13, RED, "bold", ha="left")
T(x0 + 1.86, 3.95, "official test tiles have a", 8.2, "black", ha="left"); T(x0 + 1.86, 3.78, "pixel-identical training copy", 8.2, "black", ha="left")
# dumbbell: official vs film-grouped
names = ["PRISM-Net", "VGG-11", "ResNet-18", "GFNet", "ShuffleNetV2", "MobileNetV3-S", "ViT", "EfficientNet-B0"]
off = [97.63, 97.91, 97.75, 95.21, 95.17, 94.64, 80.84, 80.02]; grp = [90.56, 89.30, 87.91, 86.19, 83.73, 82.37, 72.70, 66.08]
ax = fig.add_axes([(x0 + 1.22) / FW, 0.86 / FH, 2.40 / FW, 2.14 / FH])
for i, (o, g) in enumerate(zip(off, grp)):
    y = len(names) - 1 - i
    ax.plot([g, o], [y, y], color="#c9d1d9", lw=3.2, solid_capstyle="round", zorder=1)
    ax.scatter(o, y, s=34, color=RED, zorder=3, marker="o"); ax.scatter(g, y, s=38, color=NAVY, zorder=3, marker="D")
    ax.text(g - 1.6, y, f"$-${o-g:.1f}", ha="right", va="center", fontsize=7.6, color=GREY)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1], fontsize=8.2)
ax.get_yticklabels()[-1].set_fontweight("bold")
ax.set_xlim(55, 101); ax.set_ylim(-0.6, len(names) - 0.4); ax.set_xticks([60, 70, 80, 90, 100]); ax.tick_params(axis="x", labelsize=8, length=2.5); ax.tick_params(axis="y", length=0)
ax.set_xlabel("accuracy (%)", fontsize=8.4, labelpad=1.5); ax.grid(axis="x", color="#e6eaee", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
bg.scatter([x0 + 0.34], [3.36], s=34, color=RED, zorder=5); T(x0 + 0.46, 3.36, "official split (repeatability)", 8.2, RED, ha="left")
bg.scatter([x0 + 0.34], [3.17], s=38, color=NAVY, marker="D", zorder=5); T(x0 + 0.46, 3.17, "film-grouped 5-fold, de-duplicated", 8.2, NAVY, ha="left")
T(x0 + CW[0] / 2, 0.27, "reported accuracy is inflated by 7.1 – 13.9 points", 8.8, RED, "bold")

# ================================================================== column 2: method
x0 = X[1]; cw = CW[1]
# --- acquisition model band
by = 3.50; bh = 1.74
box(x0 + 0.14, by, cw - 0.28, bh, LIGHT, r=0.06)
mx = x0 + cw / 2
box(mx - 0.88, by + 0.60, 1.76, 0.74, "white", ec=NAVY, lw=1.3, r=0.06, z=2)
T(mx, by + 1.13, "acquisition model", 8.4, NAVY, "bold")
T(mx, by + 0.82, r"$\tilde I \;=\; c\,\cdot\, I^{\,g}$", 14.5, NAVY)
T(mx, by + 1.53, r"Beer–Lambert  $I = I_0\,e^{-\mu t}$   →   exposure gain $c$,  response exponent $g$", 8.2, GREY)
# left: PAug
box(x0 + 0.24, by + 0.60, 1.08, 0.74, "white", ec=ORG, lw=1.1, r=0.06, z=2)
T(x0 + 0.79, by + 1.12, "PAug", 10, ORG, "bold"); T(x0 + 0.79, by + 0.90, "sample $(c, g)$", 8.0); T(x0 + 0.79, by + 0.72, "while training", 8.0)
arrow(mx - 0.90, by + 0.97, x0 + 1.35, by + 0.97, ORG, 1.6, 11)
# right: PGA
box(x0 + cw - 1.32, by + 0.60, 1.08, 0.74, "white", ec=TEAL, lw=1.1, r=0.06, z=2)
T(x0 + cw - 0.79, by + 1.12, "PGA", 10, TEAL, "bold"); T(x0 + cw - 0.79, by + 0.90, r"estimate $(\hat c, \hat g)$", 8.0); T(x0 + cw - 0.79, by + 0.72, "per radiograph, invert", 7.4)
arrow(mx + 0.90, by + 0.97, x0 + cw - 1.35, by + 0.97, TEAL, 1.6, 11)
T(x0 + 0.79, by + 0.47, "forward", 7.8, ORG, style="italic"); T(x0 + cw - 0.79, by + 0.47, "inverse · no labels", 7.8, TEAL, style="italic")
# tiles: augmented source, aligned target
S = np.load(W + "swrd/swrd_tiles80.npz")["X"]; PG = np.load(W + "swrd/swrd_pga_inputs.npz")["XR"]
j = np.where(YR == 3)[0][40]; src = XR_[j].astype(float) / 255
tiles = [(x0 + 0.26, np.clip(0.85 * src ** 1.6, 0, 1)), (x0 + 0.62, src), (x0 + 0.98, np.clip(1.08 * src ** 0.7, 0, 1)),
         (x0 + cw - 1.30, S[731] / 255.0), (x0 + cw - 0.62, PG[731] / 255.0)]
for xx, im in tiles:
    a = fig.add_axes([xx / FW, (by + 0.06) / FH, 0.34 / FW, 0.34 / FH]); a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.5); s.set_color("#8a949e")
arrow(x0 + cw - 0.94, by + 0.23, x0 + cw - 0.64, by + 0.23, TEAL, 1.3, 8)
T(mx, by + 0.30, "proved to recover source", 7.8, NAVY); T(mx, by + 0.13, "intensities exactly under the model", 7.8, NAVY)

# --- network pipeline
py = 2.34; ph = 0.92
def stage(x, w, title, lines, c, fc="white"):
    box(x, py, w, ph, fc, ec=c, lw=1.2, r=0.06, z=2); T(x + w / 2, py + ph - 0.17, title, 8.8, c, "bold")
    for k, l in enumerate(lines): T(x + w / 2, py + ph - 0.38 - 0.17 * k, l, 7.6)
sx = x0 + 0.14
a = fig.add_axes([(sx + 0.02) / FW, (py + 0.22) / FH, 0.48 / FW, 0.48 / FH]); a.imshow(XR_[np.where(YR == 2)[0][10]], cmap="gray"); a.set_xticks([]); a.set_yticks([])
T(sx + 0.26, py + 0.10, "80×80 tile", 7.4, GREY)
arrow(sx + 0.54, py + 0.46, sx + 0.70, py + 0.46)
stage(sx + 0.72, 1.62, "ResNet-18 trunk + RASG", [], NAVY)
# RASG icon: polar gate
cx, cy = sx + 1.08, py + 0.36
for r_, al in ((0.26, 0.25), (0.18, 0.5), (0.10, 0.85)):
    for k in range(8):
        bg.add_patch(Wedge((cx, cy), r_, k * 45, k * 45 + 45, width=0.08, fc=BLUE, alpha=al * (0.55 + 0.45 * abs(np.cos(np.deg2rad(k * 45 + 22)))), ec="white", lw=0.4, zorder=4))
T(sx + 1.86, py + 0.52, "radial–angular", 7.6); T(sx + 1.86, py + 0.36, "spectral gate", 7.6); T(sx + 1.86, py + 0.18, "+3% params", 7.6, BLUE, "bold")
arrow(sx + 2.36, py + 0.46, sx + 2.50, py + 0.46)
stage(sx + 2.52, 1.02, "Evidential head", ["Dirichlet $\\alpha$", "uncertainty", "$u = K/\\Sigma\\alpha$"], ORG)
arrow(sx + 3.56, py + 0.46, sx + 3.70, py + 0.46)
stage(sx + 3.72, 0.90, "Conformal", ["prediction set", "coverage", "$\\geq 1-\\varepsilon$"], TEAL)
# decisions
dy = 1.56
arrow(sx + 4.17, py - 0.02, sx + 3.40, dy + 0.42, TEAL, 1.2, 9); arrow(sx + 4.17, py - 0.02, sx + 4.17, dy + 0.42, TEAL, 1.2, 9)
box(sx + 2.30, dy, 1.40, 0.40, "#e3f3ee", ec=TEAL, lw=1.0, z=2); T(sx + 3.00, dy + 0.20, "single label → accept", 8.0, TEAL, "bold")
box(sx + 3.78, dy, 0.84, 0.40, "#fdeedd", ec=ORG, lw=1.0, z=2); T(sx + 4.20, dy + 0.20, "set → refer", 8.0, ORG, "bold")
T(sx + 0.02, dy + 0.26, "classes: no defect · porosity ·", 7.8, GREY, ha="left"); T(sx + 0.02, dy + 0.09, "lack of penetration · crack", 7.8, GREY, ha="left")
# --- commissioning chain
cy0 = 0.26; box(x0 + 0.14, cy0, cw - 0.28, 1.00, LIGHT, r=0.06)
T(x0 + 0.26, cy0 + 0.82, "Commissioning on a new radiographic system", 8.4, NAVY, "bold", ha="left")
steps = [("PGA", "align input", TEAL), ("AdaBN", "re-estimate BN", TEAL), ("Recalibrate", "25 radiographs", ORG), ("Inspect", "90% coverage", NAVY)]
sw = (cw - 0.28 - 0.24 - 3 * 0.16) / 4
for k, (a_, b_, c) in enumerate(steps):
    xx = x0 + 0.26 + k * (sw + 0.16)
    box(xx, cy0 + 0.10, sw, 0.56, "white", ec=c, lw=1.0, z=2); T(xx + sw / 2, cy0 + 0.49, a_, 8.8, c, "bold"); T(xx + sw / 2, cy0 + 0.26, b_, 7.8)
    if k < 3: arrow(xx + sw + 0.01, cy0 + 0.38, xx + sw + 0.15, cy0 + 0.38, GREY, 1.1, 7)
T(x0 + cw - 0.26, cy0 + 0.82, "zero labels  →  few labels", 7.8, GREY, ha="right", style="italic")

# ================================================================== column 3: evidence
x0 = X[2]; cw = CW[2]
T(x0 + 0.16, 5.12, "RIAWELC · 5 film-disjoint folds · 29 films", 8.6, NAVY, "bold", ha="left")
mods = ["PRISM-Net", "VGG-11", "ResNet-18"]; cols = [NAVY, "#8a5a2b", "#8a949e"]
panels = [("accuracy (%)", [90.6, 89.3, 87.9], (84, 92), "{:.1f}"), ("15 corruptions (%)", [87.8, 83.4, 83.8], (80, 89), "{:.1f}"),
          ("ECE (%) ↓", [4.6, 6.0, 7.0], (0, 8), "{:.1f}"), ("GFLOPs ↓", [0.49, 1.91, 0.51], (0, 2.2), "{:.2f}")]
pw = (cw - 0.36 - 3 * 0.20) / 4
for k, (ttl, v, yl, fm) in enumerate(panels):
    a = fig.add_axes([(x0 + 0.20 + k * (pw + 0.20)) / FW, 3.70 / FH, pw / FW, 1.02 / FH])
    b = a.bar(range(3), v, color=cols, width=0.72, zorder=2); a.set_ylim(*yl); a.set_xticks([]); a.set_yticks([])
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
    for i, val in enumerate(v):
        a.text(i, val + (yl[1] - yl[0]) * 0.03, fm.format(val), ha="center", va="bottom", fontsize=7.6, color=cols[i] if i == 0 else "#333333", fontweight="bold" if i == 0 else "normal")
    a.set_title(ttl, fontsize=8.0, pad=10)
for i, (m, c) in enumerate(zip(mods, cols)):
    xx = x0 + 0.30 + i * 1.28; bg.add_patch(Rectangle((xx, 3.46), 0.13, 0.11, fc=c, ec="none", zorder=5)); T(xx + 0.19, 3.515, m, 8.2, c, "bold" if i == 0 else "normal", ha="left")
box(x0 + 0.16, 2.92, cw - 0.32, 0.42, LIGHT, r=0.05)
T(x0 + cw / 2, 3.20, "highest mean accuracy; on par with VGG-11 at ¼ of its operations", 7.9, "black")
T(x0 + cw / 2, 3.03, "best calibrated, most accurate under all 15 acquisition corruptions", 7.9, "black")

T(x0 + 0.16, 2.68, "SWRD · independent digital-radiography system", 8.6, TEAL, "bold", ha="left")
a = fig.add_axes([(x0 + 0.52) / FW, 1.12 / FH, 2.10 / FW, 1.36 / FH])
lab = ["frozen", "PGA", "+AdaBN", "79", "317"]; val = [45.8, 47.8, 50.7, 70.8, 77.7]; cc = [GREY, TEAL, TEAL, ORG, ORG]
a.plot(range(5), val, color="#b9c2cb", lw=1.4, zorder=1)
a.scatter(range(5), val, s=52, color=cc, zorder=3, edgecolor="white", lw=0.8)
for i, v in enumerate(val): a.text(i, v + 3.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.0, color=cc[i], fontweight="bold")
a.axvspan(-0.5, 2.5, color="#e3f3ee", zorder=0, lw=0); a.axvspan(2.5, 4.5, color="#fdeedd", zorder=0, lw=0)
a.text(1.0, 93, "no target labels", ha="center", fontsize=7.8, color=TEAL); a.text(3.5, 93, "labeled", ha="center", fontsize=7.8, color=ORG)
a.set_xlim(-0.5, 4.5); a.set_ylim(35, 101); a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=7.6); a.set_yticks([40, 60, 80]); a.tick_params(labelsize=7.8, length=2.5)
a.set_ylabel("PRISM-Net acc. (%)", fontsize=8.0, labelpad=2)
for s in ("top", "right"): a.spines[s].set_visible(False)
# coverage gauge
gx = x0 + 2.86; gw = cw - 2.86 - 0.18
T(gx + gw / 2, 2.40, "90% conformal sets:", 7.9, NAVY, "bold"); T(gx + gw / 2, 2.23, "coverage on SWRD", 7.9, NAVY, "bold")
for k, (v, c, l1) in enumerate([(49.2, RED, "frozen"), (90.3, TEAL, "+25 labeled")]):
    yy = 1.80 - k * 0.56
    box(gx, yy, gw, 0.20, "#e1e6ea", r=0.03); box(gx, yy, gw * v / 100, 0.20, c, r=0.03, z=2)
    bg.plot([gx + gw * 0.9] * 2, [yy - 0.04, yy + 0.24], color="black", lw=1.0, zorder=6)
    T(gx, yy + 0.31, f"{v:.1f}%", 9.5, c, "bold", ha="left"); T(gx + gw, yy + 0.31, l1.split("\n")[0], 7.4, GREY, ha="right")
    if "\n" in l1: T(gx + gw, yy - 0.11, l1.split("\n")[1], 7.4, GREY, ha="right")
T(gx + gw * 0.9, 1.10, "nominal", 7.0, "black")
T(x0 + cw / 2, 0.60, "every frozen model loses > 30 points across systems;", 7.9, "black")
T(x0 + cw / 2, 0.42, "PGA + AdaBN improve 13 of 15 models, by up to 6.8 points", 7.9, TEAL, "bold")
T(x0 + cw / 2, 0.24, "nested selection without test films retains 90.3%", 7.6, GREY)

fig.savefig(OUT + "graphical_abstract.pdf"); fig.savefig(OUT + "graphical_abstract.png", dpi=200)
print("ok")
