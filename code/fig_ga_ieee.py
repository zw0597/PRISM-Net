"""IEEE-compliant graphical abstract: exactly 660 x 295 px, sans-serif labels >= 16 px, numbers from the article's tables."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
DPI = 100; PX = lambda p: p * 72 / DPI          # pixel size -> points
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans"], "mathtext.fontset": "dejavusans"})
NAVY, TEAL, ORG, RED = "#12355b", "#0d6b58", "#a84f12", "#a3211a"
fig = plt.figure(figsize=(6.6, 2.95), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 660); ax.set_ylim(0, 295); ax.axis("off")
def T(x, y, s, px=16, c="black", w="normal", ha="center"): ax.text(x, y, s, fontsize=PX(px), color=c, fontweight=w, ha=ha, va="center", zorder=5)
def box(x, y, w, h, fc, ec="none", lw=1.2, r=6): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=1))
cols = [(4, 196, RED, "1 Audit"), (204, 252, NAVY, "2 PRISM-Net"), (460, 196, TEAL, "3 Evidence")]
for x, w, c, h in cols:
    box(x, 4, w, 287, "white", ec="#b8c0c8", lw=1.0); box(x, 257, w, 34, c); ax.add_patch(Rectangle((x, 257), w, 8, fc=c, ec="none", zorder=1))
    T(x + w / 2, 273, h, 18, "white", "bold")
# 1 audit
T(102, 226, "2443 / 2443", 26, RED, "bold"); T(102, 197, "official test tiles", 16); T(102, 177, "copied in training", 16)
box(14, 92, 176, 62, "#fbe9e7"); T(102, 136, "film-grouped folds:", 16); T(102, 111, "−7 to −14 points", 19, RED, "bold")
T(102, 62, "8 architectures", 16, "#3c4650"); T(102, 38, "re-evaluated", 16, "#3c4650")
# 2 method
T(330, 232, "Beer–Lambert model", 16, NAVY, "bold")
box(278, 176, 104, 40, "white", ec=NAVY, lw=1.6); T(330, 196, r"$\tilde I = c \cdot I^{\,g}$", 19, NAVY)
T(234, 196, "PAug", 17, ORG, "bold"); T(426, 196, "PGA", 17, TEAL, "bold")
for a, b, c in ((276, 259, ORG), (384, 403, TEAL)):
    ax.add_patch(FancyArrowPatch((a, 196), (b, 196), arrowstyle="-|>", mutation_scale=14, color=c, lw=2, shrinkA=0, shrinkB=0, zorder=4))
T(242, 160, "training", 16); T(414, 160, "no labels", 16)
box(214, 84, 232, 54, "#eef2f6"); T(330, 123, "spectral gating + evidential", 16); T(330, 100, "head + conformal sets", 16)
T(330, 58, "single label → accept", 16, TEAL, "bold"); T(330, 32, "label set → inspector", 16, ORG, "bold")
# 3 evidence
T(558, 228, "90.6%", 26, TEAL, "bold"); T(558, 200, "on unseen films", 16)
T(558, 176, "best calibrated", 16); 
box(470, 84, 176, 70, "#e3f3ee"); T(558, 138, "new X-ray system", 16, TEAL, "bold"); T(558, 114, "46 → 51 → 78%", 19, TEAL, "bold"); T(558, 94, "0 · 0 · 317 labels", 16)
T(558, 58, "90% coverage restored", 16); T(558, 34, "with 25 radiographs", 16)
fig.savefig("../latex/figs/gagraphic.png", dpi=DPI)
from PIL import Image
im = Image.open("../latex/figs/gagraphic.png").convert("RGB"); print(im.size)
im.save("../latex/figs/gagraphic.jpg", quality=78, dpi=(300, 300), optimize=True)
im.resize((1320, 590), Image.NEAREST).save("./tmp/ga_ieee_prev.png")
