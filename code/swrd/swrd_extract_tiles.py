import json, glob, os, random, numpy as np
from PIL import Image, ImageDraw
R = os.path.expanduser("~/mnt/焊接质量智能检测和评估/SWXD_Data/Raw_data/")
CLS = {"气孔": 1, "未焊透": 2, "裂纹": 3}; T = 227; H = T // 2; CAP = 1500; PER = 3
rng = random.Random(0)
names = sorted(os.path.basename(f)[:-5] for f in glob.glob(R + "json/*.json")); rng.shuffle(names)
imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(R + "images/*")}
import time, pickle, sys
ST = os.path.expanduser("~/tmp/sw/state.pkl"); t0 = time.time()
if os.path.exists(ST): start, cnt, X, Y, IMG, CEN = pickle.load(open(ST, "rb"))
else: start, cnt, X, Y, IMG, CEN = 0, [0] * 4, [], [], [], []

def inter(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

ii = start
for ii in range(start, len(names)):
    nm = names[ii]; rng = random.Random(ii)
    if min(cnt) >= CAP: break
    if time.time() - t0 > 33:
        pickle.dump((ii, cnt, X, Y, IMG, CEN), open(ST, "wb")); print("pause", ii, cnt); sys.exit(0)
    d = json.load(open(R + "json/" + nm + ".json")); S = d["shapes"]
    welds = [s for s in S if s["label"] == "焊缝"]; others = [s for s in S if s["label"] not in ("焊缝", None)]
    if not welds or nm not in imgs: continue
    labs = {s['label'] for s in others}
    if cnt[0] >= CAP and not any(cnt[c] < CAP and l in labs for l, c in CLS.items()): continue
    need = [c for c in range(4) if cnt[c] < CAP]
    if need == [0] and False: pass
    try: a = np.array(Image.open(imgs[nm]))
    except Exception: print('unreadable', nm); continue
    h, w = a.shape[:2]
    if a.ndim == 3: a = a[..., 0]
    m = Image.new("L", (w, h), 0); dr = ImageDraw.Draw(m)
    for s in welds: dr.polygon([tuple(p) for p in s["points"]], fill=1)
    m = np.array(m, bool)
    if m.sum() < 1000: continue
    lo, hi = np.percentile(a[m], [1, 99]);
    if hi <= lo: continue
    a8 = np.clip((a.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    boxes = [(np.min([p[0] for p in s["points"]]), np.min([p[1] for p in s["points"]]),
              np.max([p[0] for p in s["points"]]), np.max([p[1] for p in s["points"]]), s["label"]) for s in others]
    def ok(cx, cy, lab):
        if cx - H < 0 or cy - H < 0 or cx - H + T > w or cy - H + T > h: return False
        if not m[int(cy), int(cx)]: return False
        tb = (cx - H, cy - H, cx - H + T, cy - H + T)
        return not any(b[4] != lab and inter(tb, b) for b in boxes)
    def add(cx, cy, c):
        cx, cy = int(round(cx)), int(round(cy))
        t = Image.fromarray(a8[cy - H:cy - H + T, cx - H:cx - H + T]).resize((80, 80), Image.BILINEAR)
        X.append(np.asarray(t, np.uint8)); Y.append(c); IMG.append(nm); CEN.append((cx, cy)); cnt[c] += 1
    for lab, c in CLS.items():
        if cnt[c] >= CAP: continue
        cands = []
        for b in boxes:
            if b[4] != lab: continue
            x0, y0, x1, y1 = b[:4]; L = max(x1 - x0, y1 - y0)
            k = max(1, int(np.ceil(L / T))) if L > T else 1
            for j in range(k):
                t = (j + 0.5) / k
                cx, cy = ((x0 + (x1 - x0) * t, (y0 + y1) / 2) if (x1 - x0) >= (y1 - y0) else ((x0 + x1) / 2, y0 + (y1 - y0) * t))
                if ok(cx, cy, lab): cands.append((cx, cy))
        rng.shuffle(cands)
        for cx, cy in cands[:PER]:
            if cnt[c] < CAP: add(cx, cy, c)
    if cnt[0] < CAP:
        ys, xs = np.nonzero(m[::16, ::16]); n0 = 0
        for _ in range(200):
            if n0 >= PER or cnt[0] >= CAP: break
            j = rng.randrange(len(ys)); cx, cy = xs[j] * 16, ys[j] * 16
            if ok(cx, cy, "焊缝"): add(cx, cy, 0); n0 += 1
    if ii % 100 == 0: print(ii, cnt, flush=True)
pickle.dump((len(names), cnt, X, Y, IMG, CEN), open(ST, "wb")); print("done", ii, cnt, flush=True)
np.savez_compressed(os.path.expanduser("~/mnt/焊接质量智能检测和评估/PRISM-Net_SWRD_external_test/swrd_tiles80.npz"),
                    X=np.array(X), y=np.array(Y), img=np.array(IMG), cen=np.array(CEN))
