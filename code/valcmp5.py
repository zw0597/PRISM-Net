"""Round-2 design selection: validation films of all five folds (never reads test logits)."""
import glob, os, numpy as np
from sklearn.metrics import f1_score
D = np.load("./riawelc_80.npz"); Y = D["y"]
rows = {}
for d in ("results", "dev2"):
    for jf in glob.glob(f"./{d}/*_group_f*_s0.json"):
        t = os.path.basename(jf)[:-5]; m, f = t.split("_group_f"); f = int(f[0])
        Z = np.load(jf[:-5] + ".npz"); va = Z["va_idx"]; p = Z["Zva"].argmax(1)
        rows.setdefault(m, {})[f] = ((p == Y[va]).mean(), f1_score(Y[va], p, average="macro"))
for m in sorted(rows, key=lambda m: -np.mean([v[0] for v in rows[m].values()])):
    r = rows[m]
    print(f"{m:24s}", " ".join(f"{r[f][0]*100:5.1f}" if f in r else "    -" for f in range(5)),
          f" | mean acc {np.mean([v[0] for v in r.values()])*100:5.2f} mF1 {np.mean([v[1] for v in r.values()])*100:5.2f} (n={len(r)})")
