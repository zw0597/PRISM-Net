"""Build a compact array cache of RIAWELC with provenance metadata.

Each PNG name encodes <film>_<ImgK>_A<a>_S<s>_[r][c].png, where <film> is the
physical radiographic film, ImgK one of its digitised views, S a weld segment
and [r][c] the tile position. We keep this provenance so that leakage-free
(film-grouped) splits can be built.
"""
import os, re, hashlib, json, sys
import numpy as np
from PIL import Image

ROOT = "./data/raw/DB - Copy"
OUT = "./riawelc_%d.npz"
RES = int(sys.argv[1]) if len(sys.argv) > 1 else 112
CLS = {"NoDifetto": 0, "Difetto2": 1, "Difetto4": 2, "Difetto1": 3}  # ND, PO, LP, CR
SPL = {"training": 0, "validation": 1, "testing": 2}
pat = re.compile(r"^(.+?)_(Img\d+)_A(\d+)_S(\d+)_\[(\d+)\]\[(\d+)\]( - Copia)?\.png$")

recs, imgs, hashes = [], [], {}
dup_exact = 0
for s in sorted(SPL):
    for c in sorted(CLS):
        d = os.path.join(ROOT, s, c)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            m = pat.match(f)
            if m is None:
                print("skip", f); continue
            raw = open(os.path.join(d, f), "rb").read()
            im = Image.open(os.path.join(d, f)).convert("L")
            a = np.asarray(im)
            h = hashlib.md5(a.tobytes()).hexdigest()
            recs.append(dict(file=f, split=SPL[s], y=CLS[c], film=m.group(1),
                             view=m.group(2), seg=int(m.group(4)),
                             r=int(m.group(5)), c=int(m.group(6)),
                             copia=m.group(7) is not None, md5=h))
            imgs.append(np.asarray(im.resize((RES, RES), Image.BILINEAR), dtype=np.uint8))

N = len(recs)
md5 = [r["md5"] for r in recs]
from collections import Counter, defaultdict
cnt = Counter(md5)
groups = defaultdict(list)
for i, h in enumerate(md5):
    groups[h].append(i)
dup_groups = [g for g in groups.values() if len(g) > 1]
cross_split = sum(1 for g in dup_groups if len({recs[i]["split"] for i in g}) > 1)
cross_label = sum(1 for g in dup_groups if len({recs[i]["y"] for i in g}) > 1)
stats = dict(N=N, unique=len(groups), dup_groups=len(dup_groups),
             dup_images=sum(len(g) for g in dup_groups),
             dup_groups_cross_split=cross_split, dup_groups_cross_label=cross_label,
             copia=sum(r["copia"] for r in recs),
             films=len({r["film"] for r in recs}),
             radiographs=len({(r["film"], r["view"]) for r in recs}))
print(json.dumps(stats, indent=1))
np.savez_compressed(OUT % RES, X=np.stack(imgs),
                    y=np.array([r["y"] for r in recs], np.int64),
                    split=np.array([r["split"] for r in recs], np.int64),
                    film=np.array([r["film"] for r in recs]),
                    view=np.array([r["view"] for r in recs]),
                    md5=np.array(md5), fname=np.array([r["file"] for r in recs]))
json.dump(stats, open("./data_stats.json", "w"), indent=1)
