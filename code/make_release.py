"""Build the Zenodo release: code, splits, duplicate lists, protocols and the predictions of every model in the paper.
No images are redistributed (RIAWELC and SWRD must be obtained from their authors); no model weights (size).
Usage: python make_release.py <code_dir> <out_dir>"""
import os, sys, json, csv, shutil, glob, hashlib
from collections import defaultdict
import numpy as np

CODE, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
J = lambda *p: os.path.join(OUT, *p)

# ---- code (curated copy used for the paper) ----
shutil.copytree(CODE, J("code"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))

# ---- splits and duplicate lists ----
os.makedirs(J("splits"))
for f in ("splits.json", "leak_stats.json"): shutil.copy(f, J("splits", f))
shutil.copy("swrd/swrd_split.json", J("splits", "swrd_split.json"))
D = np.load("riawelc_80.npz")
fname, md5, split, y, film, view = (D[k] for k in ("fname", "md5", "split", "y", "film", "view"))
SPL = {0: "training", 1: "validation", 2: "testing"}; CLS = {0: "ND", 1: "PO", 2: "LP", 3: "CR"}
FOLDER = {0: "NoDifetto", 1: "Difetto2", 2: "Difetto4", 3: "Difetto1"}
path = lambda i: f"{SPL[int(split[i])]}/{FOLDER[int(y[i])]}/{fname[i]}"
S = json.load(open("splits.json"))
role = defaultdict(dict)
for f in S:
    for part in ("train", "val", "test"):
        for i in S[f][part]: role[int(i)][int(f)] = part
groups = defaultdict(list)
for i, h in enumerate(md5): groups[h].append(i)
gid = {h: k for k, h in enumerate(sorted(groups, key=lambda h: min(groups[h])))}
with open(J("splits", "riawelc_manifest.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "path", "official_split", "class", "film", "radiograph_view", "pixel_md5", "dup_group", "dup_group_size"] +
               [f"fold{f}_role" for f in range(5)])
    for i in range(len(md5)):
        w.writerow([i, path(i), SPL[int(split[i])], CLS[int(y[i])], film[i], view[i], md5[i], gid[md5[i]], len(groups[md5[i]])] +
                   [role[i].get(f, "removed") for f in range(5)])
with open(J("splits", "riawelc_duplicate_groups.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["dup_group", "pixel_md5", "size", "official_splits", "classes", "files"])
    for h, idx in sorted(groups.items(), key=lambda kv: gid[kv[0]]):
        if len(idx) < 2: continue
        w.writerow([gid[h], h, len(idx), "|".join(sorted({SPL[int(split[i])] for i in idx})),
                    "|".join(sorted({CLS[int(y[i])] for i in idx})), "|".join(path(i) for i in idx)])

# ---- protocols ----
os.makedirs(J("protocols"))
for f in ("reviewer_protocol.md", "risk_protocol.md", "swrd/swrd_protocol.md"):
    if os.path.exists(f): shutil.copy(f, J("protocols", os.path.basename(f)))

# ---- predictions ----
def copy_runs(src, dst, logs=None):
    os.makedirs(J("predictions", dst))
    n = 0
    for p in sorted(glob.glob(f"{src}/*.npz")):
        b = p[:-4]
        if not os.path.exists(b + ".json"): continue
        shutil.copy(p, J("predictions", dst)); shutil.copy(b + ".json", J("predictions", dst)); n += 1
        lg = b + ".log" if logs is None else None
        if lg and os.path.exists(lg): shutil.copy(lg, J("predictions", dst))
    if logs:
        for p in sorted(glob.glob(logs)): shutil.copy(p, J("predictions", dst))
    return n
N = {"riawelc_cpu_1000steps": copy_runs("results", "riawelc_cpu_1000steps"),
     "riawelc_imagenet_112px": copy_runs("results_pt", "riawelc_imagenet_112px"),
     "riawelc_gpu_1000steps": copy_runs("results_gpu_1000", "riawelc_gpu_1000steps", "logs_gpu/*_1000.log"),
     "riawelc_gpu_3000steps": copy_runs("results_gpu_3000", "riawelc_gpu_3000steps", "logs_gpu/*_3000.log")}
for v in ("oc", "canon", "occanon"):
    N[f"riawelc_gpu_1000steps_r8_{v}"] = copy_runs(f"results_r8_{v}", f"riawelc_gpu_1000steps_r8_{v}", f"logs_gpu/r8_{v}_f*.log")
os.makedirs(J("predictions", "swrd"))
for f in ("swrd_probs.npz", "swrd_probs_adabn.npz", "swrd_probs_pga.npz", "swrd_probs_std.npz", "swrd_probs_r8.npz", "swrd_probs_r9.npz", "pga_params.json", "round2_selection.json"):
    shutil.copy(f"swrd/{f}", J("predictions", "swrd", f))
shutil.copytree("swrd/ft", J("predictions", "swrd", "finetune"))
N["swrd_finetune_npy"] = len(glob.glob("swrd/ft/*.npy"))
json.dump(N, open(J("predictions", "counts.json"), "w"), indent=1)

# ---- README and licences ----
shutil.copy("release_README.md", J("README.md"))
open(J("LICENSE"), "w").write("""Code (folder code/): MIT License

Copyright (c) 2026 Wei Zeng, Xuewu Liu, Ziyang Chen, Jiaming Li, Yang Chen, Jinkai Lan, Shaoyi Du, Zhiqiang Tian

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

All other files (splits/, protocols/, predictions/): Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. RIAWELC and SWRD images are not included and remain under the terms of
their original publishers.
""")

# ---- remove local absolute paths from text files (warnings in logs, one hard-coded data path) ----
WD = os.path.abspath(".")
for root, _, files in os.walk(OUT):
    for f in files:
        if f.rsplit(".", 1)[-1] not in ("log", "py", "sh", "md", "json", "txt", "csv"): continue
        p = os.path.join(root, f); t = open(p, encoding="utf-8", errors="surrogateescape").read()
        u = t.replace(WD, ".").replace(os.path.expanduser("~"), "~")
        if u != t: open(p, "w", encoding="utf-8", errors="surrogateescape").write(u)

# ---- checksums ----
with open(J("SHA256SUMS"), "w") as fh:
    for root, _, files in sorted(os.walk(OUT)):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f == "SHA256SUMS": continue
            fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, OUT)}\n")
print(N)
