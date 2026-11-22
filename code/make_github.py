"""Split the full release (make_release.py output) into a GitHub repository and a predictions archive for a GitHub Release.
Usage: python make_github.py <release_dir> <out_dir>   -> <out_dir>/PRISM-Net/ (repository) and <out_dir>/PRISM-Net-predictions-v1.0.zip"""
import os, sys, shutil, hashlib, zipfile
SRC, OUT = sys.argv[1], sys.argv[2]
REPO = os.path.join(OUT, "PRISM-Net"); ZIP = os.path.join(OUT, "PRISM-Net-predictions-v1.0.zip")
if os.path.exists(REPO): shutil.rmtree(REPO)
os.makedirs(REPO)
for d in ("code", "splits", "protocols"): shutil.copytree(os.path.join(SRC, d), os.path.join(REPO, d))
shutil.copy(os.path.join(SRC, "LICENSE"), REPO)
r = open(os.path.join(SRC, "README.md")).read()
r = r.replace("Companion data set of the article", "Companion repository of the article")
r = r.replace("| `predictions/` | Logits of every run reported in the article (see below). |",
              "| `predictions/` | Not in the repository: download `PRISM-Net-predictions-v1.0.zip` from the Releases page and unzip it here (see below). |")
r = r.replace("### Predictions\n", "### Predictions\n\nThe predictions (about 380 MB) are attached to release v1.0 of this repository as `PRISM-Net-predictions-v1.0.zip`; "
              "unzipping it in the repository root creates `predictions/` with its own `SHA256SUMS`.\n\n")
open(os.path.join(REPO, "README.md"), "w").write(r)
open(os.path.join(REPO, ".gitignore"), "w").write("predictions/\nresults*/\ndata/\n*.pt\n*.ckpt\n*.partial\n__pycache__/\n.DS_Store\n")
def sums(root, out):
    with open(out, "w") as fh:
        for dp, _, fs in sorted(os.walk(root)):
            for f in sorted(fs):
                p = os.path.join(dp, f)
                if os.path.abspath(p) == os.path.abspath(out) or ".git" in p: continue
                fh.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, root)}\n")
sums(REPO, os.path.join(REPO, "SHA256SUMS"))
tmp = os.path.join(OUT, "_pred"); shutil.rmtree(tmp, ignore_errors=True); shutil.copytree(os.path.join(SRC, "predictions"), os.path.join(tmp, "predictions"))
sums(os.path.join(tmp, "predictions"), os.path.join(tmp, "predictions", "SHA256SUMS"))
if os.path.exists(ZIP): os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in os.walk(tmp):
        for f in fs:
            p = os.path.join(dp, f); z.write(p, os.path.relpath(p, tmp))
shutil.rmtree(tmp)
h = hashlib.sha256(open(ZIP, "rb").read()).hexdigest(); open(ZIP + ".sha256", "w").write(f"{h}  {os.path.basename(ZIP)}\n")
nfiles = sum(len(fs) for _, _, fs in os.walk(REPO))
print("repo files", nfiles, "repo MB", round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(REPO) for f in fs) / 1e6, 1),
      "zip MB", round(os.path.getsize(ZIP) / 1e6, 1), "sha256", h[:16])
