"""Audit numbers -> paper/numbers_audit.tex, tab_audit.tex, tab_folds.tex (no training results needed)."""
import json
import numpy as np

P = "../latex/"
S = json.load(open("./data_stats.json"))
L = json.load(open("./leak_stats.json"))
D = np.load("./riawelc_80.npz")
Y = D["y"]; Xu = D["X"]; mad = np.zeros(len(Y), np.float32)
for s0 in range(0, len(Y), 2000):
    a = -np.log(np.clip(Xu[s0:s0 + 2000].astype(np.float32) / 255, 1 / 255, 1)).reshape(-1, Xu.shape[1] * Xu.shape[2])
    med = np.median(a, 1); mad[s0:s0 + 2000] = 1.4826 * np.median(np.abs(a - med[:, None]), 1)
md = {c: float(np.median(mad[Y == c])) for c in range(4)}


def fmt(n):
    return f"{n:,}".replace(",", "\\,")


macros = {
    "AuditN": fmt(S["N"]), "AuditUnique": fmt(S["unique"]), "AuditDupGroups": fmt(S["dup_groups"]),
    "AuditDupImages": fmt(S["dup_images"]), "AuditDupGroupsCross": fmt(S["dup_groups_cross_split"]),
    "AuditFilms": str(S["films"]), "AuditRadiographs": str(S["radiographs"]),
    "AuditTestN": fmt(L["leak"]["test_n"]), "FoldObj": "%.4f" % 0.028435332857914008,
    "MadND": "%.3f" % md[0], "MadPO": "%.3f" % md[1], "MadLP": "%.3f" % md[2], "MadCR": "%.3f" % md[3],
}
with open(P + "numbers_audit.tex", "w") as f:
    for k, v in macros.items():
        f.write("\\newcommand{\\%s}{%s}\n" % (k, v))

lk = L["leak"]
rows = [
    ("Tiles in the release", fmt(S["N"])),
    ("Distinct pixel arrays", fmt(S["unique"])),
    ("Duplicate groups / tiles involved", f"{fmt(S['dup_groups'])} / {fmt(S['dup_images'])}"),
    ("Duplicate groups spanning official splits", fmt(S["dup_groups_cross_split"])),
    ("Duplicate groups with conflicting labels", str(S["dup_groups_cross_label"])),
    ("Films / digitized radiographs", f"{S['films']} / {S['radiographs']}"),
    ("Official test tiles with identical training tile", f"{fmt(lk['test_exact_dup_in_train'])} / {fmt(lk['test_n'])}"),
    ("Official test tiles whose film is in training", f"{fmt(lk['test_film_seen_in_train'])} / {fmt(lk['test_n'])}"),
    ("Official val.\\ tiles with identical training tile", f"{fmt(lk['val_exact_dup_in_train'])} / {fmt(lk['val_n'])}"),
    ("Official val.\\ tiles whose film is in training", f"{fmt(lk['val_film_seen_in_train'])} / {fmt(lk['val_n'])}"),
]
with open(P + "tab_audit.tex", "w") as f:
    f.write("\\footnotesize\n\\begin{tabular}{@{}lr@{}}\n\\toprule\nQuantity & Value\\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r[0]} & {r[1]}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

fc = np.array(L["fold_class_counts"])
fo = L["fold_of_film"]
with open(P + "tab_folds.tex", "w") as f:
    f.write("\\footnotesize\n\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{@{}clrrrrr@{}}\n\\toprule\n")
    f.write("Fold & Films & ND & PO & LP & CR & Total\\\\\n\\midrule\n")
    for k in range(5):
        films = sorted([p for p in fo if fo[p] == k], key=lambda s: (s.startswith("bam"), int(''.join(ch for ch in s if ch.isdigit()) or 0)))
        nm = ", ".join(p.replace("RRT-", "").replace("R", "") if p.startswith("RRT") else p for p in films)
        f.write(f"{k+1} & \\parbox[t]{{2.9cm}}{{\\raggedright {nm}}} & {fc[k,0]} & {fc[k,1]} & {fc[k,2]} & {fc[k,3]} & {fc[k].sum()}\\\\\n")
    f.write("\\midrule\nAll & %d films & %d & %d & %d & %d & %d\\\\\n" % ((len(fo),) + tuple(fc.sum(0)) + (fc.sum(),)))
    f.write("\\bottomrule\n\\end{tabular}\n\\\\[2pt]\\parbox{\\columnwidth}{\\scriptsize Film identifiers are the numbers of the RRT films (e.g., 13 = RRT-13R); bam5 is the remaining film.}\n")
print(macros)
