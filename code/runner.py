"""Job queue: each worker claims the next unfinished job (atomic lock file) and runs it.
Re-launching the runner resumes where it stopped; stale locks of dead jobs are retried."""
import os, subprocess, sys, time

OUT = "./results"
os.makedirs(OUT, exist_ok=True)
BASE = ["resnet18", "mobilenet_v3_small", "shufflenet_v2_x1_0", "gfnet", "efficientnet_b0", "vit"]
FINAL = ["prismr+paug"]
PAUGB = ["resnet18+paug", "mobilenet_v3_small+paug", "gfnet+paug", "prismx+paug"]
ABL = ["prismr", "prismr_ce+paug", "prismr_norasg+paug", "prismr_iso+paug", "prism+paug"]

jobs = []
for group in (BASE + FINAL, ABL, PAUGB):
    for f in range(5):
        for m in group:
            jobs.append((m, "group", f))
for m in FINAL + BASE:
    jobs.append((m, "official", 0))
jobs.sort(key=lambda j: (j[1] == "official", 0))
if len(sys.argv) > 2:
    jobs = [j for j in jobs if j[0] in sys.argv[2].split(",")]


def tag(j):
    return f"{j[0]}_{j[1]}_f{j[2]}_s0"


wid = sys.argv[1] if len(sys.argv) > 1 else "0"
T0 = time.time()
for j in jobs:
    if time.time() - T0 > 75 * 60:      # stop claiming new jobs before the 2-h background limit
        break
    t = tag(j)
    if os.path.exists(os.path.join(OUT, t + ".json")):
        continue
    lock = os.path.join(OUT, t + ".lock")
    try:
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) > 3 * 3600:
            os.remove(lock)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL); os.close(fd)
    except FileExistsError:
        continue
    cmd = ["python3", "./train.py", "--model", j[0], "--protocol", j[1],
           "--fold", str(j[2]), "--res", "80", "--bs", "64", "--iters", "1000",
           "--threads", "1", "--out", OUT]
    with open(os.path.join(OUT, t + ".log"), "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=".")
    os.remove(lock)
    print(t, "rc", rc, flush=True)
