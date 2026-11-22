"""Params, FLOPs (incl. analytic FFT cost) and single-image CPU latency. Run on an idle machine."""
import json, math, sys, time
import torch
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, ".")
from models import build, RASG

models = sys.argv[1].split(",")
torch.set_num_threads(1)
out = {}
for m in models:
    net = build(m, 80).eval()
    x = torch.rand(1, 1, 80, 80)
    fftflops = 0

    def hook(mod, inp, outp):
        global fftflops
        B, C, H, W = inp[0].shape
        n = H * W
        fftflops += 2 * int(C * 2.5 * n * math.log2(n))      # rFFT + irFFT, ~5/2 N log2 N real flops each
        fftflops += C * H * (W // 2 + 1) * 6 + mod.basis.shape[0] * mod.G * H * (W // 2 + 1) * 2
    hs = []
    for mod in net.modules():
        if isinstance(mod, RASG):
            hs.append(mod.register_forward_hook(hook))
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        net(x)
    for h in hs:
        h.remove()
    flops = fc.get_total_flops() + (fftflops if m.startswith("prism") else 0)
    if m == "gfnet":  # GFNet global filters: analytic FFT + complex multiply cost, stages 2-4
        for C, H, d in ((64, 10, 2), (128, 5, 4), (192, 3, 2)):
            n = H * H
            flops += d * (2 * int(C * 2.5 * n * math.log2(n)) + C * H * (H // 2 + 1) * 6)
    with torch.no_grad():
        for _ in range(20):
            net(x)
        ts = []
        for _ in range(200):
            t = time.perf_counter(); net(x); ts.append(time.perf_counter() - t)
    ts.sort()
    out[m] = dict(params=sum(p.numel() for p in net.parameters()), gflops=flops / 1e9,
                  lat_ms=1000 * ts[len(ts) // 2], lat_p90=1000 * ts[int(0.9 * len(ts))])
    print(m, out[m], flush=True)
E = json.load(open("./efficiency.json")); E.update(out); json.dump(E, open("./efficiency.json", "w"), indent=1)
