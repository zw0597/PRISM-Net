# Holm correction of the 14 film-level Wilcoxon p-values of tab_pair (PRISM-Net vs each baseline; from summary.json written by analyze.py).
# Output holm_pair.json: {tag: [raw p, Holm-adjusted p]}. Used by make_v45_tex.py.
import json
BASE = ["prismx+paug", "resnet18", "vgg11_bn", "mobilenet_v3_small", "shufflenet_v2_x1_0", "efficientnet_b0", "vit", "gfnet",
        "convnextv2_atto", "repvit_m0_9", "mobilenetv4_conv_small", "resnet18+paug", "mobilenet_v3_small+paug", "gfnet+paug"]
A = json.load(open("summary.json"))["agg"]
p = {m: A[m]["wilcoxon_p_vs_ref"] for m in BASE}
order = sorted(BASE, key=lambda m: p[m]); k = len(order); adj = {}; run = 0.0
for i, m in enumerate(order):
    run = max(run, min(1.0, (k - i) * p[m])); adj[m] = run
json.dump({m: [p[m], adj[m]] for m in BASE}, open("holm_pair.json", "w"), indent=1)
