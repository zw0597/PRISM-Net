"""R5: class-conditional coverage of split-conformal sets (marginal vs Mondrian), seed-0 film-grouped predictions."""
import json, math, numpy as np, torch, torch.nn.functional as F
D = np.load("./riawelc_80.npz"); Y = D["y"]; K = 4
MODELS = [("prismr+paug", "PRISM-Net", "edl"), ("prismx+paug", "PRISM-Net-Lite", "edl"), ("resnet18", "ResNet-18", "ce"),
          ("vgg11_bn", "VGG-11", "ce"), ("mobilenet_v3_small", "MobileNetV3-S", "ce"), ("shufflenet_v2_x1_0", "ShuffleNetV2", "ce"),
          ("efficientnet_b0", "EfficientNet-B0", "ce"), ("vit", "ViT", "ce"), ("gfnet", "GFNet", "ce"),
          ("convnextv2_atto", "ConvNeXt V2-Atto", "ce"), ("repvit_m0_9", "RepViT-M0.9", "ce"),
          ("mobilenetv4_conv_small", "MobileNetV4-S", "ce"), ("resnet18+paug", "ResNet-18 + PAug", "ce"),
          ("mobilenet_v3_small+paug", "MobileNetV3-S + PAug", "ce"), ("gfnet+paug", "GFNet + PAug", "ce")]
def probs(Z, lt):
    Z = torch.from_numpy(Z).double()
    if lt == "edl":
        a = F.softplus(Z) + 1.0; return (a / a.sum(1, keepdim=True)).numpy()
    return F.softmax(Z, 1).numpy()
def qhat(s, eps):
    n = len(s); return np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - eps)) / n), method="higher")
out = {}
for tag, name, lt in MODELS:
    R = {}
    for eps in (0.10, 0.05):
        acc = {"marg": [], "mond": []}
        for f in range(5):
            Z = np.load(f"results/{tag}_group_f{f}_s0.npz"); J = json.load(open(f"results/{tag}_group_f{f}_s0.json"))
            assert J["loss_type"] == lt, (tag, J["loss_type"])
            pv, pt = probs(Z["Zva"], lt), probs(Z["Zte_clean_0"], lt); yv, yt = Y[Z["va_idx"]], Y[Z["te_idx"]]
            sv = 1 - pv[np.arange(len(yv)), yv]
            S_m = pt >= 1 - qhat(sv, eps)
            qk = np.array([qhat(sv[yv == k], eps) for k in range(K)])
            S_c = (1 - pt) <= qk[None, :]
            for key, S in (("marg", S_m), ("mond", S_c)):
                hit = S[np.arange(len(yt)), yt]
                acc[key].append(dict(cov=float(hit.mean()), size=float(S.sum(1).mean()), single=float((S.sum(1) == 1).mean()),
                                     cov_cls=[float(hit[yt == k].mean()) for k in range(K)],
                                     size_cls=[float(S[yt == k].sum(1).mean()) for k in range(K)]))
        R[str(eps)] = {key: dict(cov=float(np.mean([a["cov"] for a in v])), size=float(np.mean([a["size"] for a in v])),
                                 single=float(np.mean([a["single"] for a in v])),
                                 cov_cls=np.mean([a["cov_cls"] for a in v], 0).tolist(),
                                 cov_cls_minfold=np.min([a["cov_cls"] for a in v], 0).tolist(),
                                 size_cls=np.mean([a["size_cls"] for a in v], 0).tolist()) for key, v in acc.items()}
    out[name] = R
json.dump(out, open("cp_class.json", "w"), indent=1)
for name, R in out.items():
    for eps in ("0.1",):
        m, c = R[eps]["marg"], R[eps]["mond"]
        print(f"{name:22s} marg cov {m['cov']*100:5.1f} cls {[round(x*100,1) for x in m['cov_cls']]} size {m['size']:.2f} single {m['single']*100:.1f} | "
              f"mond cov {c['cov']*100:5.1f} cls {[round(x*100,1) for x in c['cov_cls']]} size {c['size']:.2f} single {c['single']*100:.1f}")
