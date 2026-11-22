"""PRISM-Net and baselines.

PRISM-Net = Physics-guided Radial-angular Isotropy-aware Spectral Modulation Network
  * BLAN : Beer-Lambert attenuation normalisation (exposure/gamma invariant stem)
  * RASG : radial-angular spectral gating (instance-adaptive polar Fourier filter)
  * EDL  : Dirichlet (evidential) output head
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# --------------------------------------------------------------------------- BLAN
class BLAN(nn.Module):
    """Beer-Lambert attenuation normalisation.

    mode="shift" (default, contrast preserving):
        a = -ln(max(I, eps));  z = (a - med(a)) / tau,  tau = training-set median robust scale.
        Exactly invariant to exposure gain I' = c*I (no clipping); keeps attenuation contrast.
    mode="mad" (ablation, full robust standardisation):
        z = (a - med(a)) / (1.4826*MAD(a) + delta); also invariant to I' = c*I^gamma, but
        removes the contrast cue that separates defect-free from defective tiles.
    """

    def __init__(self, mode="shift", eps=1.0 / 255.0, delta=1e-3, clip=6.0, tau=0.106):
        super().__init__()
        self.mode, self.eps, self.delta, self.clip = mode, eps, delta, clip
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x):                       # x in [0,1], B x 1 x H x W
        a = -torch.log(x.clamp_min(self.eps))
        flat = a.flatten(1)
        med = flat.median(dim=1).values
        if self.mode == "mad":
            mad = (flat - med[:, None]).abs().median(dim=1).values
            den = (1.4826 * mad + self.delta)[:, None, None, None]
        else:
            den = self.tau
        z = (a - med[:, None, None, None]) / den
        return z.clamp(-self.clip, self.clip)


class DualNorm(nn.Module):
    """Two-channel stem input: dataset-standardised intensity and BLAN attenuation."""

    def __init__(self):
        super().__init__()
        self.g, self.b = GlobalNorm(), BLAN("shift")
        self.tau = self.b.tau

    def forward(self, x):
        return torch.cat([self.g(x), self.b(x)], 1)


class GlobalNorm(nn.Module):
    """Dataset-level standardisation used when BLAN is ablated."""

    def __init__(self, mean=0.5, std=0.25):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- RASG
def polar_basis(H, W, R=6, A=8, kappa=2.0):
    """Separable radial (Gaussian) x angular (axial von Mises) basis on the rFFT grid.

    Returns tensor of shape (R*A, H, W//2+1).  The angular factor depends on 2*theta,
    hence psi(theta + pi) = psi(theta): the gate is centro-symmetric and the
    filtered signal stays real-valued.
    """
    fy = torch.fft.fftfreq(H)
    fx = torch.fft.rfftfreq(W)
    v, u = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(u ** 2 + v ** 2) / (0.5 * math.sqrt(2.0))       # in [0, 1]
    theta = torch.atan2(v, u)
    mu = torch.linspace(0.0, 1.0, R)
    sig = 1.0 / (R - 1)
    phi = torch.exp(-(rho[None] - mu[:, None, None]) ** 2 / (2 * sig ** 2))          # R,H,Wr
    th_a = torch.arange(A) * math.pi / A
    psi = torch.exp(kappa * torch.cos(2 * (theta[None] - th_a[:, None, None])))      # A,H,Wr
    psi = psi / psi.sum(0, keepdim=True)                       # partition of unity over a
    psi[:, rho == 0] = 1.0 / A                                 # DC has no orientation
    B = (phi[:, None] * psi[None]).reshape(R * A, H, W // 2 + 1)
    return B


class RASG(nn.Module):
    """Y = F^{-1}[ F[X] * (1 + tanh(sum_k w_{b,g,k} B_k)) ], w = w0 + h(GAP(X))."""

    def __init__(self, C, H, W, R=6, A=8, groups=8, kappa=2.0, dynamic=True, angular=True, gate="tanh"):
        super().__init__()
        self.C, self.G, self.R, self.A = C, groups, R, (A if angular else 1)
        if angular:
            B = polar_basis(H, W, R, A, kappa)
        else:  # isotropic ablation: radial bumps only
            B = polar_basis(H, W, R, 1, kappa)
        self.register_buffer("basis", B, persistent=False)
        K = B.shape[0]
        self.w0 = nn.Parameter(torch.zeros(groups, K))
        self.dynamic = dynamic
        if dynamic:
            hid = max(C // 4, 16)
            self.h = nn.Sequential(nn.Linear(C, hid), nn.GELU(), nn.Linear(hid, groups * K))
            nn.init.zeros_(self.h[2].weight); nn.init.zeros_(self.h[2].bias)
        self.last_w = None
        self.gate_fn = gate

    def forward(self, x):
        Bsz, C, H, W = x.shape
        Fx = torch.fft.rfft2(x.float(), norm="ortho")
        w = self.w0.unsqueeze(0).expand(Bsz, -1, -1)
        if self.dynamic:
            w = w + self.h(x.mean((2, 3))).view(Bsz, self.G, -1)
        self.last_w = w.detach()
        g = torch.einsum("bgk,khw->bghw", w, self.basis)
        gate = 1.0 + (torch.tanh(g) if self.gate_fn == "tanh" else g)       # (0, 2) for tanh
        gate = gate.repeat_interleave(C // self.G, dim=1)
        y = torch.fft.irfft2(Fx * gate, s=(H, W), norm="ortho")
        return y.to(x.dtype)


class PRISMBlock(nn.Module):
    def __init__(self, C, H, W, use_rasg=True, rasg_kw=None, mlp_ratio=3, ls=1e-2, gw=1, k=3):
        super().__init__()
        self.dw = nn.Conv2d(C, C, k, padding=k // 2, groups=C // gw, bias=False)   # gw channels per group
        self.rasg = RASG(C, H, W, **(rasg_kw or {})) if use_rasg else None
        self.n1 = nn.BatchNorm2d(C)
        self.proj = nn.Conv2d(C, C, 1)
        self.n2 = nn.BatchNorm2d(C)
        self.mlp = nn.Sequential(nn.Conv2d(C, C * mlp_ratio, 1), nn.GELU(),
                                 nn.Conv2d(C * mlp_ratio, C, 1))
        self.g1 = nn.Parameter(ls * torch.ones(C, 1, 1))
        self.g2 = nn.Parameter(ls * torch.ones(C, 1, 1))

    def forward(self, x):
        m = self.dw(x)
        if self.rasg is not None:
            m = m + self.rasg(x)
        x = x + self.g1 * self.proj(F.gelu(self.n1(m)))
        x = x + self.g2 * self.mlp(self.n2(x))
        return x


class PRISMNet(nn.Module):
    def __init__(self, num_classes=4, img=112, dims=(32, 64, 128, 192), depths=(2, 2, 4, 2),
                 blan=True, rasg_stages=(1, 2, 3), rasg_kw=None, evidential=True, gw=1, k=3, stem2=False):
        super().__init__()
        if blan == "dual":
            self.norm = DualNorm()
        else:
            self.norm = (BLAN(blan if isinstance(blan, str) else "shift") if blan else GlobalNorm())
        cin = 2 if blan == "dual" else 1
        self.evidential = evidential
        self.stem = nn.Sequential(nn.Conv2d(cin, dims[0] // 2, 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0] // 2),
                                  nn.GELU(), nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1, bias=False),
                                  nn.BatchNorm2d(dims[0]))
        if stem2:   # richer stem: dense 3x3 conv at half resolution before the second downsampling
            self.stem = nn.Sequential(nn.Conv2d(cin, dims[0], 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0]), nn.GELU(),
                                      nn.Conv2d(dims[0], dims[0], 3, 1, 1, bias=False), nn.BatchNorm2d(dims[0]), nn.GELU(),
                                      nn.Conv2d(dims[0], dims[0], 3, 2, 1, bias=False), nn.BatchNorm2d(dims[0]))
        s = img // 4
        stages = []
        for i, (d, n) in enumerate(zip(dims, depths)):
            layers = []
            if i > 0:
                layers += [nn.BatchNorm2d(dims[i - 1]), nn.Conv2d(dims[i - 1], d, 3, 2, 1)]
                s = (s + 1) // 2
            for _ in range(n):
                layers.append(PRISMBlock(d, s, s, use_rasg=(i in rasg_stages), rasg_kw=rasg_kw, gw=gw, k=k))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.head_norm = nn.BatchNorm2d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def features(self, x):
        return self.stages(self.stem(self.norm(x)))

    def forward(self, x):
        f = self.head_norm(self.features(x)).mean((2, 3))
        return self.head(f)


# --------------------------------------------------------------------------- baselines
class TVWrap(nn.Module):
    """torchvision backbone, grey input replicated to 3 channels, trained from scratch."""

    def __init__(self, name, num_classes=4):
        super().__init__()
        self.norm = GlobalNorm()
        ctor = getattr(tvm, name)
        self.net = ctor(weights=None, num_classes=num_classes)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x.expand(-1, 3, -1, -1))


class SimpleViT(nn.Module):
    """Small plain ViT (patch 8) as a transformer reference."""

    def __init__(self, num_classes=4, img=112, patch=8, dim=192, depth=6, heads=3):
        super().__init__()
        self.norm = GlobalNorm()
        self.pe = nn.Conv2d(1, dim, patch, patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim)); nn.init.trunc_normal_(self.pos, std=.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 3, 0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        t = self.pe(self.norm(x)).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1) + self.pos
        return self.head(self.ln(self.enc(t)[:, 0]))


class GFNetTiny(nn.Module):
    """Global-filter (GFNet-style, fixed learnable complex filter) reference sharing PRISM's
    macro-architecture: replaces RASG by a free per-channel complex spectral weight."""

    def __init__(self, num_classes=4, img=112):
        super().__init__()
        self.body = PRISMNet(num_classes, img, blan=False, rasg_stages=(), evidential=False)
        s, filt = img // 4, []
        dims = (32, 64, 128, 192)
        for i, d in enumerate(dims):
            if i > 0:
                s = (s + 1) // 2
            filt.append(nn.Parameter(torch.zeros(d, s, s // 2 + 1, 2)) if i > 0 else None)
        self.filters = nn.ParameterList([f for f in filt if f is not None])
        self.idx = {1: 0, 2: 1, 3: 2}
        self.dims = dims

    def forward(self, x):
        b = self.body
        h = b.stem(b.norm(x))
        for i, st in enumerate(b.stages):
            for j, layer in enumerate(st):
                if isinstance(layer, PRISMBlock) and i in self.idx:
                    f = self.filters[self.idx[i]]
                    Fx = torch.fft.rfft2(h.float(), norm="ortho")
                    wc = torch.view_as_complex(f.contiguous()) + 1.0
                    m = layer.dw(h) + torch.fft.irfft2(Fx * wc, s=h.shape[-2:], norm="ortho")
                    h = h + layer.g1 * layer.proj(F.gelu(layer.n1(m)))
                    h = h + layer.g2 * layer.mlp(layer.n2(h))
                else:
                    h = layer(h)
        return b.head(b.head_norm(h).mean((2, 3)))


class RASGResNet(nn.Module):
    """ResNet-18 trunk (1-channel stem) with a layer-scaled RASG global branch after every
    BasicBlock of layers 2-4: x <- x + gamma * BN(RASG(x)). RASG starts as the identity filter."""

    def __init__(self, num_classes=4, img=80, rasg_kw=None, ls=1e-2, use_rasg=True):
        super().__init__()
        self.norm = GlobalNorm()
        r = tvm.resnet18(weights=None, num_classes=num_classes)
        r.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.r = r
        s = (img + 1) // 2; s = (s + 1) // 2                 # after conv1 and maxpool
        self.glob = nn.ModuleList()
        for li, layer in enumerate([r.layer1, r.layer2, r.layer3, r.layer4]):
            if li > 0:
                s = (s + 1) // 2
            for blk in layer:
                C = blk.conv2.out_channels
                if li == 0 or not use_rasg:
                    self.glob.append(None)
                else:
                    self.glob.append(nn.ModuleDict(dict(rasg=RASG(C, s, s, **(rasg_kw or {})), bn=nn.BatchNorm2d(C))))
        self.gam = nn.ParameterList([nn.Parameter(ls * torch.ones(m["bn"].num_features, 1, 1)) if m is not None
                                     else nn.Parameter(torch.zeros(1)) for m in self.glob])

    def forward(self, x):
        r = self.r
        x = r.maxpool(r.relu(r.bn1(r.conv1(self.norm(x)))))
        k = 0
        for layer in [r.layer1, r.layer2, r.layer3, r.layer4]:
            for blk in layer:
                x = blk(x)
                g = self.glob[k]
                if g is not None:
                    x = x + self.gam[k] * g["bn"](g["rasg"](x))
                k += 1
        return r.fc(torch.flatten(r.avgpool(x), 1))


def build(name, img=112):
    """Model zoo. Suffix '+paug' (handled in train.py) only switches on augmentation."""
    name = name.split("+")[0]
    table = {
        "prism": dict(),
        "prism_ce": dict(),                                  # CE loss instead of EDL (train.py)
        "prism_mad": dict(blan="mad"),                       # full robust standardisation
        "prism_noblan": dict(blan=False),
        "prism_norasg": dict(rasg_stages=()),
        "prism_static": dict(rasg_kw=dict(dynamic=False)),
        "prism_iso": dict(rasg_kw=dict(angular=False)),
        "prism_base": dict(blan=False, rasg_stages=()),
        "prism_lin_ce": dict(rasg_kw=dict(gate="lin")),       # dev: unbounded linear gate
        "prism_noblan_ce": dict(blan=False),
        "prism_w_ce": dict(dims=(48, 96, 192, 288)),          # dev: wider
        "prism_w": dict(dims=(48, 96, 192, 288)),
        "prism_g16": dict(gw=16),
        "prism_dual": dict(blan="dual"),
        # final PRISM-Net (selected on validation films): RASG + EDL, global normalisation, trained with PAug
        "prismx": dict(blan=False),
        "prismx_ce": dict(blan=False),
        "prismx_norasg": dict(blan=False, rasg_stages=()),
        "prismx_iso": dict(blan=False, rasg_kw=dict(angular=False)),
        "prismx_static": dict(blan=False, rasg_kw=dict(dynamic=False)),
        # round-2 design candidates (selected on validation films only)
        "prismx_k7": dict(blan=False, k=7),
        "prismx_stem2": dict(blan=False, stem2=True),
        "prismx_l05": dict(blan=False),
        "prismx_w": dict(blan=False, dims=(48, 96, 192, 288)),                      # dev: intensity + attenuation channels                             # dev: grouped (16 ch/group) local mixer
    }
    if name in table:
        return PRISMNet(img=img, **table[name])
    if name == "gfnet":
        return GFNetTiny(img=img)
    if name in ("prismr", "prismr_ce"):
        return RASGResNet(img=img)
    if name == "prismr_norasg":
        return RASGResNet(img=img, use_rasg=False)
    if name == "prismr_iso":
        return RASGResNet(img=img, rasg_kw=dict(angular=False))
    if name == "prismr_static":
        return RASGResNet(img=img, rasg_kw=dict(dynamic=False))
    if name == "vit":
        return SimpleViT(img=img)
    if name == "prismrpt":                                      # ImageNet-initialised trunk, RASG branches as usual
        m = RASGResNet(img=img); load_imagenet_r18(m.r, 1); return m
    if name == "resnet18pt":
        m = TVWrap("resnet18"); load_imagenet_r18(m.net, 3); return m
    if name in ("convnextv2_atto", "mobilenetv4_conv_small", "repvit_m0_9"):
        return TimmWrap(name)
    if name == "vgg11_bn":
        return VGGWrap()
    return TVWrap(name)


class VGGWrap(nn.Module):
    """VGG-11 (BN) convolutional part + global average pooling + linear layer; grey input replicated to 3 channels."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.norm = GlobalNorm()
        self.features = tvm.vgg11_bn(weights=None).features
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(self.norm(x).expand(-1, 3, -1, -1))
        return self.fc(x.mean((2, 3)))


class TimmWrap(nn.Module):
    """Recent timm backbone (1-channel input), trained from scratch with the same standardisation."""

    def __init__(self, name, num_classes=4):
        super().__init__()
        import timm
        self.norm = GlobalNorm()
        self.net = timm.create_model(name, pretrained=False, num_classes=num_classes, in_chans=1)

    def forward(self, x):
        return self.net(self.norm(x))


IMAGENET_R18 = "./pre/r18.pth"   # timm RSB-A1 ResNet-18 ImageNet-1k weights (GitHub release)


def load_imagenet_r18(resnet, in_ch):
    sd = torch.load(IMAGENET_R18, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    if in_ch == 1:
        sd["conv1.weight"] = sd["conv1.weight"].sum(1, keepdim=True)
    missing, unexpected = resnet.load_state_dict(sd, strict=False)
    assert all(k.startswith("fc.") for k in missing) and not unexpected, (missing, unexpected)
    return resnet


BASELINES = ["resnet18", "mobilenet_v2", "mobilenet_v3_small", "shufflenet_v2_x1_0",
             "efficientnet_b0", "regnet_y_400mf", "convnext_tiny"]
