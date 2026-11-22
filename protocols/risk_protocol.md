# Additional experiments requested to address reviewer risks — written 2026-11-12 before any of them ran

E1  Stronger training regime (ImageNet initialisation, higher resolution).
    Models: ResNet-18, ResNet-18 + PAug, PRISM-Net, all initialised from the same ImageNet-1k ResNet-18 weights
    (timm RSB-A1 release; first convolution summed over RGB for the 1-channel PRISM-Net trunk; RASG branches as in
    PRISM-Net, zero-initialised gates, layer scale 1e-2). Input 112 x 112. 1000 AdamW steps, batch 64, peak lr 5e-4,
    wd 0.05, 5% warm-up, cosine; same augmentation, losses and evaluation as the main experiments; seed 0; 5 folds.
    The settings are fixed here and not tuned.
E2  Recent efficient backbones trained exactly as the main baselines (80 x 80, from scratch, 1000 steps, lr 2e-3):
    ConvNeXt V2-Atto (2023), RepViT-M0.9 (2024), MobileNetV4-Conv-S (2024), timm implementations, 1-channel input.
    Also evaluated, frozen, on the SWRD external test under swrd_protocol.md.
All results are reported whatever their direction.
Amendment (before any run): E1 evaluates clean test tiles only (no corruption sweep) to save compute.
