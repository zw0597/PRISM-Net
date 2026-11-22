# PRISM-Net: code, leakage-free splits, duplicate lists and predictions

Companion data set of the article

> W. Zeng, X. Liu, Z. Chen, J. Li, Y. Chen, J. Lan, S. Du and Z. Tian, "Physics-Informed, Uncertainty-Aware Weld Defect
> Recognition in Radiographic Testing: Leakage-Free Evaluation and Commissioning on New X-Ray Systems,"
> submitted to *IEEE Transactions on Instrumentation and Measurement*, 2026.

It contains everything needed to check the RIAWELC leakage audit, to reuse the film-grouped protocol, and to recompute
every number in the article from the stored predictions. No radiographic images and no model weights are included:
RIAWELC and SWRD must be obtained from their authors.

## Contents

| Folder | Content |
|---|---|
| `splits/riawelc_manifest.csv` | One row per RIAWELC tile (24,407): relative path in the original release, official split, class, film, radiograph view, MD5 of the decoded pixel array, duplicate group, and its role (train / val / test / removed) in each of the five film-grouped folds. |
| `splits/riawelc_duplicate_groups.csv` | The 2,498 groups of pixel-identical tiles (5,012 tiles); 2,465 groups span more than one official split. Every one of the 2,443 official test tiles has a copy in the training split. |
| `splits/splits.json` | Tile indices (rows of the manifest) of train / val / test for folds 0-4, and the test films of each fold. Duplicates are removed before splitting. |
| `splits/leak_stats.json` | Audit statistics and the film-to-fold assignment. |
| `splits/swrd_split.json` | Radiograph-level adaptation / validation / test split of SWRD used for the external test. |
| `protocols/` | Analysis protocols, each written before the corresponding results existed (time-stamped entries). |
| `code/` | Training, evaluation, analysis and table/figure scripts exactly as used for the article. |
| `predictions/` | Logits of every run reported in the article (see below). |
| `SHA256SUMS` | Checksums of all files. |

### Predictions

| Folder | Runs |
|---|---|
| `riawelc_cpu_1000steps/` | Main experiments (Table III and everything derived from it): 15 models x 5 folds, seed 0; seeds 1-2 for PRISM-Net, ResNet-18 and VGG-11; ablations; official-split runs; single-fold runs of the first design round (disclosed in the article). 1000 AdamW steps, CPU. |
| `riawelc_imagenet_112px/` | ImageNet-initialized ResNet-18, ResNet-18 + PAug and PRISM-Net at 112 x 112 px. |
| `riawelc_gpu_1000steps/`, `riawelc_gpu_3000steps/` | Longer-budget experiments (Section V-E), Apple M5 Max (MPS), with training logs. |
| `riawelc_gpu_1000steps_r8_*/` | Training variants that use the acquisition model: orbit consistency (`oc`), canonical acquisition (`canon`) and both (`occanon`). |
| `swrd/` | Frozen-model, AdaBN, PGA and standardization predictions on SWRD, PGA parameters, recipe selection, fine-tuning predictions (`finetune/`), content-aware alignment, uncertainty propagation and the final label-free chain (`swrd_probs_r8.npz`, `swrd_probs_r9.npz`). |

Each `<model>_<protocol>_f<fold>_s<seed>.npz` holds `te_idx` / `va_idx` (rows of the manifest), `Zva` (validation
logits, used for conformal calibration) and `Zte_<corruption>_<level>` (test logits; `Zte_clean_0` is the clean test set,
the other 15 keys the corruption levels of Section V-K). The matching `.json` stores the arguments, loss type
(`ce`: softmax; `edl`: Dirichlet with alpha = softplus(z) + 1), parameter count, per-epoch training log and test accuracy.
Class order: 0 = ND (no defect, `NoDifetto`), 1 = PO (porosity, `Difetto2`), 2 = LP (lack of penetration, `Difetto4`),
3 = CR (crack, `Difetto1`).

## Reproducing

1. Python 3.11, PyTorch >= 2.2, torchvision, timm, numpy, scipy, scikit-learn, matplotlib, Pillow.
2. Place the RIAWELC release under `data/raw/DB - Copy/{training,validation,testing}/<class folder>/` and run
   `python code/prep.py 80` (builds `riawelc_80.npz`; its `md5` and `fname` arrays must match the manifest).
3. Train one run, e.g. PRISM-Net on fold 0:
   `python code/train.py --model prismr+paug --protocol group --fold 0 --res 80 --lr 2e-3 --bs 64 --iters 1000 --seed 0`
   (`--iters 3000 --device auto` for the longer budget). The job lists of the article are in `code/runner.py`, `code/vggrun*.sh`,
   `code/seedrun.sh`, `code/vggseed.sh`, `code/run_r4_mps.sh` and `code/run_r6_mps.sh`.
4. `python code/analyze.py` recomputes the main metrics, film-level tests and conformal sets from `results/`;
   the `make_*_tex.py` scripts write the tables of the article. Copy a `predictions/<folder>` to `results/` to skip training.

PRISM-Net model names in the code: `prismr+paug` (PRISM-Net), `prismx+paug` (PRISM-Net-Lite).

## Licence

Code: MIT License. Splits, duplicate lists, protocols and predictions: CC BY 4.0. RIAWELC and SWRD images remain under
the terms of their original publishers and are not redistributed here.
