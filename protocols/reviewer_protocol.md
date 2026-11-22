# Two additional experiments — protocol written 2026-11-13 16:10 (UTC+8), before any of their results was computed
Both are run once and reported in the paper whatever their direction.

## R1. PGA against generic input standardisation (zero-label, SWRD test half)
Question: does the physics-guided two-parameter alignment (PGA-R) do better than standard, physics-agnostic
intensity standardisation applied at the same level (per radiograph, no labels, classifier unchanged)?
Baselines (per radiograph, pooled pixels of its tiles; source = the same RIAWELC tile subset used by PGA, every third tile):
  ZS-R : linear moment matching in intensity, J = clip((I - mean_t)/std_t * std_s + mean_s, 0, 1).
  HM-R : exact histogram matching, J = F_s^{-1}(F_t(I)) on the 256 grey levels (CDF matching to the pooled source histogram).
Each is evaluated alone and with AdaBN (statistics re-estimated on the transformed U tiles), exactly as PGA-R was:
all models of the paper, five fold models each, frozen weights, test half (793 radiographs).
Reported: accuracy per model; radiograph-bootstrap interval (10^4, rng 0) of PGA-R minus each baseline, per model,
alone and with AdaBN; number of models for which the difference is significantly positive / negative;
the mean difference over models.

## R2. Weld-literature baseline: VGG-style network
RIAWELC studies that report the highest accuracies use VGG-style networks. We add VGG-11 with batch normalisation
(torchvision `vgg11_bn` convolutional part, global average pooling, one linear layer; grey tile replicated to three
channels as for the other torchvision models), trained from scratch with exactly the baseline recipe
(1000 AdamW steps, batch 64, lr 2e-3, wd 0.05, same augmentation, cross-entropy, seed 0, 80 px), on the official split and
on the five film-grouped folds. Because one run takes ~4x longer than ResNet-18, two CPU threads are used per run.
It then enters every analysis in which the other baselines appear: main table, film-level tests, per-class F1,
corruptions, conformal sets, nested selection pool, SWRD frozen test, recalibration, AdaBN, PGA and R1.
No hyper-parameter is tuned for it (none was tuned for any baseline).

## R3. Seed repetition of the strongest baseline (written 2026-11-14, before any seed-1/2 VGG-11 result exists)
VGG-11 became the strongest baseline with a single seed, whereas PRISM-Net and ResNet-18 have three. VGG-11 is retrained
with seeds 1 and 2 on all five film-grouped folds, with exactly the recipe of R2 (1000 AdamW steps, batch 64, lr 2e-3,
wd 0.05, 80 px, two CPU threads). No other setting is changed and no PRISM-Net run is added or repeated.
Analysis, identical to the existing seed study (make_seed_tex.py): accuracy per seed and fold; mean over folds per seed and
its std over seeds; PRISM-Net vs VGG-11 on film accuracies averaged over the three seeds (tile-weighted difference, film
bootstrap 10^4 draws rng 0, Wilcoxon over 29 films, films won/lost); number of the 15 seed-fold pairs won.
Reporting: the three-seed table and the comparison go into the paper whatever the direction (PRISM-Net significantly
ahead, indistinguishable, or VGG-11 ahead), and every statement about VGG-11 in abstract, results, limitations and
conclusion is aligned with the three-seed result. All single-seed tables stay on seed 0 for all models, as before.

## R4. Longer training budget (written 2026-11-19 09:35 UTC+8, before any long-budget run exists)
Question: do the main conclusions on the film-grouped protocol depend on the short common budget (1000 steps, ~5 epochs)?
Models: PRISM-Net (prismr+paug), ResNet-18 (its trunk), VGG-11 (strongest baseline). Recipe identical to the paper
(AdamW, batch 64, peak lr 2e-3, wd 0.05, 80 px, same augmentation and losses, seed 0, two CPU threads) except
--iters 3000 (three times the budget; warm-up and cosine decay scale with the budget as coded). Five film-grouped folds.
Output to results_long/. For robustness to container interruptions, training saves a resumable checkpoint (weights,
optimiser, scheduler, step, all RNG states) after every epoch; a resumed run is bit-for-bit the uninterrupted run.
Analysis (make_long_tex.py): per model and fold test accuracy, mean +- std over folds, ECE (15 bins), mean accuracy over
the 15 corruptions; change from 1000 to 3000 steps per model; PRISM-Net vs ResNet-18 and vs VGG-11 at 3000 steps with the
same film-level analysis as the main table (tile-weighted difference, film bootstrap 10^4 draws rng 0, Wilcoxon over
the 29 films, films won/lost). Reporting: a table and a paragraph in the paper whatever the direction; if the ranking
or any significance statement changes at 3000 steps, the abstract, results and conclusion are aligned with it.
The single-budget tables of the paper stay on 1000 steps (all 15 models share that budget). No other setting is tuned.
R4 addendum (written 2026-11-19 09:55 UTC+8, still before any long-budget result exists): the authors' GPU machine is
used instead of the 2-core container. To keep the comparison within one hardware/software stack, BOTH budgets
(1000 and 3000 steps) are run on that GPU for the three models (30 runs, outputs results_gpu_1000/ and results_gpu_3000/);
GPU kernels are not bit-identical to the CPU runs of the paper, so the R4 table compares GPU-1000 with GPU-3000 and the
film-level tests use GPU-3000. The container CPU run started at 09:40 is a fallback only: it is stopped once the first
GPU run has finished correctly, and its results are not used if the GPU set completes. Everything else as in R4.
R4 addendum 2 (written 2026-11-22, before any R4 result exists; the only earlier GPU attempt crashed in its first step
with an MPS complex-multiply assertion in torch 2.2.2 and produced no result): the GPU runs use a separate environment
.venv_mps (Python 3.11, torch 2.15.0, torchvision 0.29.2) on the authors' Apple M5 Max (MPS backend; the FFT of RASG falls
back to the CPU where MPS lacks a kernel). Two runs execute concurrently. Both budgets are run in this same stack, exactly as
in addendum 1; train.py, models.py, the recipe and the seed are unchanged. Analysis and reporting as fixed above.

R4 addendum 3 (written 2026-11-22 05:40 UTC+8, after 10 of the 30 runs had finished; nothing in the analysis is changed):
the first R4 driver ran as a child of an interactive session and was killed together with it at about 01:56, during
resnet18 fold 1 (1000 steps, after epoch 1) and vgg11_bn fold 1 (3000 steps, after epoch 6). It was restarted at 05:29 in
a detached screen session with the unchanged run_r4_mps.sh: finished runs are skipped and the two interrupted runs resume
from their per-epoch checkpoints (weights, optimiser, scheduler, step, RNG states). On MPS a resumed run need not be
bit-identical to an uninterrupted one; the two affected runs are named in the paper. Analysis and reporting as fixed above.

## R5. Class-conditional coverage of the conformal sets (written 2026-11-22, before it was computed)
Question (anticipated reviewer question for safety-critical classes such as CR): is the 90% marginal coverage of the
split-conformal sets on unseen films distributed evenly over the classes, and what does class-conditional (Mondrian)
calibration cost? Data: the existing seed-0 film-grouped predictions of all 15 models (no retraining).
Analysis (cp_class.py): for eps in {0.10, 0.05}, per fold: (a) standard split conformal as in the paper (LAC score,
threshold from the validation films); (b) Mondrian split conformal, one threshold per class from the validation tiles
of that class, set = {k : 1 - p_k <= q_k}. Report per class the empirical coverage (mean over folds), mean set size,
singleton fraction, and the worst-class coverage. Reported in the paper whatever the outcome.

## R6. Longer budget for all fifteen models (written 2026-11-22 08:11 UTC+8, before any R6 run exists)
Question: R4 showed that tripling the budget raises ResNet-18 and VGG-11 by about 2 points and PRISM-Net by 0.7, so the
ranking of Table III (1000 steps) may depend on the short budget for the other twelve models as well. R6 extends the
3000-step GPU arm of R4 to every model of Table III.
Runs: the 12 models not covered by R4 (prismx+paug, mobilenet_v3_small, shufflenet_v2_x1_0, efficientnet_b0, vit, gfnet,
convnextv2_atto, repvit_m0_9, mobilenetv4_conv_small, resnet18+paug, mobilenet_v3_small+paug, gfnet+paug) x five
film-grouped folds x 3000 steps, seed 0, exactly the R4 command (run_r6_mps.sh; same train.py, models.py, recipe and
.venv_mps stack; up to three runs concurrently). The 15 R4 runs at 3000 steps are reused unchanged. A smoke test of 20
steps per model (no evaluation of corruptions, output /tmp, discarded) was run before this protocol only to check that
every model runs on MPS. 3000 steps (about 15 epochs) is kept from R4 so that the two sets form one table; it is not a
claim of convergence, and the final training loss/accuracy are reported in the supplement.
Analysis (r6_analysis.py, written now): for all 15 models at 3000 steps, mean +- std over folds of accuracy, macro-F1,
ECE (15 bins) and mCA (mean accuracy over the 15 corruption levels); PRISM-Net vs each of the 14 baselines with the
film-level analysis of Table IV (tile-weighted difference, film bootstrap 10^4 draws rng 0, two-sided Wilcoxon over the
29 films, films won/lost) and Holm's correction over the 14 Wilcoxon p-values.
Reporting: the 15-model table replaces the 3-model R4 table in Section V-E (the 1000/3000 comparison of R4 stays in the
text), whatever the outcome. Table III and all analyses built on its models (robustness, reliability, external test,
adaptation) stay on the 1000-step CPU models, since those were fixed before the external test. If PRISM-Net is not the
most accurate model at 3000 steps, or any statement of the abstract, contributions or conclusion about its rank or about
significance does not hold at 3000 steps, those statements are rewritten to state the 3000-step result.
Dry run of r6_analysis.py (2026-11-22 ~08:20, before any R6 result): restricted to the three R4 models via R6_TEST=1; it reproduces long_summary.json; outputs discarded.
r6_analysis.py frozen (md5 304c03d7a85e0845031e2c9d1264b5e1, recorded 2026-11-22 09:03 UTC+8; unchanged since the dry run, before the R6 analysis is executed).
R6 reporting note (2026-11-22, after the analysis): at the authors' request both the 15-model table (with all film-level columns) and the R4 1000/3000 table are kept in the main text; per-fold values and training losses are in the supplement. No number or analysis changed.

## R7. Two-stage decision layer with a controlled missed-defect rate (written 2026-11-22 14:05 UTC+8, before any R7 quantity was computed)
Motivation: R5 showed that the marginal 90% sets under-cover LP on unseen films. In inspection the critical error is a defect
accepted as sound (missed defect); its rate is the tile-level complement of the probability of detection (POD).
Method (no retraining; existing seed-0 film-grouped predictions of all 15 models, logits Zva on the validation films, Zte_clean_0
on the test films; probabilities: softmax for CE models, Dirichlet mean alpha/S with alpha = softplus(z)+1 for EDL models):
 Stage 1 (sound acceptance): score s = p_ND. On the validation defect tiles (y != ND, n of them) set
   tau = s_(k), the k-th smallest score, k = ceil((n+1)(1-alpha)); tau = +inf if k > n. A tile is accepted as sound iff s > tau.
   Under exchangeability of calibration and test defect tiles, P(accept | defect) <= alpha (finite sample).
 Stage 2 (defect pathway): for tiles with s <= tau, class-conditional (Mondrian) split conformal with score 1 - p_k, one
   threshold q_k per class k from the validation tiles of class k with s <= tau (conformal quantile, level 1-eps); set
   C = {k : 1 - p_k <= q_k}. A singleton C is an automatic decision; otherwise the tile is referred to the inspector.
   Guarantee: for a non-accepted test tile of class k, P(k in C) >= 1-eps under within-class exchangeability.
 Primary setting: alpha = 0.02, eps = 0.10. Sensitivity: alpha in {0.01, 0.05}, eps = 0.05.
Comparators (same predictions): (B1) marginal split conformal of the paper (eps=0.10 and eps=alpha), a tile accepted as sound iff
 C = {ND}; (B2) Mondrian conformal of R5 (eps=0.10), same acceptance rule.
Reported per model (mean over the five test folds; RIAWELC): achieved missed-defect rate MDR = P(accepted | defect), also per
 defect class and per test film (distribution over the 29 films; fraction of films with MDR > alpha); sound acceptance rate
 AR = P(accepted | ND); per-class coverage of stage 2 among non-accepted tiles; overall automatic-decision fraction and referral
 fraction; mean set size. Film-level paired comparison of PRISM-Net vs ResNet-18 and vs VGG-11 for AR and for the
 automatic-decision fraction (tile-weighted difference, film bootstrap 1e4 rng 0, Wilcoxon over films).
Secondary (RIAWELC, evidential models only): stage-1 score b_ND = (alpha_ND - 1)/S (subjective-logic belief) instead of p_ND.
Film-level guarantee note: a guarantee for a new film (each film one exchangeable unit) needs at least ceil(1/alpha)-1
 calibration films (49 for alpha = 0.02); the per-fold validation films (5-7) cannot give it, which is reported, not hidden.
SWRD (external system): stage 1 and 2 recalibrated on the tiles of 25 labeled adaptation radiographs, the same 200 draws as in the
 paper (rng 0, draw sizes 25/50/100/200 generated in that order), for every fold model, on frozen predictions and on PGA-R +
 AdaBN predictions; report test-half MDR (mean and 90th percentile over draws x folds), AR, per-class coverage, automatic
 fraction, and how often tau = +inf. Analysis script: r7_decision.py (written now, before running). Reported in the paper
 whatever the outcome; if the achieved MDR on unseen films exceeds alpha, this is stated.
r7_decision.py written before running (md5 86ea83fff6328da6b2efc1c7f648f897, 2026-11-22 14:06 UTC+8); it also reports overall per-class coverage (accepted tiles count as the set {ND}).
R7 outcome and reporting note (2026-11-22, after the analysis; nothing re-run or changed): the explicit missed-defect threshold did
not improve on the marginal rule of the paper. At alpha = 0.02 the achieved MDR on unseen films was 2.17% for PRISM-Net, 1.95% for
PRISM-Net-Lite and 2.36-4.52% for the baselines (14 of 15 models above alpha), up to 28.5% (PRISM-Net) and 71% (baselines) on single
films, and stage 2 referred about 70% of the tiles; the marginal rule (eps = 0.10) gives MDR 0.18% (PRISM-Net) at 98.5% sound
acceptance. Reported as an analysis result (main text, Section V-L, and Supplementary Table "Missed-Defect Rate"), not as a method
contribution; the marginal rule is kept.

## R8. Methodological extensions of the acquisition model (written 2026-11-22 15:03 UTC+8, before any R8 run or computation)
Goal: test whether the acquisition model can be used more deeply than as augmentation + test-time alignment. Four candidates,
all fixed now without tuning; each is adopted into the method only if it passes the decision rule below.
 (A) Orbit consistency (train_r8.py --oc 1.0): each training tile is augmented once geometrically and then twice independently by
     PAug; loss = mean of the two evidential losses + 1.0 x Jensen-Shannon divergence between the two Dirichlet means.
 (B) Canonical-acquisition training (train_r8.py --canon): every radiograph (film x digitized view) is mapped by PGA-R to a fixed
     reference acquisition (median and 1.4826 MAD of the log-intensity of the fold's training tiles), for training, validation and
     test; corrupted test tiles are canonicalized after the corruption, per radiograph, as PGA-R would be applied in practice; on
     SWRD the same per-radiograph mapping with the fold's reference is used. PAug stays on.
 (A+B) both together.
 All three: prismr+paug, five folds, seed 0, 1000 steps, the R4 GPU stack and recipe (run_r8_mps.sh). Comparator: the R4 GPU
 1000-step PRISM-Net (results_gpu_1000), plus (B0) that comparator with test-time canonicalization of the RIAWELC test radiographs.
 (C) Propagation of acquisition uncertainty (GUM Supplement 1 style, inference only): for each SWRD radiograph the PGA statistics are
     bootstrapped over its tiles (200 resamples, rng 0), giving 200 (c, g) corrections; class probabilities are averaged over 32 of
     them (rng 0). Compared with single PGA-R on frozen models.
 (D) Content-robust PGA (inference only): the radiograph statistics are computed only from its tiles with p_ND >= 0.9 under the
     model applied to the PGA-R input (fallback: all tiles if fewer than 3), and the source reference only from RIAWELC ND tiles;
     compared with PGA-R and with linear moment matching restricted to the same tiles (ZS-CR).
Endpoints. RIAWELC (A, B, A+B, B0): accuracy, macro-F1, ECE, mCA over the 15 corruption levels; film-level comparison with the
 comparator (tile-weighted difference, film bootstrap 1e4 rng 0, Wilcoxon). SWRD test half: accuracy and ECE frozen, with PGA-R
 and with PGA-R + AdaBN (U radiographs, as in the paper); radiograph-bootstrap 95% intervals (1e4, rng 0). C and D: SWRD test-half
 accuracy, ECE, NLL for all 15 models.
Decision rule (selection on validation data only): a candidate is adopted if it improves accuracy (A, B, A+B: mean over the
 validation films of the five folds; C, D: SWRD validation radiographs V) without increasing ECE by more than 1 point; among A, B,
 A+B at most one is adopted (highest validation accuracy). Test results of every candidate are reported whatever the decision.
Scripts: train_r8.py, run_r8_mps.sh, r8_eval.py (written before results). Nothing else is tuned.
r8_eval.py written before any R8 result (md5 49fd4c7869bafd7227b977bfebb78dfd, 2026-11-22 15:08 UTC+8).

R8 decisions (written 2026-11-22 17:19 UTC+8, after the R8 results): by the pre-specified rule, (A) orbit consistency is adopted (validation
 accuracy 89.8% vs 88.9%, ECE 4.5% vs 4.6%); (B) and (A+B) are not (validation accuracy 87.7%, 88.2%); (C) is not adopted (validation
 accuracy lower), its calibration effect is reported; (D) is adopted over PGA-R (higher validation accuracy for all 15 models).
 Post-hoc and disclosed: the content-restricted linear form ZS-CR, a pre-specified comparator, had higher validation accuracy than D
 for all 15 models (mean 42.9% vs 39.3%); the test results of both were visible when this was noticed. The final method therefore
 uses content-aware alignment with the linear form, chosen on the validation radiographs; the paper states that the choice was made
 after the R8 test results were known.

## R9. Final label-free chain on SWRD (written 2026-11-22 17:19 UTC+8, before any R9 quantity was computed)
Chain for every frozen fold model of the 15 CPU models (results/): (1) tiles of each radiograph with p_ND >= 0.9 under PGA-R input
 (fallback all tiles if fewer than 3) define the sound reference region; (2) content-aware alignment, linear form (CAA): the intensity
 mean and standard deviation of these tiles are matched to those of the RIAWELC sound tiles; (3) AdaBN on the CAA inputs of the U
 radiographs; (4) acquisition-uncertainty propagation (UP): the statistics of step (2) are bootstrapped over the selected tiles of each
 radiograph (200 resamples, rng 0, 32 used) and the class probabilities averaged; (5) conformal recalibration on 25 labeled pool
 radiographs (the paper's 200 draws, rng 0) at eps = 0.10.
Reported on the SWRD test half: accuracy, ECE, NLL after steps 2, 3 and 4, against the paper's chain (PGA-R, PGA-R + AdaBN);
 radiograph-bootstrap intervals (1e4, rng 0) for CAA+AdaBN vs PGA-R+AdaBN and for +UP vs without; coverage, mean set size and singleton
 fraction after step 5, for the final chain and for the paper's chain. The same chain is applied to the orbit-consistency models
 (results_r8_oc) and to their GPU comparator (results_gpu_1000). Script r9_chain.py (written now). Reported whatever the outcome.
r9_chain.py written before running (md5 12b2436326aa1d37f2c1693549abab3a, 2026-11-22 17:23 UTC+8); for the 15 CPU models the paper's stored PGA-R and PGA-R + AdaBN predictions are used.
R9 outcome and stage choice (2026-11-22 23:33 UTC+8, after R9): on the validation radiographs V, CAA alone had the highest mean
 accuracy over the 15 models (42.9%, vs 42.5% CAA+AdaBN, 42.0% +UP, 40.9% PGA-R+AdaBN) and was best for 9 of 15 models; it is therefore
 the label-free alignment of the final method (no AdaBN, no target pool). UP is kept as an optional calibration step (validation ECE
 32.4% vs 37.2%). Test results of all stages were visible when this choice was made; this is stated in the paper. Post-analysis
 r9_post.py (same procedures as r9_chain.py): CAA vs PGA-R+AdaBN radiograph-bootstrap intervals and recalibration for the CAA stage.
