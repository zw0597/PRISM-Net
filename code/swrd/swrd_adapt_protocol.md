# SWRD adaptation experiments — protocol fixed before any of these results were computed
(written 2026-11-12; only the zero-shot results of swrd_protocol.md were known at this point)

Data split (radiograph level, seed 0): the 1586 SWRD radiographs that contributed tiles are shuffled once;
the first half is the TEST half, the second half the ADAPTATION pool. Tiles of a radiograph never cross halves.
All numbers below are computed on the TEST half only.

A. Conformal recalibration (no training).
   Frozen fold models (5 per architecture). For n_cal in {25, 50, 100, 200} radiographs drawn at random from the
   adaptation pool (200 repetitions, seed 0), the LAC threshold (eps = 0.10) is recomputed on all tiles of those
   radiographs and applied to the test half. Reported: mean coverage, mean set size, singleton rate; and, for
   reference, the RIAWELC-calibrated threshold on the same test half.
   Models: all architectures of the main table and the three PAug baselines.

B. Few-shot fine-tuning.
   Models: PRISM-Net, ResNet-18, ResNet-18 + PAug (all five fold models of each).
   Budgets: 5% and 20% of all tile-contributing radiographs (79 and 317 radiographs), drawn from the adaptation pool
   (nested: the 5% set is contained in the 20% set; seed 0); all their tiles are used.
   Recipe (identical for all models, not tuned): start from the RIAWELC weights; 200 AdamW steps, batch 64,
   peak lr 5e-4, wd 0.05, 20 warm-up steps, cosine decay; same geometric augmentation as in training; PAug only for
   models trained with PAug; same loss as in training (evidential for PRISM-Net, cross-entropy otherwise);
   one CPU thread; seed 0. No early stopping, no checkpoint selection, no use of the test half.
   Reported on the test half: accuracy, macro-F1 (mean ± std over the 5 fold models), ECE, and the paired
   radiograph-bootstrap interval of the accuracy difference (fold models averaged), plus coverage/size of the
   conformal sets recalibrated on the adaptation radiographs of the same budget.
All results are reported whatever their direction.
