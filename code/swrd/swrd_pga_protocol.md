# Physics-guided acquisition alignment (PGA) — protocol written 2026-11-13 before any PGA result was computed

Idea: the acquisition model of the paper, J = clip(c * I^g), is a two-parameter group acting on the intensity.
The unknown parameters of a NEW acquisition system relative to the source system are estimated from UNLABELED
target tiles and inverted at the input; no label and no training are needed, and any classifier can be used unchanged.

Estimation (closed form): with a = ln max(I, 1/255), let m(.) be the median and s(.) = 1.4826*MAD over all pixels.
  g_hat = s_source / s_target,   ln c_hat = m_source - g_hat * m_target,   J = clip(c_hat * I^g_hat, 0, 1).
Source statistics: all RIAWELC tiles (no labels). Target statistics:
  PGA-S (system level): pooled pixels of the unlabeled U radiographs of SWRD (238 radiographs); one (c, g) for the system.
  PGA-R (radiograph level): pooled pixels of all tiles of the same radiograph (labels unused); one (c, g) per radiograph.
Evaluation on the SWRD test half (793 radiographs), all 14 models, five fold models each, frozen weights:
  frozen; PGA-S; PGA-R; AdaBN; PGA-S + AdaBN; PGA-R + AdaBN (BN statistics re-estimated on PGA-transformed U tiles).
  Metrics: accuracy, macro-F1, ECE; radiograph-bootstrap intervals (i) of each variant against the frozen model of the
  same architecture and (ii) of PRISM-Net against every other model under the same variant.
Few-shot: the round-2 recipes already selected per model and budget (round2_selection.json) are re-run unchanged on
  PGA-transformed inputs, using the PGA variant (S or R) with the higher mean accuracy on the labeled validation
  radiographs V (frozen models, averaged over the three fine-tuned architectures); budgets 5% and 20%; five fold models.
All results are reported whatever their direction.

## Addendum (2026-11-13, written before the computation): commissioning chain
To evaluate the whole label-light procedure as one chain, the conformal recalibration of swrd_adapt_protocol.md (A)
is repeated unchanged (same 200 draws of 25 calibration radiographs from the pool, rng 0, eps = 0.10) on the
probabilities obtained with PGA-R + AdaBN, for all 14 models. Reported: coverage, mean set size and share of
single-label sets on the test half, next to the values of the frozen models. Reported whatever the direction.
