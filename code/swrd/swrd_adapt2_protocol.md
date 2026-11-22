# SWRD adaptation, round 2 — protocol written 2026-11-12 before any round-2 result was computed

Context (disclosed in the paper): the round-1 recipe (swrd_adapt_protocol.md) left PRISM-Net behind ResNet-18 at the
20% budget on the test half. Round 2 develops a better adaptation procedure WITHOUT using the test half for any decision.

Radiograph split of the adaptation pool (793 radiographs, seed 2):
  L5 / L20 : labeled budgets, unchanged from round 1 (79 / 317 radiographs, L5 ⊂ L20)
  U        : 238 radiographs of pool \ L20 used only as UNLABELED target data
  V        : the other 238 radiographs of pool \ L20, labeled, used ONLY to choose among candidate recipes (disclosed)
The test half (793 radiographs) is not used by any decision; it is evaluated once per final recipe.

Candidate recipes (identical candidate set for every model):
  C0  round-1 recipe (200 steps, lr 5e-4, PAug if the model was trained with it)
  C1  AdaBN (BN statistics re-estimated on unlabeled U tiles) + 400 steps, lr 1e-3, PAug as trained
  C2  C1 without PAug during fine-tuning (identical to C1 for ResNet-18 and therefore not run for it)
  C3  C1 followed by confidence-filtered self-training: the C1 model labels the U tiles, tiles with max probability
      >= 0.9 are added with their pseudo-label, and the model is fine-tuned for 200 further steps (lr 5e-4) on L ∪ pseudo-U
All other settings as in round 1 (AdamW, wd 0.05, batch 64, 5% warm-up, cosine, geometric augmentation, training loss).

Selection: for each model and budget, the recipe with the highest accuracy on V, computed with the fold-0 model,
is chosen (ties -> simpler recipe, order C0 < C1 < C2 < C3). The chosen recipe is then run for all five fold models and
evaluated on the test half: accuracy, macro-F1, ECE, radiograph-bootstrap difference to PRISM-Net (fold models averaged).
Zero-label adaptation: AdaBN alone (no labels, U tiles only) is also evaluated on the test half for every architecture.
Round-1 results remain in the paper; round-2 results are reported as a disclosed second round, whatever their direction.
