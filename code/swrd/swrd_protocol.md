# External test on SWRD — protocol fixed before any prediction was computed
(written 2026-11-11, before tile extraction; no SWRD prediction of any model has been looked at)

Purpose: test the frozen RIAWELC models on radiographs that played no role in development.

Models: the saved seed-0 weights of every architecture, one per RIAWELC fold (5 per architecture),
exactly as trained for Table "main". No retraining, no fine-tuning, no recalibration, no threshold change.
ResNet-18 weights for the first fold (index 0), seed 0, were not saved; that run is repeated with identical settings and its
RIAWELC test predictions are checked to be identical to the stored ones before use.

Data: SWRD Raw_data (3679 16-bit TIFF radiographs with LabelMe polygons).
Class mapping: 气孔 -> PO, 未焊透 -> LP, 裂纹 -> CR, sound weld (inside a 焊缝 polygon, no other shape) -> ND.
Other defect types (夹渣, 未熔合, 咬边, 夹钨, 焊瘤, 内凹) and pseudo-defects (伪缺陷) are not used as classes.

Tiles: 227 x 227 native pixels (the RIAWELC tile size), then resized to 80 x 80 with PIL bilinear, as for RIAWELC.
- Defect tile: centred on the polygon's bounding-box centre; if the bounding box is longer than 227 px,
  centres are placed every 227 px along its long axis. A tile is discarded if it leaves the image, if its centre
  is outside the weld polygon, or if its box intersects the bounding box of any shape with a different label
  other than 焊缝 (single-label tiles only).
- ND tile: random centre inside the weld polygon (seed 0), tile fully inside image, intersecting no non-weld shape.
- Intensity: per image, the 1st and 99th percentiles of the pixels inside the weld polygon are mapped linearly
  to 0 and 255 (clipped), then cast to uint8 — a fixed display window that uses no labels.
  Polarity is kept (SWRD, like RIAWELC, shows the weld bright and voids dark).
- Sampling: images visited in random order (seed 0); at most 3 tiles per class per image; at most 1500 tiles
  per class (balanced test set). Images, not tiles, are the independent units.

Endpoints:
- Primary: accuracy, mean over the 5 fold models of each architecture; PRISM-Net vs each baseline compared by an
  image-level bootstrap (10^4 resamples, seed 0) of the accuracy difference, with fold models averaged.
- Secondary: macro-F1, per-class recall, ECE, NLL, coverage and size of the conformal sets with the RIAWELC-calibrated
  thresholds (eps = 0.10), and the 5-model softmax-average ensemble accuracy.
All results are reported whatever their direction. Nothing in this protocol is changed after predictions are seen.
