# Vendored `shlif` runtime package
Copied from `hakaton_nornikel/shlif/` at source commit (see below). Training
scripts stay in the origin repo. Only the runtime pipeline lives here.
Do NOT import `shlif.talc_unet` or torch at module top level — GPU is optional.

Source commit: d044d1c733f99d986f5f2d46c406e6ce2d8d6660

## Divergence from origin (2026-07-04)
The vendored package has since been **extended here** (not yet ported back to
origin) with runtime improvements borrowed from peer solutions:
- `talc.py` — `blue_line_mask` also detects cyan; new `strip_annotation`
  (inpaint marks before features) and `dark_gray_phase` (dispersed talc-share proxy).
- `features.py` — `extract_features` strips annotation first (leak guard).
- `tiling.py` — `tile_blend_weight` (linear feather for seamless panorama stitch).
- `talc_unet.py` — `resolve_threshold` adaptive talc-map cascade; `talc_unet_mask(thr=None)`.
- `analyze.py` — reports `talc_share_est` in metrics.
- new `uncertainty.py` — ensemble-perturbation confidence / undetermined_fraction / zones.
- new `ore_unet.py` — guarded loader/inference for the trained ore/matrix U-Net
  (`unet_ore.pt`), ported from `hakaton_nornikel/scripts/sam2_prelabel.py::build_unet` /
  `unet_ore_decision`. Not present as a standalone module in origin (origin's version lives
  inline in a CLI script); wired into `panorama.py`'s ore/matrix gate with a classical fallback.
- `segment.py` — removed the `not_olive` gate from `segment_phases`'s magnetite
  criterion (was `mid & neutral & not_olive`, now `mid & neutral`). Confirmed on
  LumenStone ground truth + the project's own labelled images that the gate's
  assumption (olive hue = matrix) is backwards (olive = sulfide) and the
  underlying absolute Lab b-channel threshold doesn't transfer across images
  with different lighting. `config/default.yaml`'s `segment.green_b_min` is
  removed accordingly. `backend/models/classifier.pkl` (not git-tracked --
  `backend/models/` is gitignored) was re-extracted and retrained locally
  against the new segmentation (see `backend/scripts/retrain_classifier.py`);
  3-class stratified-CV macro-F1 0.746->0.739, macro-AUC OvR 0.908->0.907 (both
  well within the plan's regression tolerance -- see
  `docs/superpowers/plans/2026-07-04-segment-phases-magnetite-fix.md`). Note:
  this 3-class macro-F1/AUC is a *different* metric from the binary
  ordinary-vs-hard F1/AUC this README's Models table cites elsewhere (~0.84/0.92,
  per `hakaton_nornikel`'s own WORKLOG) -- the two are not directly comparable,
  and the binary metric was not re-measured as part of this fix (retrain_classifier.py
  only computes the 3-class metric, matching `hakaton_nornikel/scripts/train_classifier.py`).
  Not yet ported back to origin `hakaton_nornikel` (that repo had unrelated
  in-progress work at the time of this fix -- port forward when convenient).
When syncing origin, port these forward rather than overwriting from the pinned commit.
