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
When syncing origin, port these forward rather than overwriting from the pinned commit.
