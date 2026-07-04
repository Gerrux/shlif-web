# Panorama/Close-up Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the manual «крупный план / панорама» mode toggle, auto-detect it from image size, and make the panorama pipeline produce the exact same verdict shape (whole-image mineral-area fractions, not per-tile) and the exact same editable-mask experience that close-up already has.

**Architecture:** Panorama gains a whole-canvas mask assembly step (`_assemble_masks`) that tiles the section, segments/talc-detects each tile, and reassembles one continuous per-pixel mask via core-crop (each tile contributes only its non-overlapping stride, so no pixel is ever counted by two tiles). That assembled mask is fed into `verdict_from_masks` — the exact function close-up already calls — so both paths compute the verdict identically. A shared, capped-resolution "editing copy" (`EDIT_MAX_SIDE`, reused for both paths) is what the Corrector edits and what superpixels/darkness/confidence are computed on; edits upscale (nearest-neighbor) back to native analysis resolution before the final recompute.

**Tech Stack:** Python 3.12 / FastAPI backend (`backend/app`), OpenCV + scikit-image classical CV, pytest; Next.js/TypeScript frontend (`frontend/`), node `--test`.

## Global Constraints

- `direct_max_pixels: 50_000_000` (50 MP) is the tiling-decision threshold — verified against the real sample dataset (`hakaton_nornikel/sumple_dataset`): close-ups top out at ~26 MP, panoramas start at ~126.5 MP, nothing falls in between.
- `EDIT_MAX_SIDE = 2400` px (longer side) is the editing/display resolution cap for **both** paths — the existing, UX-proven close-up constant (`im.thumbnail((2400, 2400))`), not a new number.
- Close-up's own analysis behavior must not change at all (same thumbnail cap, same `analyze_image` call) — only its internal helpers get shared with panorama.
- Every new/modified function must keep working with `models/classifier.pkl` absent for close-up (existing graceful `None` degrade) — panorama continues to hard-require the classifier for `sort`, matching its current behavior (not something this plan changes).
- Follow existing repo conventions: `pytest` from `backend/` (`pythonpath=["."]`, `testpaths=["tests"]`), `@pytest.mark.skipif(loader.load_classifier() is None, ...)` on any panorama test that needs the classifier, the `tiny_rgb` fixture in `backend/tests/conftest.py` for close-up tests.

---

## Task 1: Shared editing-resolution helpers in `masks.py`

**Files:**
- Modify: `backend/app/pipeline/masks.py`
- Test: `backend/tests/test_masks.py`

**Interfaces:**
- Produces: `masks.EDIT_MAX_SIDE: int`, `masks.fit_max_side(arr: np.ndarray, max_side: int, interpolation: int) -> np.ndarray`, `masks.persist_editor_artifacts(jid: str, r: dict) -> None` (`r` needs keys `phase_map, talc, superpixels, darkness, confidence`).

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_masks.py`:

```python
def test_fit_max_side_no_op_when_within_budget():
    arr = np.zeros((100, 200, 3), np.uint8)
    out = masks.fit_max_side(arr, 2400, cv2.INTER_AREA)
    assert out.shape == arr.shape

def test_fit_max_side_downscales_preserving_aspect():
    arr = np.zeros((4000, 2000, 3), np.uint8)
    out = masks.fit_max_side(arr, 2000, cv2.INTER_AREA)
    assert max(out.shape[:2]) == 2000
    assert out.shape[0] == 2 * out.shape[1]  # aspect ratio kept (4000:2000 == 2:1)

def test_persist_editor_artifacts_writes_all_files(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    r = {
        "phase_map": np.zeros((8, 8), np.uint8),
        "talc": np.zeros((8, 8), bool),
        "superpixels": np.zeros((8, 8), np.uint16),
        "darkness": np.zeros((8, 8), np.uint8),
        "confidence": np.ones((8, 8), np.float32),
    }
    masks.persist_editor_artifacts("jobx", r)
    assert (tmp_path / "masks" / "jobx" / "phases.png").exists()
    assert (tmp_path / "masks" / "jobx" / "talc.png").exists()
    assert (tmp_path / "maps" / "jobx" / "superpixels.png").exists()
    assert (tmp_path / "maps" / "jobx" / "darkness.png").exists()
    assert (tmp_path / "maps" / "jobx" / "confidence.png").exists()
```

Add `import cv2` at the top of `test_masks.py` if not already present (it is not — the file currently only imports `io, numpy as np, PIL.Image, app.pipeline.masks, app.pipeline.loader`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks.py -v`
Expected: FAIL with `AttributeError: module 'app.pipeline.masks' has no attribute 'fit_max_side'` (and similarly for `persist_editor_artifacts`).

- [ ] **Step 3: Implement**

In `backend/app/pipeline/masks.py`, change the import block from:

```python
from __future__ import annotations
import io, cv2, numpy as np
from PIL import Image
from skimage.segmentation import slic
from app.shlif import phases
from app.shlif.analyze import verdict_from_masks
```

to:

```python
from __future__ import annotations
import io, cv2, numpy as np
from PIL import Image
from skimage.segmentation import slic
from app.core import paths
from app.shlif import phases
from app.shlif.analyze import verdict_from_masks
```

Then append at the end of the file:

```python

EDIT_MAX_SIDE = 2400  # editing/display working resolution: the already-proven
# close-up budget (previously inlined in api/analyze.py as
# `im.thumbnail((2400, 2400))`); applied uniformly so panorama editing is
# exactly as responsive as close-up editing is today.


def fit_max_side(arr: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    """Resize `arr` (image or integer label map) so its longer side is
    `max_side`, preserving aspect ratio. No-op if already within budget."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    s = max_side / float(max(h, w))
    return cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interpolation)


def persist_editor_artifacts(jid: str, r: dict) -> None:
    """Write the phase/talc masks + superpixel/darkness/confidence maps a
    finished job needs for the Corrector editor. Shared by the closeup and
    panorama result assembly so both produce identically-shaped, equally
    editable artifacts."""
    md = paths.masks_dir(jid)
    mp = paths.maps_dir(jid)
    (md / "phases.png").write_bytes(encode_png_gray(r["phase_map"]))
    (md / "talc.png").write_bytes(encode_png_gray((r["talc"].astype(np.uint8) * 255)))
    (mp / "superpixels.png").write_bytes(encode_png_label_rgb(r["superpixels"]))
    (mp / "darkness.png").write_bytes(encode_png_gray(r["darkness"]))
    (mp / "confidence.png").write_bytes(
        encode_png_gray(np.clip(r["confidence"] * 255.0, 0, 255).astype(np.uint8)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks.py -v`
Expected: PASS (all tests in the file, old and new)

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/masks.py backend/tests/test_masks.py
git commit -m "feat(pipeline): shared editing-resolution + artifact-persist helpers in masks.py"
```

---

## Task 2: Share the uncertainty-ensemble helper between close-up and panorama

**Files:**
- Modify: `backend/app/pipeline/masks.py`
- Modify: `backend/app/pipeline/closeup.py`
- Test: `backend/tests/test_closeup_uncertainty.py` (existing — must still pass unchanged), `backend/tests/test_masks.py` (new test)

**Interfaces:**
- Consumes: nothing new.
- Produces: `masks.uncertainty_for_editor(rgb: np.ndarray, cfg) -> dict` with keys `confidence` (HxW float, resized to `rgb`'s own shape), `undetermined_fraction` (float), `low_conf_zones` (list).

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_masks.py`:

```python
def test_uncertainty_for_editor_returns_full_res_confidence(tiny_rgb):
    cfg = loader.get_config()
    u = masks.uncertainty_for_editor(tiny_rgb, cfg)
    assert u["confidence"].shape == tiny_rgb.shape[:2]
    assert 0.0 <= u["undetermined_fraction"] <= 1.0
    assert isinstance(u["low_conf_zones"], list)
```

(`tiny_rgb` fixture already available via `backend/tests/conftest.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks.py::test_uncertainty_for_editor_returns_full_res_confidence -v`
Expected: FAIL with `AttributeError: module 'app.pipeline.masks' has no attribute 'uncertainty_for_editor'`

- [ ] **Step 3: Implement — move the function into `masks.py`**

Add to the import block of `backend/app/pipeline/masks.py` (from Task 1's version):

```python
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
```

Append to the end of `masks.py`:

```python

_UNC_MAX_SIDE = 1024  # cap the ensemble-segmentation resolution — the fraction is scale-robust


def uncertainty_for_editor(rgb: np.ndarray, cfg) -> dict:
    """Ensemble-perturbation uncertainty, computed on a downscaled copy for
    speed and the confidence map resized back to `rgb`'s own frame. Shared by
    closeup and panorama so both report confidence/low_conf_zones the same way."""
    h, w = rgb.shape[:2]
    s = min(1.0, _UNC_MAX_SIDE / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    u = ensemble_uncertainty(small, cfg)
    conf = cv2.resize(u["confidence"], (w, h), interpolation=cv2.INTER_LINEAR)
    return {"confidence": conf, "undetermined_fraction": u["undetermined_fraction"],
            "low_conf_zones": find_low_conf_zones(u)}
```

- [ ] **Step 4: Update `closeup.py` to use the shared helper**

Replace the full contents of `backend/app/pipeline/closeup.py` with:

```python
from __future__ import annotations
import numpy as np
from app.shlif import analyze_image
from app.shlif.features import extract_features
from app.pipeline import masks, loader

def _sort_card(rgb, cfg):
    bundle = loader.load_classifier()
    if bundle is None:
        return None
    clf, feat, classes = bundle
    feats = extract_features(rgb, cfg)
    proba = clf.predict_proba(np.array([[feats[k] for k in feat]], float))[0]
    probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return {"classes": probs, "top": max(probs, key=lambda k: probs[k])}

def analyze_closeup(rgb: np.ndarray, cfg) -> dict:
    """Classical/CPU path (GPU U-Net wiring is added later behind loader.gpu_available)."""
    res = analyze_image(rgb, cfg, detect_talc_flag=True)  # classical talc seed
    m = res.masks
    phase_map = masks.phase_label_map(m["sulfide"], m["magnetite"])

    unc = masks.uncertainty_for_editor(rgb, cfg)
    metrics = dict(res.metrics)
    metrics["undetermined_fraction"] = unc["undetermined_fraction"]

    return {
        "verdict": {"ore_class": res.ore_class, "text": res.text, "metrics": metrics},
        "sort": _sort_card(rgb, cfg),
        "phase_map": phase_map,
        "talc": m["talc"].astype(bool),
        "superpixels": masks.build_superpixel_map(rgb),
        "darkness": masks.build_darkness_map(rgb),
        "confidence": unc["confidence"],
        "low_conf_zones": unc["low_conf_zones"],
        "text": res.text,
    }
```

(Removes the now-unused `cv2` import, `_UNC_MAX_SIDE` constant, and the private `_uncertainty` function — same behavior, just relocated.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks.py tests/test_closeup_uncertainty.py tests/test_pipeline.py -v`
Expected: PASS — `test_closeup_uncertainty.py` and `test_pipeline.py` exercise `analyze_closeup` only through its public return value, so they must pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add backend/app/pipeline/masks.py backend/app/pipeline/closeup.py backend/tests/test_masks.py
git commit -m "refactor(pipeline): share the uncertainty-ensemble helper between closeup and panorama"
```

---

## Task 3: Tiling primitives for whole-canvas reconstruction

**Files:**
- Modify: `backend/app/shlif/tiling.py`
- Test: `backend/tests/test_tile_core_bounds.py` (new)

**Interfaces:**
- Produces: `tiling.load_working_array(path, cfg) -> np.ndarray`; `tiling.iter_tiles(path, cfg, arr: np.ndarray | None = None) -> Iterator[Tile]` (new optional `arr` param, same yielded `Tile` type); `tiling.axis_tile_starts(size: int, tile: int, step: int) -> list[int]`; `tiling.axis_core_bounds(size: int, tile: int, step: int) -> dict[int, int]` (maps each tile start along one axis to its non-overlapping core end).

**Design note — why per-axis lookup tables, not a per-tile local check:** an earlier version of this task computed each tile's core end locally from `(x, tw, step, W)` by checking "does this tile's own clipped width reach the canvas edge?" That check is unsound: whenever `tile` exceeds `step` by more than one stride (true for every tile/overlap pair in this project's config), *more than one* tile-start near a row's end independently has its raw pixel data reach the true edge — not only the actual last tile the iterator yields — so a local "am I last?" check cannot tell them apart and produces overlapping cores for a large fraction of real canvas widths (empirically ~13% of widths under this project's default tile=1024/overlap=128). The fix computes the full, ordered sequence of tile starts along an axis **once**, up front — replicating `iter_tiles`' exact loop and its tail-tile size filter — and derives each core's end from the *next* start in that sequence (or the canvas size, for the true last one). This is correct by construction: there is no "is this last?" inference left to get wrong.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_tile_core_bounds.py`:

```python
"""Core-crop reconstruction: each tile contributes only its non-overlapping
core when reassembling one continuous canvas from overlapping tiles, so
summing every tile's core covers the canvas exactly once (no gap, no
double count from the overlap band).

axis_core_bounds derives each tile's core end from the *actual* sequence of
tile starts iter_tiles will yield along that axis (including its tw/th < 8
tail-tile skip) -- not from re-deriving "is this the last tile" locally
from a single tile's own clipped width, which cannot disambiguate a
genuinely last tile from an earlier tile whose data also happens to reach
the edge (this happens whenever `tile` is more than one `step` larger than
the canvas remainder -- a common case with this project's tile/overlap
ratio, not a rare corner case)."""
import numpy as np

from app.shlif.tiling import axis_core_bounds, axis_tile_starts


def test_axis_tile_starts_matches_naive_loop_with_tail_filter():
    # naive replica of iter_tiles' 1-D loop + tail-tile skip, for comparison
    size, tile, step = 2000, 320, 256
    expected = []
    for x in range(0, max(1, size - 1), step):
        if min(tile, size - x) < 8:
            continue
        expected.append(x)
    assert axis_tile_starts(size, tile, step) == expected


def test_axis_tile_starts_matches_iter_tiles_actual_yielded_positions(tmp_path):
    """Cross-check against iter_tiles' real behavior (not just a second copy
    of the same loop/filter logic as the test above) -- guards against the
    two implementations silently drifting apart if either one's loop bound
    or tail-filter threshold ever changes without the other."""
    import copy

    from PIL import Image

    from app.pipeline import loader
    from app.shlif.tiling import iter_tiles

    W, H, tile, overlap = 2000, 1500, 320, 64
    step = tile - overlap
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    p = tmp_path / "grid.jpg"
    Image.fromarray(img).save(p, "JPEG", quality=95)

    cfg = copy.deepcopy(loader.get_config())
    cfg.tiling.tile = tile
    cfg.tiling.overlap = overlap

    xs = sorted({t.x for t in iter_tiles(str(p), cfg.tiling)})
    ys = sorted({t.y for t in iter_tiles(str(p), cfg.tiling)})
    assert xs == axis_tile_starts(W, tile, step)
    assert ys == axis_tile_starts(H, tile, step)


def test_axis_core_bounds_last_tile_extends_to_true_edge():
    bounds = axis_core_bounds(2000, 320, 256)
    last_start = max(bounds)
    assert bounds[last_start] == 2000


def test_axis_core_bounds_handles_multiple_tail_tiles_reaching_the_edge():
    # W=1800 with tile=320/step=256: BOTH x=1536 (tw=264) and x=1792 (tw=8)
    # independently have their raw pixel data reach the true edge (x+tw>=W)
    # -- exactly the case a per-tile-local "is this last?" check cannot
    # disambiguate. axis_core_bounds must still give exactly-contiguous,
    # non-overlapping cores.
    W, tile, overlap = 1800, 320, 64
    step = tile - overlap
    bounds = axis_core_bounds(W, tile, step)
    starts = sorted(bounds)
    for i, s in enumerate(starts):
        expected_end = starts[i + 1] if i + 1 < len(starts) else W
        assert bounds[s] == expected_end


def test_full_grid_reconstruction_has_no_gap_or_overlap():
    # sweep several (W, H) pairs, including ones where the stride does not
    # evenly divide the canvas and ones where multiple tail tiles reach the
    # edge (e.g. W=1800) -- the property must hold for every size, not one
    # hand-picked pair.
    tile, overlap = 320, 64
    step = tile - overlap
    for W, H in [(2000, 1500), (1800, 1500), (1801, 1499), (2049, 2049), (640, 640)]:
        x_bounds = axis_core_bounds(W, tile, step)
        y_bounds = axis_core_bounds(H, tile, step)
        canvas = np.zeros((H, W), np.int32)
        for y, y1 in y_bounds.items():
            for x, x1 in x_bounds.items():
                canvas[y:y1, x:x1] += 1
        assert (canvas == 1).all(), f"gap/overlap for W={W}, H={H}"


def test_production_tile_overlap_config_has_no_gap_or_overlap_across_many_widths():
    # the real config (default.yaml): tile=1024, overlap=128 -- sweep a wide
    # range of widths so we don't rely on one dimension happening to avoid
    # the defect this reconstruction must rule out for every image size.
    tile, overlap = 1024, 128
    step = tile - overlap
    for W in range(2000, 6000, 137):  # arbitrary irregular stride, broad coverage
        bounds = axis_core_bounds(W, tile, step)
        canvas = np.zeros(W, np.int32)
        for x, x1 in bounds.items():
            canvas[x:x1] += 1
        assert (canvas == 1).all(), f"gap/overlap for W={W}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_tile_core_bounds.py -v`
Expected: FAIL with `ImportError: cannot import name 'axis_core_bounds'`

- [ ] **Step 3: Implement**

In `backend/app/shlif/tiling.py`, replace:

```python
def iter_tiles(path: str | Path, cfg) -> Iterator[Tile]:
    """Yield overlapping tiles across a (possibly gigapixel) image.

    ``cfg`` is the ``tiling`` config block. Empty tiles are yielded with
    ``empty=True`` (and no heavy work done) unless ``skip_empty`` is false.
    """
    w, h = image_size(path)
    factor = decode_factor(w, h, int(cfg.max_pixels))
    arr = load_rgb(path, max_pixels=int(cfg.max_pixels))
    H, W = arr.shape[:2]
```

with:

```python
def load_working_array(path: str | Path, cfg) -> np.ndarray:
    """Decode the image once at the tiling working scale (memory-safe draft
    decode above ``cfg.max_pixels``). Shared by `iter_tiles` and any caller
    that also needs the full working-scale canvas (e.g. a display copy), so
    a gigapixel file is only ever decoded once per job."""
    return load_rgb(path, max_pixels=int(cfg.max_pixels))


def iter_tiles(path: str | Path, cfg, arr: np.ndarray | None = None) -> Iterator[Tile]:
    """Yield overlapping tiles across a (possibly gigapixel) image.

    ``cfg`` is the ``tiling`` config block. Empty tiles are yielded with
    ``empty=True`` (and no heavy work done) unless ``skip_empty`` is false.
    Pass a pre-loaded ``arr`` (from :func:`load_working_array`) to avoid
    decoding the image twice when the caller also needs the full canvas.
    """
    w, h = image_size(path)
    factor = decode_factor(w, h, int(cfg.max_pixels))
    if arr is None:
        arr = load_working_array(path, cfg)
    H, W = arr.shape[:2]
```

Then, at the end of `tiling.py`, append:

```python


def axis_tile_starts(size: int, tile: int, step: int) -> list[int]:
    """Replicate `iter_tiles`' 1-D loop (`range(0, max(1, size-1), step)`) and
    its tail-tile skip filter (clipped extent < 8px), returning the actual
    sequence of tile starts that will be yielded along one axis. Both
    `iter_tiles` and `axis_core_bounds` derive tile positions from this one
    function, so "the next tile's start" and "the next tile the iterator
    actually yields" can never disagree."""
    starts = []
    for x in range(0, max(1, size - 1), step):
        if min(tile, size - x) < 8:
            continue
        starts.append(x)
    return starts


def axis_core_bounds(size: int, tile: int, step: int) -> dict[int, int]:
    """For each tile start along one axis, the non-overlapping core end it
    contributes when reassembling one continuous canvas: the next tile's
    start, or `size` for the last tile (no next tile exists to claim the
    remainder). Consecutive cores are exactly contiguous by construction —
    this is what makes summing every tile's core cover the canvas once, with
    no gap and no overlap, regardless of how the stride divides the canvas
    or how many tail tiles independently reach the true edge."""
    starts = axis_tile_starts(size, tile, step)
    return {s: (starts[i + 1] if i + 1 < len(starts) else size) for i, s in enumerate(starts)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_tile_core_bounds.py tests/test_tiling_feather.py -v`
Expected: PASS

- [ ] **Step 5: Confirm the existing panorama tests still pass unaffected**

Run: `cd backend && .venv/bin/python -m pytest tests/test_panorama.py tests/test_panorama_aggregate.py -v`
Expected: PASS (`iter_tiles(path, cfg.tiling)` call sites without `arr=` still work — `arr` is optional).

- [ ] **Step 6: Commit**

```bash
git add backend/app/shlif/tiling.py backend/tests/test_tile_core_bounds.py
git commit -m "feat(tiling): load_working_array + axis_core_bounds for whole-canvas mask reassembly"
```

---

## Task 4: Auto-detect mode from image size

**Files:**
- Modify: `backend/app/config/default.yaml`
- Create: `backend/app/pipeline/detect.py`
- Test: `backend/tests/test_detect_mode.py` (new)

**Interfaces:**
- Produces: `detect.detect_mode(width: int, height: int, cfg) -> str` (returns `"closeup"` or `"panorama"`); `detect.detect_mode_from_path(path: str, cfg) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_detect_mode.py`:

```python
"""Auto-detect closeup vs panorama by pixel count — replaces the manual
mode toggle. Threshold picked from the real dataset: closeups top out at
~26 MP, panoramas start at ~126.5 MP (see docs/superpowers/specs/
2026-07-04-panorama-closeup-unification-design.md §1)."""
from app.pipeline import detect, loader

CFG = loader.get_config()


def test_detect_mode_closeup_below_threshold():
    assert detect.detect_mode(5000, 4000, CFG) == "closeup"  # 20 MP


def test_detect_mode_panorama_above_threshold():
    assert detect.detect_mode(13330, 9489, CFG) == "panorama"  # 126.5 MP, real sample size


def test_detect_mode_boundary_is_inclusive_of_threshold():
    thr = int(CFG.tiling.direct_max_pixels)
    assert detect.detect_mode(thr, 1, CFG) == "closeup"       # exactly at threshold
    assert detect.detect_mode(thr + 1, 1, CFG) == "panorama"  # one pixel over
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_detect_mode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline.detect'`

- [ ] **Step 3: Implement**

Add to `backend/app/config/default.yaml`, inside the `tiling:` block (after `ore_density_pct: 92`):

```yaml
  direct_max_pixels: 50000000  # <= это: один проход без тайлинга (крупный план);
                                # выше — тайловый путь (панорама). Разделение по
                                # реальным данным: crop ≤26 МП, панорама ≥126.5 МП.
```

Create `backend/app/pipeline/detect.py`:

```python
"""Automatic close-up/panorama routing by image size — no manual mode toggle.

A close-up is a single field of view (a few MP); a panorama is a stitched
whole-section scan (100+ MP in this dataset). There is a wide, empty gap
between the two in practice (see the design spec), so a single pixel-count
threshold cleanly separates them.
"""

from __future__ import annotations

from app.shlif.imageio import image_size


def detect_mode(width: int, height: int, cfg) -> str:
    """"closeup" (single pass) or "panorama" (tiled) from raw pixel count."""
    return "panorama" if width * height > int(cfg.tiling.direct_max_pixels) else "closeup"


def detect_mode_from_path(path: str, cfg) -> str:
    """Read just the image header (no full decode) and classify it."""
    w, h = image_size(path)
    return detect_mode(w, h, cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_detect_mode.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/config/default.yaml backend/app/pipeline/detect.py backend/tests/test_detect_mode.py
git commit -m "feat(pipeline): auto-detect closeup/panorama mode from image size"
```

---

## Task 5: Whole-canvas mask assembly for panorama (`_assemble_masks`)

**Files:**
- Modify: `backend/app/pipeline/panorama.py`
- Test: `backend/tests/test_panorama_assemble.py` (new)

**Interfaces:**
- Consumes: `tiling.load_working_array`, `tiling.iter_tiles(..., arr=)`, `tiling.axis_core_bounds` (Task 3); `phases.MATRIX/MAGNETITE/SULFIDE` (`app.shlif.phases`); `segment_phases`, `preprocess`, `detect_talc`, `dark_gray_phase` (all pre-existing).
- Produces: `panorama._assemble_masks(path: str, cfg, arr: np.ndarray) -> dict` with keys `sulfide, magnetite, matrix, talc, dg` — all boolean arrays shaped `arr.shape[:2]`, partitioning every pixel into exactly one of `sulfide/magnetite/matrix`.

This task only *adds* `_assemble_masks`; it does not yet wire it into `analyze_panorama` (Task 6 does that), so the existing `_run_panorama`/`analyze_panorama` behavior is untouched here.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_panorama_assemble.py`:

```python
"""_assemble_masks tiles a section, segments/talc-detects each tile, and
reassembles one continuous mask via core-crop — every pixel must land in
exactly one phase (no gap from the last-tile edge, no double count from
the overlap band)."""
import copy

import numpy as np
from PIL import Image

from app.pipeline import loader, panorama
from app.shlif.imageio import load_rgb


def _synthetic_section():
    rng = np.random.default_rng(3)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)  # dark matrix
    # Bright sulfide blob straddling the x=448 core boundary that
    # tile=512/overlap=64 produces (step=448, so core ends fall at
    # 0/448/896/...) -- sized/positioned (not the naive [100:400,100:400])
    # because a bigger/higher-contrast patch pushes `tiling._is_empty`'s
    # adaptive threshold (mean + 2*std) above 255 on this bimodal image,
    # which misflags the whole tile as empty (a pre-existing quirk of
    # `_is_empty`, unrelated to `_assemble_masks` and out of scope here) and
    # would hide the very seam behaviour this test checks.
    img[100:300, 350:550] = 220
    # Mid-grey magnetite blob straddling the y=896 core boundary instead (x
    # safely inside the [896,1344) x-core so only the y-seam is exercised
    # here). Value 60, not the naive 120: 120 converts to Lab L high enough
    # that segment_phases classifies it as sulfide, not magnetite.
    #
    # IMPORTANT: segment_phases runs PER TILE inside _assemble_masks (each
    # tile gets its own independent 3-class Otsu split), not once over the
    # whole image -- verified directly against the real per-tile path, not
    # just a whole-image segment_phases() call (an earlier attempt at this
    # fixture was wrongly validated that way and passed only by accident).
    # A tile containing just dark background + ONE brighter blob is
    # effectively bimodal, and 3-class Otsu on a bimodal population reliably
    # puts the blob in the brightest ("sulfide") band regardless of its
    # absolute value -- there's no genuine "middle" population for it to
    # land in. Getting a real magnetite (middle-band) classification requires
    # a truly trimodal histogram within that same tile: dark matrix + this
    # mid-grey blob + something distinctly brighter still. The small sulfide
    # anchor below supplies that third population (placed in the y=[896,960)
    # overlap band shared by both tiles this blob straddles, so one anchor
    # serves both). Verified empirically: with the anchor present, 60
    # classifies as ~100% magnetite in the blob region across both tiles,
    # and both tiles stay comfortably non-empty under tiling._is_empty
    # (bright_frac ~0.005-0.007, threshold is 0.002).
    img[800:1000, 1000:1200] = 60
    img[890:930, 890:930] = 220  # sulfide anchor -- gives the two tiles the
    # magnetite blob straddles a real trimodal histogram (see note above);
    # not itself asserted on, it only exists to make Otsu's split meaningful
    return img


def test_assemble_masks_partitions_every_pixel_exactly_once(tmp_path):
    img = _synthetic_section()
    p = tmp_path / "section.jpg"
    Image.fromarray(img).save(p, "JPEG", quality=95)

    cfg = copy.deepcopy(loader.get_config())
    cfg.tiling.tile = 512
    cfg.tiling.overlap = 64  # forces multiple tiles over the 1200x2400 image

    arr = load_rgb(str(p), max_pixels=int(cfg.tiling.max_pixels))
    assembled = panorama._assemble_masks(str(p), cfg, arr)

    total = (assembled["sulfide"].astype(np.int32) + assembled["magnetite"].astype(np.int32)
             + assembled["matrix"].astype(np.int32))
    assert total.shape == arr.shape[:2]
    assert (total == 1).all()  # exactly one phase per pixel — no gap, no double-write

    # the seeded bright blob (which straddles an x-tile boundary at this tile
    # size) must still be picked up as sulfide, not lost at the seam
    assert assembled["sulfide"][100:300, 350:550].mean() > 0.5

    # the seeded mid-grey blob (which straddles a y-tile boundary) must
    # still be picked up as magnetite, not lost at the seam
    assert assembled["magnetite"][800:1000, 1000:1200].mean() > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_panorama_assemble.py -v`
Expected: FAIL with `AttributeError: module 'app.pipeline.panorama' has no attribute '_assemble_masks'`

- [ ] **Step 3: Implement**

In `backend/app/pipeline/panorama.py`, change the import block from:

```python
from app.shlif import load_config  # noqa: F401 (kept for parity)
from app.shlif.features import extract_features
from app.shlif.imageio import load_rgb
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import detect_talc
from app.shlif.tiling import iter_tiles, tile_blend_weight, tile_grid
from app.pipeline import loader
from app.core import paths
```

to:

```python
from app.shlif import load_config, phases  # noqa: F401 (load_config kept for parity)
from app.shlif.features import extract_features
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import dark_gray_phase, detect_talc
from app.shlif.tiling import axis_core_bounds, iter_tiles, load_working_array, tile_blend_weight, tile_grid
from app.pipeline import loader, masks
from app.core import paths
```

(`load_rgb` is dropped from panorama.py's own imports — Task 6 replaces its one call site with the shared working array.)

Then insert this new function right after `aggregate_section` (before `def _run_panorama`):

```python
def _assemble_masks(path: str, cfg, arr: np.ndarray) -> dict:
    """Tile the section, segment + talc-detect each tile, and reassemble one
    continuous mask set for the whole working canvas — core-crop (no overlap
    double count, see `axis_core_bounds`) — so `verdict_from_masks` sees the
    same kind of input it gets from a single close-up pass."""
    H, W = arr.shape[:2]
    sulfide = np.zeros((H, W), bool)
    magnetite = np.zeros((H, W), bool)
    matrix = np.zeros((H, W), bool)
    talc = np.zeros((H, W), bool)
    dg = np.zeros((H, W), bool)
    tile_px = int(cfg.tiling.tile)
    step = max(1, tile_px - int(cfg.tiling.overlap))
    x_core_end = axis_core_bounds(W, tile_px, step)
    y_core_end = axis_core_bounds(H, tile_px, step)

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        # a tile's core always starts at its own (x, y) — only the end can be
        # pulled in earlier than the tile's full extent, per axis_core_bounds
        cx0, cy0 = tile.x, tile.y
        cx1, cy1 = x_core_end[tile.x], y_core_end[tile.y]
        lx1, ly1 = cx1 - tile.x, cy1 - tile.y

        if tile.empty:
            matrix[cy0:cy1, cx0:cx1] = True
            continue

        pre = preprocess(tile.rgb, cfg.preprocess)
        seg = segment_phases(pre, cfg.segment)
        tk = detect_talc(pre, seg.labels == phases.MATRIX, cfg.talc)
        dgm, _ = dark_gray_phase(tile.rgb, cfg.talc)

        sulfide[cy0:cy1, cx0:cx1] = seg.sulfide[:ly1, :lx1]
        magnetite[cy0:cy1, cx0:cx1] = seg.magnetite[:ly1, :lx1]
        matrix[cy0:cy1, cx0:cx1] = seg.labels[:ly1, :lx1] == phases.MATRIX
        talc[cy0:cy1, cx0:cx1] = tk[:ly1, :lx1]
        dg[cy0:cy1, cx0:cx1] = dgm[:ly1, :lx1] & (seg.labels[:ly1, :lx1] == phases.MATRIX)

    return {"sulfide": sulfide, "magnetite": magnetite, "matrix": matrix, "talc": talc, "dg": dg}
```

(`x_core_end`/`y_core_end` are computed once per call, outside the tile loop — `axis_core_bounds` replicates `iter_tiles`' own start/filter logic exactly, so every `tile.x`/`tile.y` the loop yields is guaranteed to be a key in the corresponding dict; no `cx1 <= cx0` guard is needed since core spans are non-empty by construction.)

**Note for the implementer (do not "fix" this):** this re-runs `preprocess`/`segment_phases`/`detect_talc` on each tile a second time (the existing `_run_panorama` loop below still runs its own pass for the overlay + sort classifier). That's an accepted, documented tradeoff (see design spec §"Риски") — merging the two loops is a valid future perf optimization, not required for correctness here. Also note tile-local `preprocess` (gray-world white balance + CLAHE) depends on each tile's own statistics, so `_assemble_masks`'s per-pixel labels will *not* bit-for-bit match a single whole-image `segment_phases` pass — this is pre-existing panorama behavior (already true today), not a bug introduced here. That's why the test above checks the *partition* property and a coarse region sanity check, not exact agreement with a single-pass segmentation.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_panorama_assemble.py -v`
Expected: PASS

- [ ] **Step 5: Confirm nothing else broke**

Run: `cd backend && .venv/bin/python -m pytest tests/test_panorama.py tests/test_panorama_aggregate.py -v`
Expected: PASS (unchanged — `_run_panorama`/`analyze_panorama` haven't been touched yet)

- [ ] **Step 6: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama_assemble.py
git commit -m "feat(panorama): assemble one whole-canvas phase/talc mask via core-crop tiling"
```

---

## Task 6: Wire the assembled masks into `analyze_panorama` — same verdict, same sort shape, editable artifacts

**Files:**
- Modify: `backend/app/pipeline/panorama.py`
- Modify: `backend/tests/test_panorama.py` (existing assertions extended)

**Interfaces:**
- Consumes: `panorama._assemble_masks` (Task 5), `masks.verdict_from_masks_dict`, `masks.fit_max_side`, `masks.EDIT_MAX_SIDE`, `masks.uncertainty_for_editor`, `masks.persist_editor_artifacts`, `masks.build_superpixel_map`, `masks.build_darkness_map`, `masks.phase_label_map` (all pre-existing or from Tasks 1-2).
- Produces: `panorama.analyze_panorama(path, cfg, jid) -> dict` now returns the **same shape** `analyze_closeup` + the API wrapper produce: `{mode, verdict: {ore_class, text, metrics}, sort: {classes, top}, text, size, native_size, low_conf_zones, overlay_url, n_ore, n_tiles}`.

- [ ] **Step 1: Extend the existing panorama test to assert the unified shape**

In `backend/tests/test_panorama.py`, replace `test_panorama_runs` with:

```python
@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_runs(tmp_path):
    # a small 2-tile synthetic panorama
    img = (np.random.default_rng(1).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()
    r = panorama.analyze_panorama(str(p), cfg, "testjob")
    assert r["mode"] == "panorama"
    assert r["n_tiles"] >= 1
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
    # same metrics keys close-up produces, computed over the whole image
    for key in ("sulfide_frac", "magnetite_frac", "matrix_frac", "talc_frac",
                "normal_share", "fine_share", "confidence", "talc_share_est"):
        assert key in r["verdict"]["metrics"]
    # same top-level sort-card shape as closeup, not buried in metrics
    assert set(r["sort"]["classes"]) <= {"ordinary", "hard", "talcose"}
    assert r["sort"]["top"] in r["sort"]["classes"]
    assert r["size"][0] > 0 and r["size"][1] > 0
    assert r["native_size"][0] >= r["size"][0] and r["native_size"][1] >= r["size"][1]
    assert isinstance(r["low_conf_zones"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_panorama.py::test_panorama_runs -v`
Expected: FAIL — current result has no `sulfide_frac`/`magnetite_frac`/`matrix_frac`/`normal_share`/`fine_share`/`talc_share_est` in metrics, no `size`/`native_size` keys, and `sort` doesn't exist at top level.

- [ ] **Step 3: Implement — rewrite `_run_panorama` and `analyze_panorama`**

In `backend/app/pipeline/panorama.py`, delete the `DISPLAY_MP = 4_000_000` module constant (no longer used — replaced by `masks.EDIT_MAX_SIDE`), leaving:

```python
SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate
```

Replace the entire `_run_panorama` function body with:

```python
def _run_panorama(path, clf, feat_names, classes, cfg, arr: np.ndarray, min_ore: float = 0.04) -> dict:
    """Tile a panorama, classify ore-rich tiles for the `sort` card (ore-density
    weighted aggregation — unchanged mechanism, see design spec §4.2), and
    stitch the display overlay. The whole-image phase/talc masks and the
    `ore_class` verdict come from `_assemble_masks` + `verdict_from_masks`
    instead (design spec §4.1) — this function no longer decides ore_class."""
    Wt, Ht, factor = tile_grid(path, cfg.tiling)
    edit = masks.fit_max_side(arr, masks.EDIT_MAX_SIDE, cv2.INTER_AREA)
    dh, dw = edit.shape[:2]
    rx, ry = dw / Wt, dh / Ht
    ore_pct = float(getattr(cfg.tiling, "ore_density_pct", ORE_DENSITY_PCT))
    bright_thr = float(np.percentile(cv2.cvtColor(edit, cv2.COLOR_RGB2GRAY), ore_pct))

    base = edit.astype(np.float32)
    # Feathered stitch: accumulate weight*colour per tile and normalise, so
    # overlapping tiles blend seamlessly in the *display* overlay (no double-
    # darkened overlap band, no hard seam) — cosmetic only, unrelated to the
    # whole-canvas mask assembly above.
    color_num = np.zeros((dh, dw, 3), np.float32)
    weight_den = np.zeros((dh, dw), np.float32)
    talc_disp = np.zeros((dh, dw), bool)
    records = []
    n_tiles = n_ore = n_matrix = 0
    t0 = time.time()
    sort_alpha = 0.32

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        n_tiles += 1
        if tile.empty:
            continue
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        matrix = segment_phases(pre, cfg.segment).labels == phases.MATRIX
        talc = detect_talc(pre, matrix, cfg.talc)
        ore_px = int((~matrix).sum())
        ore_frac = ore_px / max(matrix.size, 1)

        dx0, dy0 = int(tile.x * rx), int(tile.y * ry)
        dx1, dy1 = min(int((tile.x + rgb.shape[1]) * rx), dw), min(int((tile.y + rgb.shape[0]) * ry), dh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue

        if ore_frac >= min_ore:
            n_ore += 1
            feats = extract_features(rgb, cfg)
            proba = clf.predict_proba(np.array([[feats[k] for k in feat_names]], float))[0]
            pd = {classes[i]: float(proba[i]) for i in range(len(classes))}
            dens = ore_density(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), bright_thr)
            records.append((pd, dens))
            col = np.array(SORT_RGB[max(pd, key=lambda k: pd[k])], np.float32)
            wgt = tile_blend_weight(dy1 - dy0, dx1 - dx0)
            color_num[dy0:dy1, dx0:dx1] += wgt[..., None] * col
            weight_den[dy0:dy1, dx0:dx1] += wgt
        else:
            n_matrix += 1
        if talc.any():
            td = cv2.resize(talc.astype(np.uint8), (dx1 - dx0, dy1 - dy0),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            talc_disp[dy0:dy1, dx0:dx1] |= td

    sec = aggregate_section(records, classes)
    sort_proba = {classes[i]: float(sec[i]) for i in range(len(classes))}
    sort_top = classes[int(sec.argmax())] if records else classes[0]

    overlay = base.copy()
    cov = weight_den > 0
    if cov.any():
        blended = color_num[cov] / weight_den[cov][..., None]
        overlay[cov] = (1.0 - sort_alpha) * base[cov] + sort_alpha * blended
    out = overlay
    out[talc_disp] = 0.68 * out[talc_disp] + 0.32 * np.array(TALC_RGB, np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return {
        "overlay": out, "edit_rgb": edit, "sort": {"classes": sort_proba, "top": sort_top},
        "n_ore": n_ore, "n_matrix": n_matrix, "n_tiles": n_tiles,
        "seconds": time.time() - t0, "factor": factor,
    }
```

Replace the entire `analyze_panorama` function with:

```python
def analyze_panorama(path: str, cfg, jid: str) -> dict:
    """Public wrapper called by the API. Builds the whole-canvas phase/talc
    masks (design spec §4) and reuses `verdict_from_masks_dict` — the exact
    helper close-up uses — so the result has the same shape and the same
    meaning, computed over the whole image instead of per tile."""
    cfg = copy.deepcopy(cfg)  # don't mutate the shared @lru_cache'd Config
    cfg.tiling.tile = 2048
    cfg.talc.detect_dark_frac = 0.15
    bundle = loader.load_classifier()
    if bundle is None:
        raise RuntimeError("classifier.pkl required for panorama sort")
    clf, feat, classes = bundle

    arr = load_working_array(path, cfg.tiling)
    H, W = arr.shape[:2]

    assembled = _assemble_masks(path, cfg, arr)
    verdict = masks.verdict_from_masks_dict(
        assembled["sulfide"], assembled["magnetite"], assembled["matrix"], assembled["talc"], cfg)
    verdict["metrics"]["talc_share_est"] = float(assembled["dg"].mean())

    run = _run_panorama(path, clf, feat, classes, cfg, arr)
    Image.fromarray(run["overlay"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)

    edit = run["edit_rgb"]
    eh, ew = edit.shape[:2]
    sulfide_small = cv2.resize(assembled["sulfide"].astype(np.uint8), (ew, eh),
                               interpolation=cv2.INTER_NEAREST) > 0
    magnetite_small = cv2.resize(assembled["magnetite"].astype(np.uint8), (ew, eh),
                                 interpolation=cv2.INTER_NEAREST) > 0
    talc_small = cv2.resize(assembled["talc"].astype(np.uint8), (ew, eh),
                            interpolation=cv2.INTER_NEAREST) > 0
    phase_small = masks.phase_label_map(sulfide_small, magnetite_small)
    unc = masks.uncertainty_for_editor(edit, cfg)

    masks.persist_editor_artifacts(jid, {
        "phase_map": phase_small, "talc": talc_small,
        "superpixels": masks.build_superpixel_map(edit),
        "darkness": masks.build_darkness_map(edit),
        "confidence": unc["confidence"],
    })

    return {
        "mode": "panorama",
        "verdict": verdict,
        "sort": run["sort"],
        "text": verdict["text"],
        "size": [ew, eh],
        "native_size": [W, H],
        "low_conf_zones": unc["low_conf_zones"],
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": run["n_ore"], "n_tiles": run["n_tiles"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_panorama.py tests/test_panorama_aggregate.py tests/test_panorama_assemble.py -v`
Expected: PASS (skips instead of failing if `models/classifier.pkl` is absent locally)

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama.py
git commit -m "feat(panorama): verdict + sort + editable artifacts now match closeup's shape exactly"
```

---

## Task 7: API — drop the manual mode field, auto-detect and route

**Files:**
- Modify: `backend/app/api/analyze.py`
- Test: none new here — Task 9 updates the existing API tests that this breaks

**Interfaces:**
- Consumes: `detect.detect_mode` (Task 4), `masks.EDIT_MAX_SIDE`, `masks.persist_editor_artifacts` (Tasks 1-2), `panorama.analyze_panorama` (Task 6).
- Produces: `POST /api/analyze` now takes only `image` (no `mode` field); response unchanged (`{job_id}`).

- [ ] **Step 1: Implement**

Replace the entire contents of `backend/app/api/analyze.py` with:

```python
from __future__ import annotations
import io, numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from PIL import Image
from app.pipeline import closeup, panorama, loader, masks, detect
from app.core import paths
from app.runtime import get_runtime

router = APIRouter()
Image.MAX_IMAGE_PIXELS = None

@router.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    cfg = loader.get_config()
    iw, ih = Image.open(io.BytesIO(data)).size
    mode = detect.detect_mode(iw, ih, cfg)
    jid = get_runtime().store.create(mode)
    up = paths.uploads_dir() / f"{jid}_{Path(image.filename or 'up').name}"
    up.write_bytes(data)

    def work():
        if mode == "panorama":
            return panorama.analyze_panorama(str(up), cfg, jid)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((masks.EDIT_MAX_SIDE, masks.EDIT_MAX_SIDE))
        rgb = np.asarray(im)
        r = closeup.analyze_closeup(rgb, cfg)
        disp = paths.images_dir() / f"{jid}.jpg"
        Image.fromarray(rgb).save(disp, "JPEG", quality=90)
        masks.persist_editor_artifacts(jid, r)
        h, w = rgb.shape[:2]
        return {"mode": "closeup", "verdict": r["verdict"], "sort": r["sort"],
                "text": r["text"], "size": [w, h],
                "low_conf_zones": r["low_conf_zones"]}

    get_runtime().runner.submit(jid, work)
    return {"job_id": jid}
```

(Drops the `mode: str = Form(...)` param and the `Form` import; drops the local `_persist_maps` in favor of `masks.persist_editor_artifacts`; the close-up branch's `2400` constant becomes `masks.EDIT_MAX_SIDE` — same value, single source.)

- [ ] **Step 2: Run the full backend suite to see what this breaks**

Run: `cd backend && .venv/bin/python -m pytest tests/ -v`
Expected: FAIL only in tests that still POST `data={"mode": "closeup"}` (`test_api.py`, `test_api_uncertainty.py`) — FastAPI silently ignores unknown form fields, so these won't error on the request itself, but confirm by running; if they already pass because the field is simply ignored, that's fine, Task 9 still cleans up the now-misleading `data={"mode": ...}` calls for clarity.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/analyze.py
git commit -m "feat(api): auto-detect closeup/panorama mode server-side, no client mode field"
```

---

## Task 8: Upscale edited masks to native resolution before recompute

**Files:**
- Modify: `backend/app/api/masks.py`
- Test: `backend/tests/test_masks_api_upscale.py` (new)

**Interfaces:**
- Consumes: `job.result["native_size"]` (present for panorama jobs since Task 6, absent for closeup jobs).
- Produces: `POST /api/masks/{jid}` behavior unchanged for closeup (no `native_size` → no resize); for panorama, edited label arrays are upscaled (nearest-neighbor) to `native_size` before `verdict_from_masks_dict`.

- [ ] **Step 1: Write the failing test**

A plain area-fraction check will not do here: nearest-neighbor resize of a clean rectangular region preserves its area fraction almost exactly regardless of target resolution, so `sulfide_frac` looks the same whether or not the upscale happens. The property that genuinely depends on resolution is `fine_share`/`normal_share` (`shlif/analyze.py::_intergrowth_split`), which classifies sulfide as "fine" (laced) when it is within an **absolute** `dist_px=12` pixels of magnetite. Build a case where that 12px threshold means "almost all sulfide" at editing resolution but "only a sliver of sulfide" once upscaled to native resolution — that difference only shows up if the upscale actually happens before the recompute.

Create `backend/tests/test_masks_api_upscale.py`:

```python
"""When a job's stored result carries a native_size larger than the edited
PNGs (the panorama case — edited at EDIT_MAX_SIDE, analyzed at native tiled
resolution), POST /masks must upscale (nearest-neighbor) before recomputing
the verdict, so the corrected verdict is computed at native resolution just
like the original one was.

fine_share is the property that actually exposes a missing upscale: it
classifies sulfide as "fine" within an absolute dist_px=12 of magnetite. At
the small edited resolution below, the whole 10px-wide sulfide band sits
within 12px of magnetite (fine_share ~= 1.0). Upscaled 10x to native
resolution, the same band is 100px wide and only its first ~12 native
pixels are within dist_px=12 of magnetite (fine_share should drop sharply).
If save_masks recomputes at the small (un-upscaled) resolution, fine_share
stays ~1.0 and the test fails.
"""
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from app.runtime import get_runtime


def _png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def test_save_masks_upscales_to_native_size_before_recompute():
    c = TestClient(app)
    jid = get_runtime().store.create("panorama")
    get_runtime().store.set_result(jid, {"native_size": [200, 200]})  # 10x the edited size below

    w = h = 20
    pm = np.zeros((h, w), np.uint8)
    pm[:, 8:10] = 1   # magnetite band, 2px wide
    pm[:, 10:20] = 2  # sulfide band, 10px wide, right next to magnetite
    talc = np.zeros((h, w), np.uint8)

    r = c.post(f"/api/masks/{jid}",
               files={"phases": ("phases.png", _png_bytes(pm), "image/png"),
                      "talc": ("talc.png", _png_bytes(talc), "image/png")})
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    # at native (upscaled) resolution only ~12 of the 100 native sulfide
    # columns are within dist_px=12 of magnetite -> fine_share well under 1.0
    assert metrics["fine_share"] < 0.3

    # the saved edit on disk stays at editing resolution — only the
    # in-memory recompute upscales
    from app.core import paths
    saved = np.asarray(Image.open(paths.masks_dir(jid) / "phases.png"))
    assert saved.shape == (h, w)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks_api_upscale.py -v`
Expected: FAIL on `assert metrics["fine_share"] < 0.3` — today's `save_masks` recomputes directly on the 20×20 upload, where the entire 10px sulfide band is within `dist_px=12`, so `fine_share` comes back ~1.0.

- [ ] **Step 3: Implement**

Replace the full contents of `backend/app/api/masks.py` with:

```python
from __future__ import annotations
import cv2, numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.core import paths
from app.pipeline import masks as M, loader
from app.runtime import get_runtime

router = APIRouter()

@router.get("/masks/{jid}/{layer}.png")
def get_mask(jid: str, layer: str):
    p = paths.masks_dir(jid) / f"{layer}.png"
    if layer not in {"phases", "talc"} or not p.exists():
        raise HTTPException(404, "mask not found")
    return FileResponse(p, media_type="image/png")

@router.get("/maps/{jid}/{name}.png")
def get_map(jid: str, name: str):
    p = paths.maps_dir(jid) / f"{name}.png"
    if name not in {"superpixels", "darkness", "confidence"} or not p.exists():
        raise HTTPException(404, "map not found")
    return FileResponse(p, media_type="image/png")

@router.get("/images/{jid}.jpg")
def get_image(jid: str):
    p = paths.images_dir() / f"{jid}.jpg"
    if not p.exists():
        raise HTTPException(404, "image not found")
    return FileResponse(p, media_type="image/jpeg")

@router.post("/masks/{jid}")
async def save_masks(jid: str, phases: UploadFile = File(...), talc: UploadFile = File(...)):
    pm = M.decode_png_gray(await phases.read()).astype(np.uint8)
    tk = M.decode_png_gray(await talc.read()) > 127
    paths.masks_dir(jid).joinpath("phases.png").write_bytes(M.encode_png_gray(pm))
    paths.masks_dir(jid).joinpath("talc.png").write_bytes(M.encode_png_gray(tk.astype(np.uint8) * 255))

    job = get_runtime().store.get(jid)
    native = (job.result or {}).get("native_size") if job else None
    if native and tuple(native) != (pm.shape[1], pm.shape[0]):
        nw, nh = int(native[0]), int(native[1])
        pm = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_NEAREST)
        tk = cv2.resize(tk.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0

    su, mg, mx = M.split_phase_map(pm)
    cfg = loader.get_config()
    v = M.verdict_from_masks_dict(su, mg, mx, tk & mx, cfg)
    get_runtime().store.log_correction(jid, "phases+talc", int(pm.size))
    return v
```

(Note: the *saved* `phases.png`/`talc.png` on disk stay at the uploaded/editing resolution — only the in-memory arrays used for the verdict recompute get upscaled. Closeup jobs never set `native_size`, so `native` is falsy and this is a no-op there — identical to today's behavior.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks_api_upscale.py -v`
Expected: PASS

- [ ] **Step 5: Run the existing mask/API tests to confirm no regression**

Run: `cd backend && .venv/bin/python -m pytest tests/test_masks.py tests/test_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/masks.py backend/tests/test_masks_api_upscale.py
git commit -m "feat(api): upscale edited masks to native analysis resolution before recompute"
```

---

## Task 9: Update tests broken by dropping the client-supplied mode field

**Files:**
- Modify: `backend/tests/test_api.py`
- Modify: `backend/tests/test_api_uncertainty.py`

**Interfaces:** none — pure test cleanup, no production code changes.

- [ ] **Step 1: Update `test_api.py`**

In `backend/tests/test_api.py`, change:

```python
def test_closeup_analyze_and_edit(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", data={"mode": "closeup"},
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    assert up.status_code == 200
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["status"] == "done"
    assert done["result"]["verdict"]["ore_class"] in {"ordinary","hard","talcose","review"}
```

to:

```python
def test_closeup_analyze_and_edit(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze",
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    assert up.status_code == 200
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["status"] == "done"
    assert done["result"]["mode"] == "closeup"  # tiny_rgb (256x256) is well under direct_max_pixels
    assert done["result"]["verdict"]["ore_class"] in {"ordinary","hard","talcose","review"}
```

(The rest of the test — mask fetch + edit + recompute — is unchanged.)

- [ ] **Step 2: Update `test_api_uncertainty.py`**

In `backend/tests/test_api_uncertainty.py`, change:

```python
def test_closeup_result_has_uncertainty(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", data={"mode": "closeup"},
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
```

to:

```python
def test_closeup_result_has_uncertainty(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze",
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
```

- [ ] **Step 3: Run the full backend suite**

Run: `cd backend && .venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests; panorama-classifier tests skip if `models/classifier.pkl` is absent locally)

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_api.py backend/tests/test_api_uncertainty.py
git commit -m "test: drop the client-supplied mode field now that the API auto-detects it"
```

---

## Task 10: Frontend — remove the mode toggle, unify the verdict view and editor

**Files:**
- Modify: `frontend/lib/api/client.ts`
- Modify: `frontend/lib/api/hooks.ts`
- Modify: `frontend/lib/api/types.ts`
- Modify: `frontend/app/page.tsx`
- Modify: `frontend/components/verdict/VerdictPanel.tsx`
- Delete: `frontend/components/PanoramaWorkspace.tsx`

**Interfaces:**
- Produces: `analyze(file: File): Promise<{job_id: string}>` (drops the `mode` param); `useAnalyze()` mutation takes `{file: File}`; `<Corrector/>` is now used for every finished job, not just `mode === "closeup"`.

- [ ] **Step 1: `client.ts` — drop the mode param**

In `frontend/lib/api/client.ts`, change:

```typescript
import type { Job, Mode, Verdict } from "./types";

const base = "";

export async function analyze(file: File, mode: Mode): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("mode", mode);
  const r = await fetch(`${base}/api/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`analyze failed: ${r.status}`);
  return r.json();
}
```

to:

```typescript
import type { Job, Verdict } from "./types";

const base = "";

export async function analyze(file: File): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("image", file);
  const r = await fetch(`${base}/api/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`analyze failed: ${r.status}`);
  return r.json();
}
```

- [ ] **Step 2: `hooks.ts` — drop the mode param**

In `frontend/lib/api/hooks.ts`, change:

```typescript
import { useMutation, useQuery } from "@tanstack/react-query";
import { analyze, getJob } from "./client";
import type { Mode } from "./types";

export function useAnalyze() {
  return useMutation({ mutationFn: (v: { file: File; mode: Mode }) => analyze(v.file, v.mode) });
}
```

to:

```typescript
import { useMutation, useQuery } from "@tanstack/react-query";
import { analyze, getJob } from "./client";

export function useAnalyze() {
  return useMutation({ mutationFn: (v: { file: File }) => analyze(v.file) });
}
```

- [ ] **Step 3: Run the frontend unit tests**

Run: `cd frontend && npm test`
Expected: PASS — `tests/client.test.mjs` only tests the URL builders (`maskUrl`/`mapUrl`/`imageUrl`/`reportUrl`), untouched by this change.

- [ ] **Step 4: `page.tsx` — remove the mode toggle and the mode-based workspace branch**

Replace the full contents of `frontend/app/page.tsx` with:

```tsx
"use client";
import { useState } from "react";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { reportUrl } from "@/lib/api/client";
import type { Verdict } from "@/lib/api/types";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";
import { Welcome } from "@/components/Welcome";
import { ThemeToggle } from "@/components/ThemeToggle";
import { IconHex, IconAlert, IconDownload } from "@/components/icons";

const STATUS: Record<string, [string, string]> = {
  queued: ["queued", "в очереди"], running: ["running", "анализ"],
  done: ["done", "готово"], error: ["error", "ошибка"],
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function runAnalyze(f: File) {
    setFile(f);
    setVOverride(null);
    setJobId(null);
    analyze.mutate({ file: f }, { onSuccess: (r) => setJobId(r.job_id) });
  }

  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
  const started = !!jobId || analyze.isPending;
  const badgeKey = analyze.isPending ? "running" : job.data?.status;

  const infoNode = (
    <>
      <div className="card">
        <div className="side-h">Образец<span className="ann">{shown?.mode === "panorama" ? "панорама" : "крупный план"}</span></div>
        <div className="side-b"><div className="meta-rows">
          <div className="kv"><span className="k">Файл</span><span className="v">{file?.name ?? "—"}</span></div>
          {shown?.size ? <div className="kv"><span className="k">Размер</span><span className="v">{shown.size[0]}×{shown.size[1]}</span></div> : null}
        </div></div>
      </div>
      {shown ? <VerdictPanel result={shown} /> : null}
      {shown && jobId ? (
        <a className="btn ghost" href={reportUrl(jobId)} target="_blank" rel="noopener noreferrer">
          <IconDownload /> Скачать протокол (PDF)
        </a>
      ) : null}
    </>
  );

  if (!started) {
    return (
      <>
        <Welcome onFile={runAnalyze} />
        <div className="theme-float"><ThemeToggle /></div>
      </>
    );
  }

  return (
    <main className="app-main">
      <header className="topbar" style={{ flexWrap: "wrap" }}>
        <div className="logo"><IconHex className="ico-md" /></div>
        <div><div className="crumb">DATA FORCE · классификация руд</div><h1>Скажи мне кто твой шлиф</h1></div>
        <div className="grow" />
        {badgeKey && STATUS[badgeKey] ? (
          <span className={`status-badge ${STATUS[badgeKey][0]}`}><span className="bd" />{STATUS[badgeKey][1]}</span>
        ) : null}
        <ThemeToggle />
      </header>

      {result && result.size ? (
        <Corrector jobId={jobId!} size={result.size} info={infoNode} onVerdict={setVOverride} />
      ) : (
        <div className="workspace">
          <aside className="ws-side">{infoNode}</aside>
          <div className="ws-view">
            <div className="zoom-vp">
              {job.data?.status === "error" ? (
                <div className="stage-empty">
                  <IconAlert className="ico-lg" />
                  <div className="hint">Ошибка анализа</div>
                  <div className="sub">{job.data.message ?? "неизвестная ошибка"}</div>
                </div>
              ) : (
                <div className="stage-empty">
                  <div className="hint">Анализ снимка…</div>
                  <div className="sub">сегментация фаз</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
```

(Removed: `MODES`, `mode` state, `onMode`, the режим `<div className="seg">` toggle, the `imageUrl`/`PanoramaWorkspace` import and branch. `Corrector` now renders whenever `result.size` is present, regardless of `result.mode`.)

- [ ] **Step 5: `VerdictPanel.tsx` — always render the closeup-style verdict**

Replace the full contents of `frontend/components/verdict/VerdictPanel.tsx` with:

```tsx
import type { AnalyzeResult } from "@/lib/api/types";
import { SortCard } from "./SortCard";
import { PhaseBars } from "./PhaseBars";
export function VerdictPanel({ result }: { result: AnalyzeResult }) {
  return (
    <div>
      <SortCard sort={result.sort} />
      <PhaseBars verdict={result.verdict} />
      {result.text ? <div className="note" style={{ marginTop: 12 }}>{result.text}</div> : null}
    </div>
  );
}
```

(`PhaseBars` already renders only the metric rows present in `verdict.metrics` — no changes needed there since panorama now supplies the same keys. The `oreRu` import and the panorama-only branch are removed as dead code.)

- [ ] **Step 6: Delete `PanoramaWorkspace.tsx`**

```bash
git rm frontend/components/PanoramaWorkspace.tsx
```

- [ ] **Step 7: Manually sanity-check the frontend builds**

Run: `cd frontend && npm run build`
Expected: builds cleanly with no unused-import/type errors (`Mode` is still exported from `types.ts` and used by `AnalyzeResult`/`Job`, so no changes needed there).

- [ ] **Step 8: Commit**

```bash
git add frontend/lib/api/client.ts frontend/lib/api/hooks.ts frontend/app/page.tsx frontend/components/verdict/VerdictPanel.tsx
git rm frontend/components/PanoramaWorkspace.tsx
git commit -m "feat(frontend): remove the mode toggle, edit panorama masks like closeup"
```

---

## Task 11: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Start the stack locally**

Follow the project's existing local-dev instructions (`docker-compose.override.yml` per the unified-service design doc, or run `backend`/`frontend` directly per their READMEs) so `/api/analyze` and the Next.js app are reachable.

- [ ] **Step 2: Upload a real close-up**

Pick any file from `hakaton_nornikel/sumple_dataset/Фото руд по сортам. ч2/` (e.g. one of the ~3.9 MP files). Upload it with no mode picker visible. Confirm:
- The result shows a full phase-composition verdict + sort card (same as before).
- «Доработать маски» opens the Corrector, edits recompute, save works.

- [ ] **Step 3: Upload a real panorama**

Pick a file from `hakaton_nornikel/sumple_dataset/Панорамы/` (e.g. `13.jpg`, ~126.5 MP — the smallest, for a faster local run). Confirm:
- No mode picker, and the badge reads «панорама» once the result lands.
- The verdict panel shows the SAME layout as the close-up case: sort card + phase bars with sulfide/magnetite/matrix/talc/normal/fine percentages (not the old «Тальк-кандидаты / N рудных тайлов» mini-card).
- «Доработать маски» opens the Corrector on this panorama job too (previously view-only), edits recompute, save works.
- The downloaded PDF protocol (`/api/report/{jid}.pdf`) shows the full metrics table, same as a close-up report.

- [ ] **Step 4: Report results**

Note actual wall-clock time for the panorama analysis (the design doc flagged the doubled tile-processing pass as an accepted tradeoff — confirm it's still acceptable in practice, not a blocker) and any visual issues in the Corrector at the ≤2400px editing resolution.
