# Прогресс анализа снимка в интерфейсе — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a real progress bar + current stage + elapsed/ETA time in the UI while a close-up or
panorama analysis job is running, instead of the current static "Анализ снимка…" placeholder.

**Architecture:** Thread an optional `on_progress(progress: float, message: str | None)` callback
down through the existing job runner and pipeline call chain (`JobRunner` → `api/analyze.py::work` →
`closeup.analyze_closeup` / `panorama.analyze_panorama`), reporting at natural checkpoints that
already exist (the panorama tile loop, the close-up ensemble-uncertainty loop, and fixed stage
boundaries). The `JobRecord.progress`/`.message` fields and the `/api/jobs/{jid}` polling contract
are unchanged — only intermediate writes are added. The frontend replaces the static placeholder
with a new `AnalysisProgress` component driven by the already-polled `job.progress`/`job.message`.

**Tech Stack:** Python (FastAPI, pytest), TypeScript/React (Next.js App Router, `node --test` via
`tsx` for pure-logic unit tests).

## Global Constraints

- API contract unchanged: `JobRecord` (`backend/app/schemas/jobs.py`) and `GET /api/jobs/{jid}` keep
  exactly the same fields (`progress: float`, `message: str | None`) — no new schema fields, no
  transport change (stays on the existing 800ms TanStack Query poll, no SSE/WebSocket).
- Every new callback parameter (`on_progress`, `on_step`, `report`) defaults to `None` / is otherwise
  optional, and existing callers/tests that omit it must keep working unchanged.
- UI copy is Russian, matching the existing tone already in `frontend/app/page.tsx` /
  `frontend/components/Welcome.tsx` (e.g. "Анализ снимка…", "крупный план", "панорама").
- Do not modify the topbar `.status-badge` — only the central workspace placeholder
  (`.stage-empty` block in `frontend/app/page.tsx`) changes.
- `cd backend && .venv/bin/pytest -q` and (`cd frontend && npm test` + `npm run build`) must stay
  green after every task.
- No code comments unless documenting a genuinely non-obvious constraint (matches this codebase's
  existing style — see e.g. `backend/app/jobs/runner.py`'s one comment on the exception handler).

---

### Task 0: development environment setup (this worktree has none of it yet)

**Files:** none (environment only — `.venv/`, `node_modules/`, and `backend/models/` are all
gitignored, so this fresh worktree at
`/home/claude/shlif-web/.claude/worktrees/idempotent-wibbling-key` starts with none of them, even
though the main checkout at `/home/claude/shlif-web` already has all three set up).

- [ ] **Step 1: Create the backend venv and install dependencies**

Run:
```bash
cd backend
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
cd ..
```
Expected: completes without error; `backend/.venv/bin/python -c "import app.shlif"` runs with no
output/error afterward.

- [ ] **Step 2: Reuse the trained model weights from the main checkout**

These are large (~350MB), untracked-in-git binaries that already exist in the main checkout and are
identical regardless of branch/worktree — symlink rather than re-download/retrain (there is no
retrain path available here; `README.md`'s Models section is the source of truth on what each file
enables). This only adds a read path inside the worktree; it does not modify the main checkout.

Run:
```bash
rm -rf backend/models   # currently just an empty dir with .gitkeep
ln -s /home/claude/shlif-web/backend/models backend/models
```

Expected: `ls backend/models` lists `classifier.pkl`, `unet_ore.pt`, `unet_talc.pt`, `unet_s2.pt`.
If you'd rather not symlink (e.g. this worktree will outlive the main checkout), skip this step —
every test in this plan that needs a model is already marked
`@pytest.mark.skipif(loader.load_classifier() is None, ...)` and will SKIP cleanly instead of
FAILing; Task 4/6/10's "expected" notes both call this out explicitly.

- [ ] **Step 3: Install frontend dependencies**

Run: `cd frontend && npm install && cd ..`
Expected: completes without error; `cd frontend && npx tsc --noEmit` (or `npm run build`) succeeds
against the pre-existing codebase (i.e. this step alone shouldn't fail — if it does, the environment,
not this plan's later tasks, is the problem).

- [ ] **Step 4: Confirm both suites run (not necessarily all-green yet — this repo's baseline)**

Run: `cd backend && .venv/bin/pytest -q ; cd ../frontend && npm test`
Expected: both commands execute (no "command not found" / import errors); this is the baseline this
plan's tasks build on top of.

No commit for this task — none of its outputs are tracked by git.

---

### Task 1: `count_tiles` — cheap upfront tile-count estimate for panorama progress

**Files:**
- Modify: `backend/app/shlif/tiling.py`
- Test: `backend/tests/test_tiling_count.py` (new)

**Interfaces:**
- Produces: `count_tiles(path: str | Path, cfg) -> int` — same `cfg` shape as `iter_tiles`/`tile_grid`
  (the `tiling` config sub-block: `.max_pixels`, `.tile`, `.overlap`). Used by Task 4.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_tiling_count.py`:

```python
"""count_tiles gives the panorama progress bar a cheap upfront total-tile estimate,
without decoding any tile pixels."""
import numpy as np
from PIL import Image

from app.shlif import tiling
from app.shlif.config import Config


def test_count_tiles_matches_iter_tiles_exactly(tmp_path):
    img = np.zeros((300, 300, 3), np.uint8)
    p = tmp_path / "t.png"
    Image.fromarray(img).save(p, "PNG")
    cfg = Config({"tile": 128, "overlap": 32, "max_pixels": 1_000_000,
                  "skip_empty": False, "empty_bright_frac": 0.002})

    counted = tiling.count_tiles(str(p), cfg)
    actual = sum(1 for _ in tiling.iter_tiles(str(p), cfg))

    assert counted == actual == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_tiling_count.py -v`
Expected: FAIL with `AttributeError: module 'app.shlif.tiling' has no attribute 'count_tiles'`

- [ ] **Step 3: Implement `count_tiles`**

In `backend/app/shlif/tiling.py`, add after `tile_grid`:

```python
def count_tiles(path: str | Path, cfg) -> int:
    """Total tile count `iter_tiles` will yield, without decoding any tile pixels —
    a cheap upfront total for progress reporting. Mirrors iter_tiles's own loop
    bounds; may overcount by a tile or two at the edge (iter_tiles drops slivers
    under 8px), which is fine for a progress estimate."""
    w, h = image_size(path)
    factor = decode_factor(w, h, int(cfg.max_pixels))
    W, H = w // factor, h // factor
    tile = int(cfg.tile)
    step = max(1, tile - int(cfg.overlap))
    n_y = len(range(0, max(1, H - 1), step))
    n_x = len(range(0, max(1, W - 1), step))
    return n_x * n_y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_tiling_count.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/shlif/tiling.py backend/tests/test_tiling_count.py
git commit -m "feat(tiling): add count_tiles for cheap upfront progress totals"
```

---

### Task 2: `ensemble_uncertainty` reports per-perturbation progress

**Files:**
- Modify: `backend/app/shlif/uncertainty.py`
- Test: `backend/tests/test_uncertainty.py`

**Interfaces:**
- Produces: `ensemble_phase_labels(rgb, cfg, perturbations=_PERTURBATIONS, on_step=None) -> np.ndarray`
  and `ensemble_uncertainty(rgb, cfg, conf_thr=0.7, on_step=None) -> dict` — `on_step`, when given,
  is `Callable[[int, int], None]` called after each perturbation with `(completed_count, total_count)`,
  1-indexed. Used by Task 3.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_uncertainty.py`:

```python
def test_ensemble_uncertainty_reports_progress_per_perturbation():
    rgb = np.full((64, 64, 3), 10, np.uint8)
    rgb[16:48, 16:48] = 245
    calls = []
    uncertainty.ensemble_uncertainty(rgb, CFG, on_step=lambda i, total: calls.append((i, total)))

    total = len(uncertainty._PERTURBATIONS)
    assert calls == [(i, total) for i in range(1, total + 1)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_uncertainty.py -v -k progress`
Expected: FAIL with `TypeError: ensemble_uncertainty() got an unexpected keyword argument 'on_step'`

- [ ] **Step 3: Implement the `on_step` hook**

In `backend/app/shlif/uncertainty.py`, replace `ensemble_phase_labels` and `ensemble_uncertainty`:

```python
def ensemble_phase_labels(rgb: np.ndarray, cfg, perturbations=_PERTURBATIONS, on_step=None) -> np.ndarray:
    """Stack of phase-label maps (K, H, W) — one classical segmentation per
    photometric perturbation. `on_step(i, total)`, if given, is called after each
    perturbation completes (1-indexed) — for progress reporting."""
    maps = []
    total = len(perturbations)
    for i, (gamma, gain) in enumerate(perturbations, 1):
        pre = preprocess(_perturb(rgb, gamma, gain), cfg.preprocess)
        maps.append(segment_phases(pre, cfg.segment).labels.astype(np.uint8))
        if on_step:
            on_step(i, total)
    return np.stack(maps)


def ensemble_uncertainty(rgb: np.ndarray, cfg, conf_thr: float = 0.7, on_step=None) -> dict:
    """Run the perturbation ensemble and summarise its disagreement.

    Returns ``confidence`` (HxW float 0..1), ``entropy`` (HxW float 0..1),
    ``low_conf`` (HxW bool — pixels whose modal phase held in fewer than
    ``conf_thr`` of the runs), ``undetermined_fraction`` (scalar) and the
    ensemble ``labels`` stack. ``on_step``, if given, is forwarded to
    ``ensemble_phase_labels`` for progress reporting.
    """
    stack = ensemble_phase_labels(rgb, cfg, on_step=on_step)
    conf = confidence_map(stack)
    low_conf = conf < float(conf_thr)
    return {
        "confidence": conf,
        "entropy": entropy_map(stack),
        "low_conf": low_conf,
        "undetermined_fraction": float(low_conf.mean()),
        "labels": stack,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_uncertainty.py -v`
Expected: PASS (all tests in the file, including the two pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add backend/app/shlif/uncertainty.py backend/tests/test_uncertainty.py
git commit -m "feat(uncertainty): report per-perturbation progress via on_step"
```

---

### Task 3: `analyze_closeup` reports progress through its pipeline stages

> **Plan-drift note (recorded during execution, after Task 1 landed):** this task was originally
> written against a version of `closeup.py` that had a local `_uncertainty` helper. Concurrent work
> that had already merged into `origin/master` by the time this worktree was created (visible via
> `git log -- backend/app/pipeline/closeup.py`: `bcce984 refactor(pipeline): share the
> uncertainty-ensemble helper between closeup and panorama`, part of merge `57f5361`) extracted that
> helper to `masks.uncertainty_for_editor` in `backend/app/pipeline/masks.py`, shared with panorama
> (Task 4). The steps below target the **current** `closeup.py`/`masks.py`; the design intent
> (thread `on_progress` through the existing stage sequence, ensemble sub-progress scaled into
> 0.30→0.75) is unchanged from the original spec. `backend/app/shlif/uncertainty.py` itself
> (Task 2's target) was **not** touched by this drift — Task 2 is unaffected.

**Files:**
- Modify: `backend/app/pipeline/closeup.py`
- Modify: `backend/app/pipeline/masks.py`
- Test: `backend/tests/test_closeup_progress.py` (new)

**Interfaces:**
- Consumes: `ensemble_uncertainty(rgb, cfg, conf_thr=0.7, on_step=None)` from Task 2.
- Produces: `analyze_closeup(rgb: np.ndarray, cfg, on_progress=None) -> dict` — `on_progress`, when
  given, is `Callable[[float, str], None]`, called multiple times with non-decreasing values in
  `[0, 1]` and a Russian stage message. Used by Task 6. Return dict shape is unchanged from today.
  Also produces `masks.uncertainty_for_editor(rgb, cfg, on_step=None) -> dict` — `on_step`, when
  given, is forwarded verbatim to `ensemble_uncertainty`'s `on_step` (no progress-fraction scaling
  inside `masks.py` — it's a shared helper, and panorama (Task 4) calls it too with its own,
  different scaling needs; scaling is each caller's own responsibility).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_closeup_progress.py`:

```python
"""analyze_closeup reports progress through on_progress across its pipeline stages."""
import numpy as np

from app.pipeline import closeup, loader

CFG = loader.get_config()


def test_analyze_closeup_reports_progress():
    rgb = np.full((256, 256, 3), 10, np.uint8)
    rgb[80:176, 80:176] = 245
    calls = []
    closeup.analyze_closeup(rgb, CFG, on_progress=lambda p, msg: calls.append((p, msg)))

    assert len(calls) >= 5
    progresses = [p for p, _ in calls]
    assert progresses == sorted(progresses)
    assert all(0.0 <= p <= 1.0 for p in progresses)

    messages = " ".join(msg for _, msg in calls if msg)
    assert "сегментация" in messages
    assert "неопределённост" in messages
    assert "карт" in messages


def test_analyze_closeup_works_without_on_progress():
    rgb = np.full((64, 64, 3), 10, np.uint8)
    r = closeup.analyze_closeup(rgb, CFG)
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_closeup_progress.py -v`
Expected: FAIL with `TypeError: analyze_closeup() got an unexpected keyword argument 'on_progress'`

- [ ] **Step 3: Add `on_step` passthrough to `masks.uncertainty_for_editor`**

In `backend/app/pipeline/masks.py`, replace the `uncertainty_for_editor` function (it currently ends
the file):

```python
def uncertainty_for_editor(rgb: np.ndarray, cfg, on_step=None) -> dict:
    """Ensemble-perturbation uncertainty, computed on a downscaled copy for
    speed and the confidence map resized back to `rgb`'s own frame. Shared by
    closeup and panorama so both report confidence/low_conf_zones the same way.
    `on_step`, if given, is forwarded verbatim to `ensemble_uncertainty` — this
    function does no progress-fraction scaling itself since callers (closeup,
    panorama) need different scaling for the same shared computation."""
    h, w = rgb.shape[:2]
    s = min(1.0, _UNC_MAX_SIDE / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    u = ensemble_uncertainty(small, cfg, on_step=on_step)
    conf = cv2.resize(u["confidence"], (w, h), interpolation=cv2.INTER_LINEAR)
    return {"confidence": conf, "undetermined_fraction": u["undetermined_fraction"],
            "low_conf_zones": find_low_conf_zones(u)}
```

- [ ] **Step 4: Wire `on_progress` through `closeup.py`**

Replace `backend/app/pipeline/closeup.py`'s `analyze_closeup` function (the whole file's only other
function besides `_sort_card`, which is unchanged):

```python
def analyze_closeup(rgb: np.ndarray, cfg, on_progress=None) -> dict:
    """Uses the trained talc U-Net when its weights are loadable (GPU or CPU);
    falls back to the classical darkness-based talc seed when they aren't."""
    def report(p, msg):
        if on_progress:
            on_progress(p, msg)

    report(0.08, "загрузка модели талька")
    unet = loader.load_talc_unet()
    report(0.15, "сегментация фаз")
    if unet is not None:
        model, device = unet
        talc_mask = talc_unet_mask(rgb, model, device, thr=None)
        res = analyze_image(rgb, cfg, talc_mask=talc_mask)
    else:
        res = analyze_image(rgb, cfg, detect_talc_flag=True)  # classical talc seed
    m = res.masks
    phase_map = masks.phase_label_map(m["sulfide"], m["magnetite"])

    report(0.30, "оценка неопределённости")

    def on_step(i, total):
        if on_progress:
            on_progress(0.30 + 0.45 * (i / total), f"оценка неопределённости ({i}/{total})")

    unc = masks.uncertainty_for_editor(rgb, cfg, on_step=on_step)
    metrics = dict(res.metrics)
    metrics["undetermined_fraction"] = unc["undetermined_fraction"]

    report(0.80, "классификация сорта")
    sort = _sort_card(rgb, cfg)

    report(0.88, "построение карт")
    superpixels = masks.build_superpixel_map(rgb)
    darkness = masks.build_darkness_map(rgb)

    return {
        "verdict": {"ore_class": res.ore_class, "text": res.text, "metrics": metrics},
        "sort": sort,
        "phase_map": phase_map,
        "talc": m["talc"].astype(bool),
        "superpixels": superpixels,
        "darkness": darkness,
        "confidence": unc["confidence"],
        "low_conf_zones": unc["low_conf_zones"],
        "text": res.text,
    }
```

(The 0.30→0.75 ensemble sub-progress scaling now lives in `closeup.py`'s own `on_step` closure,
*not* inside `masks.uncertainty_for_editor` — that function is shared with panorama, which needs a
different scaling for the same call. Do not move this scaling into `masks.py`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_closeup_progress.py tests/test_pipeline.py tests/test_closeup_uncertainty.py tests/test_api_uncertainty.py -v`
Expected: PASS (new tests plus the 3 pre-existing files that exercise `analyze_closeup`)

- [ ] **Step 6: Commit**

```bash
git add backend/app/pipeline/closeup.py backend/app/pipeline/masks.py backend/tests/test_closeup_progress.py
git commit -m "feat(closeup): report progress across pipeline stages"
```

---

### Task 4: `analyze_panorama` reports progress through its two tile loops

> **Plan-drift note (recorded during execution):** this task was originally written against a much
> simpler `panorama.py` (one tile loop, `_run_panorama` returning the verdict directly, a
> `display_mp`-based single decode). Concurrent work already merged into `origin/master` before this
> worktree was created (`git log -- backend/app/pipeline/panorama.py`: merge `57f5361`, plus
> `0565bd3 feat(panorama): verdict + sort + editable artifacts now match closeup's shape exactly` and
> `ee691d8 feat(panorama): assemble one whole-canvas phase/talc mask via core-crop tiling`)
> restructured the panorama pipeline into **two** tile loops — `_assemble_masks` (builds the
> whole-canvas phase/talc masks that drive the reported verdict, via `verdict_from_masks_dict`) and
> `_run_panorama` (classifies ore tiles for the sort card, runs per-tile ensemble uncertainty, and
> stitches the display overlay) — plus a single extra `masks.uncertainty_for_editor` pass at the end
> for the editor's confidence layer. The design intent is unchanged (thread `on_progress` through
> every loop that already exists, using `count_tiles` from Task 1 for the shared tile-count
> denominator); the steps below target the **current** `panorama.py`.

**Files:**
- Modify: `backend/app/pipeline/panorama.py`
- Test: `backend/tests/test_panorama_progress.py` (new)

**Interfaces:**
- Consumes: `count_tiles(path, cfg) -> int` from Task 1; `masks.uncertainty_for_editor(rgb, cfg,
  on_step=None)` from Task 3.
- Produces: `_assemble_masks(path, cfg, arr, on_progress=None) -> dict`, `_run_panorama(path, clf,
  feat_names, classes, cfg, arr, min_ore=0.04, on_progress=None) -> dict`, and
  `analyze_panorama(path: str, cfg, jid: str, on_progress=None) -> dict` — `on_progress`, when given,
  is `Callable[[float, str], None]`, called multiple times with non-decreasing values in `[0, 1]`.
  Used by Task 6. Return dict shapes unchanged from today.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_panorama_progress.py`:

```python
"""analyze_panorama reports progress through on_progress across both tile loops
(_assemble_masks, _run_panorama) and the tail stages."""
import numpy as np
import pytest
from PIL import Image

from app.pipeline import panorama, loader


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_reports_progress(tmp_path):
    img = (np.random.default_rng(5).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"
    Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()
    calls = []

    r = panorama.analyze_panorama(str(p), cfg, "progresstest",
                                   on_progress=lambda pr, msg: calls.append((pr, msg)))

    assert r["mode"] == "panorama"
    assert len(calls) >= 5
    progresses = [pr for pr, _ in calls]
    assert progresses == sorted(progresses)
    assert all(0.0 <= pr <= 1.0 for pr in progresses)
    messages = " ".join(msg for _, msg in calls if msg)
    assert "сборка масок" in messages
    assert "сегментация тайлов" in messages
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_panorama_progress.py -v`
Expected: FAIL with `TypeError: analyze_panorama() got an unexpected keyword argument 'on_progress'`
(or SKIPPED if `models/classifier.pkl` isn't present in this environment — if so, still apply Step 3
and confirm via Step 4 that it's collected correctly)

- [ ] **Step 3: Wire `on_progress` through `panorama.py`**

In `backend/app/pipeline/panorama.py`, update the `tiling` import line to add `count_tiles`:

```python
from app.shlif.tiling import axis_core_bounds, count_tiles, iter_tiles, load_working_array, tile_blend_weight, tile_grid
```

Replace `_assemble_masks`:

```python
def _assemble_masks(path: str, cfg, arr: np.ndarray, on_progress=None) -> dict:
    """Tile the section, segment + talc-detect each tile, and reassemble one
    continuous mask set for the whole working canvas — core-crop (no overlap
    double count, see `axis_core_bounds`) — so `verdict_from_masks` sees the
    same kind of input it gets from a single close-up pass. Classical
    segmentation only (see module docstring). `on_progress(progress, message)`,
    if given, is called once per tile, scaled into the 0.05-0.35 job-progress
    range (this is the first of panorama's two tile loops)."""
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
    total = max(1, count_tiles(path, cfg.tiling))
    n = 0

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        n += 1
        if on_progress:
            on_progress(0.05 + 0.30 * min(1.0, n / total), f"сборка масок ({n}/{total})")
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

Replace `_run_panorama` (only the signature, the `total_tiles_est` line, and the `on_progress` call
inside the tile loop are new — every other line is unchanged from today):

```python
def _run_panorama(path, clf, feat_names, classes, cfg, arr: np.ndarray, min_ore: float = 0.04,
                  on_progress=None) -> dict:
    """Tile a panorama, classify ore-rich tiles for the `sort` card (ore-density
    weighted aggregation — unchanged mechanism, see design spec §4.2), estimate
    per-tile ensemble uncertainty, and stitch the display overlay. Matrix
    segmentation uses the trained ore/matrix U-Net when available (IoU 0.975 vs
    classical 0.81), falling back to classical segmentation otherwise; talc
    similarly prefers the trained talc U-Net over the classical detector. The
    whole-image phase/talc masks and the `ore_class` verdict come from
    `_assemble_masks` + `verdict_from_masks` instead (design spec §4.1) — this
    function no longer decides ore_class. `on_progress(progress, message)`, if
    given, is called once per tile, scaled into the 0.35-0.85 job-progress
    range (this is the second of panorama's two tile loops, and the more
    expensive one — it runs a 5-perturbation ensemble per tile)."""
    unet = loader.load_talc_unet()
    ore_bundle = loader.load_ore_unet()
    ore_source = "unet" if ore_bundle is not None else "classical"

    Wt, Ht, factor = tile_grid(path, cfg.tiling)
    edit = masks.fit_max_side(arr, masks.EDIT_MAX_SIDE, cv2.INTER_AREA)
    dh, dw = edit.shape[:2]
    rx, ry = dw / Wt, dh / Ht
    ore_pct = float(getattr(cfg.tiling, "ore_density_pct", ORE_DENSITY_PCT))
    bright_thr = float(np.percentile(cv2.cvtColor(edit, cv2.COLOR_RGB2GRAY), ore_pct))

    base = edit.astype(np.float32)
    color_num = np.zeros((dh, dw, 3), np.float32)
    weight_den = np.zeros((dh, dw), np.float32)
    talc_disp = np.zeros((dh, dw), bool)
    records = []
    low_conf_zones = []
    undet_weighted_sum = 0.0
    undet_px_total = 0
    n_tiles = n_ore = n_matrix = 0
    t0 = time.time()
    sort_alpha = 0.32
    total_tiles_est = max(1, count_tiles(path, cfg.tiling))

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        n_tiles += 1
        if on_progress:
            on_progress(0.35 + 0.50 * min(1.0, n_tiles / total_tiles_est),
                        f"сегментация тайлов ({n_tiles}/{total_tiles_est})")
        if tile.empty:
            continue
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        if ore_bundle is not None:
            ore_model, ore_device = ore_bundle
            matrix = ~ore_unet_mask(rgb, ore_model, ore_device)
        else:
            matrix = segment_phases(pre, cfg.segment).labels == phases.MATRIX
        if unet is not None:
            model, device = unet
            talc = talc_unet_mask(rgb, model, device, thr=None) & matrix
        else:
            talc = detect_talc(pre, matrix, cfg.talc)
        ore_px = int((~matrix).sum())
        ore_frac = ore_px / max(matrix.size, 1)

        dx0, dy0 = int(tile.x * rx), int(tile.y * ry)
        dx1, dy1 = min(int((tile.x + rgb.shape[1]) * rx), dw), min(int((tile.y + rgb.shape[0]) * ry), dh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue

        th, tw = rgb.shape[:2]
        unc_scale = min(1.0, _UNC_MAX_SIDE / max(th, tw))
        unc_rgb = (cv2.resize(rgb, (int(tw * unc_scale), int(th * unc_scale)),
                              interpolation=cv2.INTER_AREA) if unc_scale < 1 else rgb)
        unc = ensemble_uncertainty(unc_rgb, cfg)
        undet_weighted_sum += unc["undetermined_fraction"] * (th * tw)
        undet_px_total += th * tw
        bx, by = rx / unc_scale, ry / unc_scale
        for z in find_low_conf_zones(unc):
            zx, zy, zw, zh = z["bbox"]
            low_conf_zones.append({
                "bbox": [int(dx0 + zx * bx), int(dy0 + zy * by), int(zw * bx), int(zh * by)],
                "area": z["area"], "phase_a": z["phase_a"], "phase_b": z["phase_b"],
            })

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
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
        "ore_source": ore_source,
    }
```

Replace `analyze_panorama`:

```python
def analyze_panorama(path: str, cfg, jid: str, on_progress=None) -> dict:
    """Public wrapper called by the API. Builds the whole-canvas phase/talc
    masks (design spec §4) and reuses `verdict_from_masks_dict` — the exact
    helper close-up uses — so the result has the same shape and the same
    meaning, computed over the whole image instead of per tile."""
    def report(p, msg):
        if on_progress:
            on_progress(p, msg)

    cfg = copy.deepcopy(cfg)  # don't mutate the shared @lru_cache'd Config
    cfg.tiling.tile = 2048
    cfg.talc.detect_dark_frac = 0.15
    bundle = loader.load_classifier()
    if bundle is None:
        raise RuntimeError("classifier.pkl required for panorama sort")
    clf, feat, classes = bundle

    report(0.03, "загрузка изображения")
    arr = load_working_array(path, cfg.tiling)
    H, W = arr.shape[:2]

    assembled = _assemble_masks(path, cfg, arr, on_progress=on_progress)
    report(0.35, "вердикт по фазам")
    verdict = masks.verdict_from_masks_dict(
        assembled["sulfide"], assembled["magnetite"], assembled["matrix"], assembled["talc"], cfg)
    verdict["metrics"]["talc_share_est"] = float(assembled["dg"].mean())

    run = _run_panorama(path, clf, feat, classes, cfg, arr, on_progress=on_progress)
    verdict["metrics"]["undetermined_fraction"] = run["undetermined_fraction"]
    report(0.85, "сохранение оверлея")
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
    report(0.88, "карта уверенности для редактора")
    unc = masks.uncertainty_for_editor(edit, cfg)

    report(0.93, "построение карт")
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
        "low_conf_zones": run["low_conf_zones"],
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": run["n_ore"], "n_tiles": run["n_tiles"],
        "ore_source": run["ore_source"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_panorama_progress.py tests/test_panorama.py tests/test_panorama_assemble.py tests/test_panorama_aggregate.py tests/test_panorama_uncertainty.py tests/test_panorama_unet_gate.py -v`
Expected: PASS (or SKIPPED for the classifier-gated tests if `models/classifier.pkl` is absent —
either way, no FAIL/ERROR)

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama_progress.py
git commit -m "feat(panorama): report progress through both tile loops"
```

---

### Task 5: `JobRunner` passes a `report` callback into the submitted job

**Files:**
- Modify: `backend/app/jobs/runner.py`
- Test: `backend/tests/test_jobs.py`

**Interfaces:**
- Produces: `JobRunner.submit(jid: str, fn: Callable[[Callable[[float, str | None], None]], dict]) -> None`
  — `fn` now takes one argument, `report(progress: float, message: str | None = None) -> None`, which
  writes straight to the store as `status="running"`. Used by Task 6.

- [ ] **Step 1: Update the two existing tests for the new `fn` signature, and add a progress test**

Replace `backend/tests/test_jobs.py`'s first two tests and add a third, leaving
`test_log_correction_inserts_row` untouched:

```python
import sqlite3
import time
from app.jobs.store import JobStore
from app.jobs.runner import JobRunner

def test_job_lifecycle_success(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")
    assert store.get(jid).status == "queued"
    runner.submit(jid, lambda report: {"ore_class": "ordinary"})
    for _ in range(50):
        if store.get(jid).status == "done": break
        time.sleep(0.05)
    rec = store.get(jid)
    assert rec.status == "done" and rec.result == {"ore_class": "ordinary"}

def test_job_lifecycle_error(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")
    def boom(report): raise ValueError("nope")
    runner.submit(jid, boom)
    for _ in range(50):
        if store.get(jid).status == "error": break
        time.sleep(0.05)
    rec = store.get(jid)
    assert rec.status == "error" and "nope" in rec.message

def test_job_reports_intermediate_progress_before_done(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")

    def fn(report):
        report(0.5, "halfway")
        rec = store.get(jid)
        assert rec.status == "running"
        assert rec.progress == 0.5
        assert rec.message == "halfway"
        return {"ore_class": "ordinary"}

    runner.submit(jid, fn)
    for _ in range(50):
        if store.get(jid).status == "done": break
        time.sleep(0.05)
    assert store.get(jid).status == "done"

def test_log_correction_inserts_row(tmp_path):
    db_path = tmp_path / "t.db"
    store = JobStore(db_path)
    jid = store.create("closeup")
    store.log_correction(jid, "talc", 42)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT job_id, layer, n_pixels FROM corrections WHERE job_id=?", (jid,)
        ).fetchone()
    finally:
        conn.close()

    assert row == (jid, "talc", 42)
```

- [ ] **Step 2: Run tests to verify the new/changed ones fail**

Run: `cd backend && .venv/bin/pytest tests/test_jobs.py -v`
Expected: `test_job_lifecycle_success` and `test_job_lifecycle_error` FAIL with
`TypeError: <lambda>() takes 0 positional arguments but 1 was given` (or similar for `boom`);
`test_job_reports_intermediate_progress_before_done` FAILs likewise once `_run` starts passing an
argument — confirm all three fail against the *old* `runner.py` before implementing.

- [ ] **Step 3: Implement `report` in `JobRunner`**

Replace `backend/app/jobs/runner.py`:

```python
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from app.jobs.store import JobStore

class JobRunner:
    def __init__(self, store: JobStore, max_workers: int = 1):
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, jid: str, fn: Callable[[Callable[[float, str | None], None]], dict]) -> None:
        self._store.set_status(jid, "running", progress=0.05)
        self._pool.submit(self._run, jid, fn)

    def _run(self, jid: str, fn: Callable[[Callable[[float, str | None], None]], dict]) -> None:
        def report(progress: float, message: str | None = None) -> None:
            self._store.set_status(jid, "running", progress=progress, message=message)
        try:
            result = fn(report)
            self._store.set_result(jid, result)
            self._store.set_status(jid, "done", progress=1.0)
        except Exception as e:  # noqa: BLE001 — surfaced to the client as status=error
            self._store.set_status(jid, "error", message=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_jobs.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/jobs/runner.py backend/tests/test_jobs.py
git commit -m "feat(jobs): pass a report callback into submitted work"
```

---

### Task 6: wire `report` through the `/api/analyze` endpoint

> **Plan-drift note (recorded during execution):** this task was originally written against a
> version of the endpoint that took a client-supplied `mode: str = Form("closeup")` field. Concurrent
> work already merged into `origin/master` before this worktree was created (`git log -- 
> backend/app/api/analyze.py`: `22f9e63 feat(api): auto-detect closeup/panorama mode server-side, no
> client mode field`) replaced that with server-side detection via `app.pipeline.detect.detect_mode`,
> and relocated the close-up thumbnail budget/artifact-persist helpers to `masks.EDIT_MAX_SIDE`/
> `masks.persist_editor_artifacts` (this module no longer has a local `_persist_maps`). The design
> intent is unchanged (thread `report` into `work()`, call it before/after the two expensive
> closeup steps); the steps below target the **current** `analyze.py` and `test_api.py` (which
> already dropped the `mode` form field — see `git log`: `4166fc3 test: drop the client-supplied mode
> field now that the API auto-detects it`).

**Files:**
- Modify: `backend/app/api/analyze.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `JobRunner.submit(jid, fn)` with `fn(report)` from Task 5; `closeup.analyze_closeup(rgb,
  cfg, on_progress=report)` from Task 3; `panorama.analyze_panorama(path, cfg, jid,
  on_progress=report)` from Task 4.
- Produces: no new interface — `POST /api/analyze` response shape (`{"job_id": str}`) is unchanged.

- [ ] **Step 1: Strengthen the existing API test**

In `backend/tests/test_api.py`, add one assertion to `test_closeup_analyze_and_edit` right after
`assert done["status"] == "done"`:

```python
def test_closeup_analyze_and_edit(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze",
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    assert up.status_code == 200
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["status"] == "done"
    assert done["progress"] == 1.0
    assert done["result"]["mode"] == "closeup"  # tiny_rgb (256x256) is well under direct_max_pixels
    assert done["result"]["verdict"]["ore_class"] in {"ordinary","hard","talcose","review"}

    # layers + maps are fetchable
    assert c.get(f"/api/masks/{jid}/phases.png").status_code == 200
    assert c.get(f"/api/maps/{jid}/superpixels.png").status_code == 200
    assert c.get(f"/api/maps/{jid}/darkness.png").status_code == 200

    # edit: mark everything talc → verdict recomputes to talcose
    h, w = tiny_rgb.shape[:2]
    all_talc = np.full((h, w), 255, np.uint8)
    phases_png = c.get(f"/api/masks/{jid}/phases.png").content
    r = c.post(f"/api/masks/{jid}",
               files={"talc": ("talc.png", _png_bytes(all_talc), "image/png"),
                      "phases": ("phases.png", phases_png, "image/png")})
    assert r.status_code == 200
    assert r.json()["ore_class"] == "talcose"
```

(Only the two new lines — `assert done["progress"] == 1.0` and the `mode` assertion's comment
context — are new; everything else in this test is unchanged from today.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_api.py -v`
Expected: FAIL — `work()` in `analyze.py` still takes zero arguments, so `JobRunner._run` (from Task
5) calling `fn(report)` raises `TypeError`, which the runner catches and turns into `status="error"`;
the test then fails on `assert done["status"] == "done"`.

- [ ] **Step 3: Wire `report` through `work()`**

Replace `backend/app/api/analyze.py`'s `analyze` route body (the `Image.MAX_IMAGE_PIXELS = None` line
and the imports above it are unchanged):

```python
@router.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    cfg = loader.get_config()
    iw, ih = Image.open(io.BytesIO(data)).size
    mode = detect.detect_mode(iw, ih, cfg)
    jid = get_runtime().store.create(mode)
    up = paths.uploads_dir() / f"{jid}_{Path(image.filename or 'up').name}"
    up.write_bytes(data)

    def work(report):
        if mode == "panorama":
            return panorama.analyze_panorama(str(up), cfg, jid, on_progress=report)
        report(0.05, "загрузка изображения")
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((masks.EDIT_MAX_SIDE, masks.EDIT_MAX_SIDE))
        rgb = np.asarray(im)
        r = closeup.analyze_closeup(rgb, cfg, on_progress=report)
        report(0.95, "сохранение результатов")
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

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all tests PASS or SKIPPED (classifier/model-gated tests skip cleanly if
`backend/models/*.pkl`/`*.pt` aren't present) — no FAIL/ERROR.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/analyze.py backend/tests/test_api.py
git commit -m "feat(api): report progress through /api/analyze"
```

---

### Task 7: frontend pure progress-math helpers

**Files:**
- Create: `frontend/lib/progress.ts`
- Test: `frontend/tests/progress.test.mjs` (new)

**Interfaces:**
- Produces: `clampPct(progress: number): number`, `formatDuration(seconds: number): string`,
  `computeEta(elapsedSec: number, progress: number): number | null`. Used by Task 8.

- [ ] **Step 1: Write the failing test**

Create `frontend/tests/progress.test.mjs`:

```javascript
import { test } from "node:test";
import assert from "node:assert";
import { clampPct, formatDuration, computeEta } from "../lib/progress.ts";

test("clampPct rounds to a 0-100 integer and clamps out-of-range input", () => {
  assert.strictEqual(clampPct(0), 0);
  assert.strictEqual(clampPct(1), 100);
  assert.strictEqual(clampPct(0.421), 42);
  assert.strictEqual(clampPct(-0.5), 0);
  assert.strictEqual(clampPct(1.5), 100);
});

test("formatDuration formats seconds and minutes in Russian", () => {
  assert.strictEqual(formatDuration(14), "14 с");
  assert.strictEqual(formatDuration(59), "59 с");
  assert.strictEqual(formatDuration(60), "1 мин 0 с");
  assert.strictEqual(formatDuration(92), "1 мин 32 с");
});

test("computeEta is null below the 8% noise floor and extrapolates linearly above it", () => {
  assert.strictEqual(computeEta(5, 0.05), null);
  assert.strictEqual(computeEta(0, 0), null);
  assert.strictEqual(computeEta(10, 0.5), 10);
  assert.strictEqual(computeEta(20, 1), 0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL — `Cannot find module '../lib/progress.ts'` (or similar module-resolution error)

- [ ] **Step 3: Implement `frontend/lib/progress.ts`**

```typescript
export function clampPct(progress: number): number {
  const clamped = Math.min(1, Math.max(0, progress));
  return Math.round(clamped * 100);
}

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} с`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m} мин ${rem} с`;
}

export function computeEta(elapsedSec: number, progress: number): number | null {
  if (progress < 0.08) return null;
  const total = elapsedSec / progress;
  return Math.max(0, total - elapsedSec);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: PASS (all tests across all `tests/*.test.mjs` files)

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/progress.ts frontend/tests/progress.test.mjs
git commit -m "feat(frontend): add pure progress/ETA formatting helpers"
```

---

### Task 8: `AnalysisProgress` component + styles

**Files:**
- Create: `frontend/components/AnalysisProgress.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Consumes: `clampPct`, `formatDuration`, `computeEta` from Task 7; `Job` type from
  `frontend/lib/api/types.ts` (already has `progress: number; message: string | null`).
- Produces: `AnalysisProgress({ job, startedAt, fallback }: { job?: Job; startedAt: number; fallback:
  string }): JSX.Element`. Used by Task 9.

- [ ] **Step 1: Add progress-bar styles**

In `frontend/app/globals.css`, add immediately after the `.stage-empty` block (after the line
`.stage-empty .sub { font-family: var(--font-mono); font-size: 11.5px; color: oklch(56% 0.014 258); }`):

```css
.progress-track { width: 100%; max-width: 340px; height: 6px; background: var(--surface-2); border-radius: 999px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--brand); border-radius: 999px; transition: width .3s ease; }
.progress-meta { display: flex; gap: 10px; font-family: var(--font-mono); font-size: 11.5px; color: var(--muted); }
```

- [ ] **Step 2: Create the component**

Create `frontend/components/AnalysisProgress.tsx`:

```tsx
"use client";
import { useEffect, useState } from "react";
import type { Job } from "@/lib/api/types";
import { clampPct, computeEta, formatDuration } from "@/lib/progress";

export function AnalysisProgress({ job, startedAt, fallback }: { job?: Job; startedAt: number; fallback: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, []);

  const progress = job?.progress ?? 0;
  const pct = clampPct(progress);
  const elapsedSec = Math.max(0, (now - startedAt) / 1000);
  const etaSec = computeEta(elapsedSec, progress);

  return (
    <div className="stage-empty">
      <div className="hint">Анализ снимка…</div>
      <div className="sub">{job?.message || fallback}</div>
      <div className="progress-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="progress-meta">
        <span>{pct}%</span>
        <span>{formatDuration(elapsedSec)}{etaSec != null ? ` · осталось ≈ ${formatDuration(etaSec)}` : ""}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Type-check**

Run: `cd frontend && npm run build`
Expected: build succeeds (this also type-checks `AnalysisProgress.tsx` even though it isn't wired
into `page.tsx` yet — an unused exported component is not a type error)

- [ ] **Step 4: Commit**

```bash
git add frontend/components/AnalysisProgress.tsx frontend/app/globals.css
git commit -m "feat(frontend): add AnalysisProgress component"
```

---

### Task 9: wire `AnalysisProgress` into the workspace placeholder

> **Plan-drift note (recorded during execution):** this task was originally written against a
> version of `page.tsx` with a manual closeup/panorama mode toggle and a separate
> `PanoramaWorkspace` view. Concurrent work already merged into `origin/master` before this worktree
> was created (`git log -- frontend/app/page.tsx`: `0f0add3 feat(frontend): remove the mode toggle,
> edit panorama masks like closeup`) removed the toggle entirely — mode is now auto-detected
> server-side (Task 6's `detect.detect_mode`) and both closeup and panorama results render through
> the same `Corrector` component (`PanoramaWorkspace.tsx` is no longer imported anywhere — confirmed
> via `grep -rn "PanoramaWorkspace" frontend/app frontend/components`, zero matches). The design
> intent is unchanged (swap the static placeholder for `AnalysisProgress`, track `startedAt`); the
> steps below target the **current** `page.tsx`, which is simpler than originally planned (no
> mode-conditional branching needed anywhere in this task).

**Files:**
- Modify: `frontend/app/page.tsx`

**Interfaces:**
- Consumes: `AnalysisProgress` from Task 8.
- Produces: no new interface — this task only changes what's rendered inside `Home()`.

- [ ] **Step 1: Add `startedAt` state and set it in `runAnalyze`**

In `frontend/app/page.tsx`, add the import alongside the existing component imports:

```tsx
import { AnalysisProgress } from "@/components/AnalysisProgress";
```

Add `startedAt` state and set it in `runAnalyze`:

```tsx
export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function runAnalyze(f: File) {
    setFile(f);
    setVOverride(null);
    setJobId(null);
    setStartedAt(Date.now());
    analyze.mutate({ file: f }, { onSuccess: (r) => setJobId(r.job_id) });
  }
```

- [ ] **Step 2: Replace the static placeholder with `AnalysisProgress`**

Replace the `stage-empty` block currently at the end of `Home()` (the non-error branch of the
`job.data?.status === "error"` ternary inside `.zoom-vp`):

```tsx
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
                <AnalysisProgress
                  job={job.data}
                  startedAt={startedAt ?? Date.now()}
                  fallback={jobId ? "сегментация фаз" : "загрузка файла на сервер"}
                />
              )}
            </div>
          </div>
        </div>
      )}
```

(Mode isn't known client-side until the job resolves — the server auto-detects it — so, unlike the
original plan, there's no mode-conditional fallback text here; `"сегментация фаз"` matches today's
existing hardcoded fallback copy.)

- [ ] **Step 3: Type-check and run frontend tests**

Run: `cd frontend && npm run build && npm test`
Expected: both succeed

- [ ] **Step 4: Manual verification**

Run: `cd frontend && npm run dev` (and, in another terminal, the backend per README's "Local
development" section, or `docker compose up -d --build` for the full stack)
Then in a browser: upload a close-up image, and separately a panorama image, and confirm the
workspace placeholder now shows a moving progress bar, a changing stage message (e.g. "сегментация
фаз" → "оценка неопределённости (2/5)" → …), a percentage, and an elapsed-time counter that ticks —
instead of the old static "Анализ снимка…" text. Confirm the error branch (e.g. temporarily stop the
backend mid-upload) still shows the existing red error card unchanged.

- [ ] **Step 5: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat(frontend): show live progress bar during analysis"
```

---

### Task 10: full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all PASS/SKIP, no FAIL/ERROR (skips are fine if `backend/models/*.pkl`/`*.pt` aren't
present in this environment)

- [ ] **Step 2: Run the full frontend suite + production build**

Run: `cd frontend && npm test && npm run build`
Expected: both succeed

- [ ] **Step 3: Confirm git log tells a coherent story**

Run: `git log --oneline -12`
Expected: one commit per task (1 through 9), each with a `feat(...)`-scoped message matching the
work in this plan

No commit needed for this task — it's a read-only gate before considering the feature done.
