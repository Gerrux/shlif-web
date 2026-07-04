# Panorama Inference Perf (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut single-panorama analysis latency on the prod single-GPU (L4) VM by fixing three concrete resource-underutilisation spots in the tiled inference path — batch=1 GPU calls, no mixed precision, and a fully sequential CPU-bound classical ensemble — without changing any external contract or existing test.

**Architecture:** No structural change to `panorama.py`'s tile loops. Only the *internals* of `ore_unet.py::ore_unet_mask`, `ore_unet.py`/`talc_unet.py`'s forward passes, and `uncertainty.py::ensemble_phase_labels` change. See `docs/superpowers/specs/2026-07-04-panorama-inference-perf-design.md` for the full design rationale.

**Tech Stack:** Python 3.12, PyTorch 2.5.1 (cu121, GPU-optional — see Global Constraints), OpenCV, NumPy, pytest.

## Global Constraints

- Preserve every existing external contract: `ore_unet_mask(rgb, model, device, tile=512)` gains only a new *optional* `batch_size` parameter (default 32) — the call site in `panorama.py` (`~ore_unet_mask(rgb, ore_model, ore_device)`) is untouched. `ensemble_phase_labels(rgb, cfg, perturbations, on_step)` keeps its exact signature and return shape.
- `ensemble_phase_labels`'s `on_step(i, total)` callback MUST keep firing in exact order `(1,total), (2,total), ..., (total,total)` — pinned down by the existing test `backend/tests/test_uncertainty.py::test_ensemble_uncertainty_reports_progress_per_perturbation`. Do not use `concurrent.futures.as_completed` for progress reporting (its order is nondeterministic under threading).
- The CPU fallback path (`device == "cpu"`, or `build_ore_unet`/`build_talc_unet` returning `None`) must stay byte-for-byte the same control flow as today — autocast/TF32 only ever engage on a CUDA device string.
- This is a perf-only change: no verdict/classification behaviour may change. Batching does not reorder which crop maps to which output pixel; fp16 autocast may introduce tiny floating-point drift versus fp32, which is expected and acceptable.
- **Test environment:** this sandbox has no `torch` installed — it's an optional, GPU-flavoured dependency (`torch==2.5.1+cu121` in `backend/pyproject.toml`, only resolvable via the `download.pytorch.org/whl/cu121` index used at deploy time, per the comment above that dependency). A working venv with every *other* project dependency already exists at `backend/.venv` (created via `uv venv .venv && uv pip install --python .venv/bin/python <base deps, see below> && uv pip install --python .venv/bin/python -e . --no-deps`). All commands below assume `cd backend && .venv/bin/python -m pytest ...`. If `backend/.venv` doesn't exist when you start, recreate it with:
  ```bash
  cd backend
  uv venv .venv --python 3.12
  uv pip install --python .venv/bin/python \
    fastapi granian python-multipart pydantic pydantic-settings numpy pillow \
    opencv-python-headless scikit-image scikit-learn scipy pandas pyyaml reportlab \
    pytest httpx
  uv pip install --python .venv/bin/python -e . --no-deps
  ```
  Tests that need `torch` (Task 1's batching tests, and the forward-pass half of Task 2) use `pytest.importorskip("torch")` and will show as **skipped** here — same as the existing torch-dependent tests already do in this sandbox. They must be additionally verified on a machine with `torch` installed (a CPU-only `pip install torch` build is enough for the batching-mechanics assertions; the fp16-autocast behaviour itself can only be truly exercised on real CUDA hardware — the prod L4 VM).
- Baseline before any change in this plan: `cd backend && .venv/bin/python -m pytest -q` → `68 passed, 8 skipped` (the 8 skips are the existing `classifier.pkl`/torch-gated tests — unrelated to this work).

---

### Task 1: Batch the ore U-Net's under-tile crops

**Files:**
- Modify: `backend/app/shlif/ore_unet.py:47-71` (the `ore_unet_mask` function)
- Test: `backend/tests/test_ore_unet.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ore_unet_mask(rgb: np.ndarray, model, device: str, tile: int = 512, batch_size: int = 32) -> np.ndarray` — same bool `(H, W)` return as before. Task 2 wraps this function's forward-pass call in autocast.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_ore_unet.py` (keep the two existing tests in the file as-is; add everything below):

```python
import pytest


class _CountingOreModel:
    """Stand-in for the real U-Net: records the batch size of every forward
    call. Flags class 1 (ore) per-sample based on that sample's own mean
    value (positive vs. negative after ImageNet normalisation) -- content
    dependent, independent of what else shares the batch, exactly how a real
    batched CNN behaves (each sample is processed independently of its
    batch-mates)."""
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, batch):
        import torch
        self.batch_sizes.append(batch.shape[0])
        n, c, h, w = batch.shape
        ore = (batch.mean(dim=(1, 2, 3)) > 0).float().view(n, 1, 1)
        out = torch.zeros(n, 2, h, w)
        out[:, 1] = ore
        out[:, 0] = 1 - ore
        return out


def _quadrant_tile(bright_row, bright_col):
    """1024x1024 tile split into four 512x512 quadrants; (bright_row,
    bright_col) in {0,1}x{0,1} is filled 240 (bright), the rest 10 (dark)."""
    rgb = np.full((1024, 1024, 3), 10, np.uint8)
    rgb[bright_row * 512:(bright_row + 1) * 512,
        bright_col * 512:(bright_col + 1) * 512] = 240
    return rgb


@pytest.fixture(autouse=True)
def _no_clahe(monkeypatch):
    # CLAHE's behaviour on a perfectly flat crop isn't the point of this
    # test -- stub it to identity so only the batching logic is exercised.
    monkeypatch.setattr("app.shlif.preprocess.wb_clahe", lambda rgb, *a, **k: rgb)


def test_batches_all_crops_into_one_forward_pass_when_within_batch_size():
    pytest.importorskip("torch")
    from app.shlif.ore_unet import ore_unet_mask

    rgb = _quadrant_tile(0, 1)  # top-right quadrant bright
    model = _CountingOreModel()

    mask = ore_unet_mask(rgb, model, "cpu", tile=512, batch_size=32)

    assert model.batch_sizes == [4], model.batch_sizes
    assert mask[0:512, 512:1024].all()
    assert not mask[0:512, 0:512].any()
    assert not mask[512:1024, :].any()


def test_chunks_when_more_crops_than_batch_size():
    pytest.importorskip("torch")
    from app.shlif.ore_unet import ore_unet_mask

    rgb = _quadrant_tile(1, 0)  # bottom-left quadrant bright
    model = _CountingOreModel()

    mask = ore_unet_mask(rgb, model, "cpu", tile=512, batch_size=2)

    assert model.batch_sizes == [2, 2], model.batch_sizes
    assert mask[512:1024, 0:512].all()
    assert not mask[0:512, :].any()
    assert not mask[512:1024, 512:1024].any()
```

Also add `import numpy as np` at the top of the file if not already present (it currently is not — the existing two tests don't use numpy directly).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_ore_unet.py -v`
Expected (in this sandbox, no torch): the two new tests **SKIP** (`pytest.importorskip("torch")`), the two existing guard tests still PASS. This is expected here — to see a real RED, run the same command on a machine with `torch` installed: expected FAIL with `TypeError: ore_unet_mask() got an unexpected keyword argument 'batch_size'` (current signature has no `batch_size` param).

- [ ] **Step 3: Implement batching**

Replace the current `ore_unet_mask` function body (`backend/app/shlif/ore_unet.py:47-71`) with:

```python
def ore_unet_mask(rgb: np.ndarray, model, device: str, tile: int = 512,
                   batch_size: int = 32) -> np.ndarray:
    """Bool (H, W): True = ore (sulfide+magnetite), tiled U-Net inference.

    Applies gray-world WB + CLAHE per sub-tile before the ImageNet
    normalisation -- IDENTICAL to training (``wb_clahe``). This MUST stay on
    for this checkpoint (unlike the talc U-Net, which trained on raw RGB).

    All under-tile crops are stacked into as few forward passes as
    ``batch_size`` allows (default 32 -- a typical 2048px panorama tile at
    tile=512 is 16 crops, comfortably one batch), instead of one model call
    per crop.
    """
    import torch

    from .preprocess import wb_clahe

    H, W = rgb.shape[:2]
    ore = np.zeros((H, W), bool)

    coords, dims, crops = [], [], []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            crop = rgb[y:y + tile, x:x + tile]
            ch, cw = crop.shape[:2]
            cp = cv2.copyMakeBorder(crop, 0, tile - ch, 0, tile - cw, cv2.BORDER_REFLECT)
            cp = wb_clahe(cp)
            t = ((cp.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)
            coords.append((y, x))
            dims.append((ch, cw))
            crops.append(t)

    if not crops:
        return ore

    batch = torch.from_numpy(np.stack(crops)).to(device)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            chunk = batch[start:start + batch_size]
            preds = model(chunk).argmax(1).cpu().numpy()
            for i, p in enumerate(preds):
                (y, x), (ch, cw) = coords[start + i], dims[start + i]
                ore[y:y + ch, x:x + cw] = (p[:ch, :cw] != 0)
    return ore
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_ore_unet.py -v`
Expected: 2 PASS (existing guard tests), 2 SKIPPED (torch absent in this sandbox — re-run this same command on a machine with torch installed and expect 4 PASS there).

Run the full suite to confirm no regression: `cd backend && .venv/bin/python -m pytest -q`
Expected: `68 passed, 8 skipped` (same baseline as before — this task adds 2 more skips in a torch-less environment, or 2 more passes where torch is installed; if you're in this sandbox you should still see the original 8 skips, now alongside 2 new ones from this file — check the count matches `70 passed` on a torch-enabled machine, `68 passed, 10 skipped` here).

- [ ] **Step 5: Commit**

```bash
git add backend/app/shlif/ore_unet.py backend/tests/test_ore_unet.py
git commit -m "perf(ore-unet): batch under-tile crops into one forward pass instead of one-per-crop"
```

---

### Task 2: fp16 autocast + TF32 for both U-Nets

**Files:**
- Modify: `backend/app/shlif/ore_unet.py` (adds `_use_amp`, wraps the forward pass added in Task 1, adds TF32 flags to `build_ore_unet`)
- Modify: `backend/app/shlif/talc_unet.py` (adds `_use_amp`, wraps `talc_unet_mask`'s forward pass, adds TF32 flags to `build_talc_unet`)
- Test: `backend/tests/test_ore_unet.py` (extend)
- Test: `backend/tests/test_talc_unet.py` (new file)

**Interfaces:**
- Consumes: Task 1's `ore_unet_mask` body (the `for start in range(0, len(crops), batch_size): ... preds = model(chunk)...` block).
- Produces: `_use_amp(device) -> bool` in both modules (identical tiny helper, duplicated per this codebase's existing convention of duplicating small constants like `_MEAN`/`_STD` across these two files rather than sharing a module).

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_ore_unet.py`:

```python
def test_use_amp_true_for_cuda_devices():
    from app.shlif.ore_unet import _use_amp
    assert _use_amp("cuda") is True
    assert _use_amp("cuda:0") is True


def test_use_amp_false_for_cpu():
    from app.shlif.ore_unet import _use_amp
    assert _use_amp("cpu") is False
```

Create `backend/tests/test_talc_unet.py`:

```python
"""talc_unet_mask/build_talc_unet gate fp16 autocast + TF32 on CUDA only; the
CPU fallback path must stay plain fp32. This only covers the device-gating
logic -- there's no live CUDA in this sandbox to exercise the actual
autocast/TF32 behaviour; that needs the real L4 VM."""
from app.shlif.talc_unet import _use_amp


def test_use_amp_true_for_cuda_devices():
    assert _use_amp("cuda") is True
    assert _use_amp("cuda:0") is True


def test_use_amp_false_for_cpu():
    assert _use_amp("cpu") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_ore_unet.py tests/test_talc_unet.py -v`
Expected: FAIL with `ImportError: cannot import name '_use_amp' from 'app.shlif.ore_unet'` (and same for `talc_unet`) — neither module defines it yet. This works in this sandbox without torch, since `_use_amp` itself won't need torch.

- [ ] **Step 3: Implement `_use_amp` + autocast + TF32 in `ore_unet.py`**

In `backend/app/shlif/ore_unet.py`, add this function right after the `_STD` constant (before `build_ore_unet`):

```python
def _use_amp(device) -> bool:
    """True when ``device`` names a CUDA device -- gates fp16 autocast, which
    only helps (and is only valid) on CUDA; the CPU fallback path must stay
    plain fp32."""
    return str(device).startswith("cuda")
```

In `build_ore_unet`, add the TF32 flags right after `dev = device or (...)`:

```python
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=2)
```

In `ore_unet_mask` (from Task 1), change the forward-pass block to:

```python
    use_amp = _use_amp(device)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            chunk = batch[start:start + batch_size]
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(chunk)
            else:
                logits = model(chunk)
            preds = logits.argmax(1).cpu().numpy()
            for i, p in enumerate(preds):
                (y, x), (ch, cw) = coords[start + i], dims[start + i]
                ore[y:y + ch, x:x + cw] = (p[:ch, :cw] != 0)
```

- [ ] **Step 4: Implement `_use_amp` + autocast + TF32 in `talc_unet.py`**

Add the same `_use_amp` helper right after the `_STD` constant (before `resolve_threshold`):

```python
def _use_amp(device) -> bool:
    """True when ``device`` names a CUDA device -- gates fp16 autocast; the
    CPU fallback path must stay plain fp32."""
    return str(device).startswith("cuda")
```

In `build_talc_unet`, add the TF32 flags right after `dev = device or (...)`:

```python
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
```

Replace `talc_unet_mask`'s body with:

```python
def talc_unet_mask(rgb: np.ndarray, model, device: str, thr: float | None = 0.5) -> np.ndarray:
    """Binary talc mask (bool HxW) from the trained U-Net: ``sigmoid >= thr``.

    Mirrors ``annotate_talc.unet_mask``: the whole image is resized to 512,
    ImageNet-normalised, run through the sigmoid head, resized back to native
    resolution and thresholded. No WB/CLAHE — the talc model was trained on raw
    RGB (unlike the ore U-Net's ``wb_clahe`` path). ``thr`` is a 0..1 fraction, or
    ``None`` to pick it adaptively from the map (:func:`resolve_threshold`).
    """
    import torch

    H, W = rgb.shape[:2]
    im = cv2.resize(rgb, (SZ, SZ)).astype(np.float32) / 255.0
    im = (im - _MEAN) / _STD
    x = torch.from_numpy(im.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    use_amp = _use_amp(device)
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pr = torch.sigmoid(model(x))[0, 0]
        else:
            pr = torch.sigmoid(model(x))[0, 0]
        pr = pr.float().cpu().numpy()
    pr = cv2.resize(pr, (W, H))
    t = resolve_threshold(pr) if thr is None else float(thr)
    return pr >= t
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_ore_unet.py tests/test_talc_unet.py -v`
Expected: all `_use_amp` tests PASS (no torch needed for these); the Task 1 batching tests still SKIP here (torch absent).

Run the full suite: `cd backend && .venv/bin/python -m pytest -q`
Expected: no regressions vs. the count after Task 1.

- [ ] **Step 6: Commit**

```bash
git add backend/app/shlif/ore_unet.py backend/app/shlif/talc_unet.py backend/tests/test_ore_unet.py backend/tests/test_talc_unet.py
git commit -m "perf(unet): gate fp16 autocast + TF32 matmul on CUDA for both U-Nets"
```

---

### Task 3: Parallelise the `ensemble_phase_labels` perturbation ensemble

**Files:**
- Modify: `backend/app/shlif/uncertainty.py:13-49`
- Test: `backend/tests/test_uncertainty.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ensemble_phase_labels(rgb, cfg, perturbations=_PERTURBATIONS, on_step=None) -> np.ndarray` — identical signature and return; `ensemble_uncertainty` (unchanged) already calls it with `on_step=on_step` and needs no code change.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_uncertainty.py` (add `import threading` and `import time` near the top alongside the existing `import numpy as np`):

```python
import threading
import time


def test_perturbations_run_concurrently(monkeypatch):
    lock = threading.Lock()
    current = [0]
    max_concurrent = [0]

    def fake_segment_phases(pre, cfg):
        with lock:
            current[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current[0])
        time.sleep(0.05)
        with lock:
            current[0] -= 1
        class _R:
            labels = np.zeros(pre.shape[:2], np.uint8)
        return _R()

    monkeypatch.setattr(uncertainty, "segment_phases", fake_segment_phases)
    rgb = np.zeros((16, 16, 3), np.uint8)
    uncertainty.ensemble_phase_labels(rgb, CFG)

    assert max_concurrent[0] >= 2, (
        f"expected overlapping perturbation calls, max concurrent was {max_concurrent[0]}")


def test_matches_manual_sequential_reference():
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[8:24, 8:24] = 220
    expected = np.stack([
        uncertainty.segment_phases(
            uncertainty.preprocess(uncertainty._perturb(rgb, gamma, gain), CFG.preprocess),
            CFG.segment,
        ).labels.astype(np.uint8)
        for gamma, gain in uncertainty._PERTURBATIONS
    ])
    actual = uncertainty.ensemble_phase_labels(rgb, CFG)
    assert np.array_equal(actual, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: `test_perturbations_run_concurrently` FAILS with `assert 1 >= 2` (today's loop is strictly sequential — one `segment_phases` call in flight at a time). `test_matches_manual_sequential_reference` PASSES already (today's implementation *is* the sequential reference) — that's fine, it's a safety net for the next step, not a red/green signal.

- [ ] **Step 3: Implement the thread-pool parallelisation**

Replace `backend/app/shlif/uncertainty.py:13-49` with:

```python
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import phases
from .preprocess import preprocess
from .segment import segment_phases

# (gamma, gain) photometric perturbations — identity plus mild brighten/darken.
_PERTURBATIONS = (
    (1.0, 1.0),
    (0.82, 1.0),
    (1.22, 1.0),
    (1.0, 0.88),
    (1.0, 1.12),
)
_N_PHASES = 3  # matrix / magnetite / sulfide
_PHASE_RU = {phases.MATRIX: "матрица", phases.MAGNETITE: "магнетит", phases.SULFIDE: "сульфид"}

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    """Lazily-created, process-wide thread pool for the perturbation ensemble.
    Persistent (not re-created per call) — this runs on every non-empty tile,
    potentially thousands of times per gigapixel panorama. segment_phases and
    its preprocessing are cv2/numpy/skimage calls on large arrays, which
    release the GIL, so threads (not processes) give real parallelism here
    without pickling/IPC overhead per tile."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=min(len(_PERTURBATIONS), os.cpu_count() or 1))
    return _POOL


def _perturb(rgb: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    x = np.clip((rgb.astype(np.float32) / 255.0) ** gamma * gain, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def ensemble_phase_labels(rgb: np.ndarray, cfg, perturbations=_PERTURBATIONS, on_step=None) -> np.ndarray:
    """Stack of phase-label maps (K, H, W) — one classical segmentation per
    photometric perturbation, run concurrently across a thread pool (they are
    independent of each other). `on_step(i, total)`, if given, is called once
    per perturbation in the same fixed 1..total order as before — every
    perturbation is submitted to the pool up front (so they run in parallel),
    but progress is still reported in original order, not completion order."""
    def _one(pert):
        gamma, gain = pert
        pre = preprocess(_perturb(rgb, gamma, gain), cfg.preprocess)
        return segment_phases(pre, cfg.segment).labels.astype(np.uint8)

    total = len(perturbations)
    futures = [_pool().submit(_one, pert) for pert in perturbations]
    maps = []
    for i, f in enumerate(futures, 1):
        maps.append(f.result())
        if on_step:
            on_step(i, total)
    return np.stack(maps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_uncertainty.py -v`
Expected: all PASS, including `test_ensemble_uncertainty_reports_progress_per_perturbation` (pre-existing — confirms `on_step` ordering survived).

Run the full suite: `cd backend && .venv/bin/python -m pytest -q`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add backend/app/shlif/uncertainty.py backend/tests/test_uncertainty.py
git commit -m "perf(uncertainty): run the 5-perturbation ensemble concurrently instead of sequentially"
```

---

### Task 4: Bump `JobRunner` concurrency

**Files:**
- Modify: `backend/app/runtime.py:10`
- Test: `backend/tests/test_runtime.py` (new file)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `Runtime().runner` is a `JobRunner` constructed with `max_workers=2` (was 1).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_runtime.py`:

```python
"""Runtime wires JobRunner with max_workers=2 so a second job (or a
health-check) isn't serialised behind a long-running panorama analysis --
the priority is single-panorama latency, but this is a free, low-risk fix
for job-level concurrency on top of that."""
from app.runtime import Runtime


def test_runner_allows_two_concurrent_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.paths.db_path", lambda: tmp_path / "t.db")
    rt = Runtime()
    assert rt.runner._pool._max_workers == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/python -m pytest tests/test_runtime.py -v`
Expected: FAIL with `assert 1 == 2`.

- [ ] **Step 3: Implement the bump**

In `backend/app/runtime.py`, change:

```python
        self.runner = JobRunner(self.store)
```

to:

```python
        self.runner = JobRunner(self.store, max_workers=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/python -m pytest tests/test_runtime.py -v`
Expected: PASS.

Run the full suite: `cd backend && .venv/bin/python -m pytest -q`
Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add backend/app/runtime.py backend/tests/test_runtime.py
git commit -m "perf(runtime): allow 2 concurrent jobs instead of serialising everything behind 1"
```

---

## After all 4 tasks

Run the full suite one final time: `cd backend && .venv/bin/python -m pytest -q` and confirm the pass/skip count only grew (no prior test broken).

This sandbox cannot validate the GPU-side win (no CUDA here). Once merged, verify on the actual L4 VM per the design spec's "Rollout / measurement" section: compare `_run_panorama`'s wall-clock time before/after on a representative panorama, and watch `nvidia-smi` during the run to confirm GPU utilisation actually went up.
