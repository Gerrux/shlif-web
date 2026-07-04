# Panorama Phase-Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the reported bug ("under overexposure, magnetite gets classified as sulfide on panoramas") by (1) surfacing the classical segmenter's exposure-driven instability as honest low-confidence zones instead of a silently wrong label, and (2) wiring the already-trained, exposure-invariant `unet_ore.pt` into the panorama ore/matrix gate with a graceful classical fallback.

**Architecture:** `backend/app/shlif/uncertainty.py::ensemble_uncertainty` (gamma/gain-perturbation ensemble) already exists and is used by `closeup.py` but never by `panorama.py` — Task 1 wires it into the panorama tile loop, downscaled per tile for cost, with zone bboxes remapped into the panorama's display-overlay coordinate space. `backend/models/unet_ore.pt` (binary ore/matrix, IoU 0.975 vs classical 0.81) sits unused — Tasks 2–3 port a small, guarded-import inference module (mirroring the existing `talc_unet.py` pattern) and swap it in for the classical `segment_phases`-based ore/matrix decision in `panorama.py`, falling back to the classical path when the checkpoint or torch/segmentation-models-pytorch aren't installed (the case in this dev sandbox today).

**Tech Stack:** Python 3.12, FastAPI/granian backend, numpy/opencv/scikit-image (classical CV), optional PyTorch + `segmentation_models_pytorch` (guarded, not a hard dependency — matches the existing `talc_unet.py` convention), pytest.

## Global Constraints

- Never import `torch` or `segmentation_models_pytorch` at module top level anywhere in `app.shlif` or `app.pipeline` — always inside a guarded function body (existing project rule; `backend/app/shlif/VENDORED.md` and `talc_unet.py` state it explicitly). `import app.shlif` and the full test suite must keep working with torch absent.
- Do **not** add `torch` / `segmentation-models-pytorch` to `backend/pyproject.toml` — this is a deliberate existing convention (verified: neither package is declared there today, and `talc_unet.py` already relies on the same undeclared-optional-import pattern).
- `backend/app/pipeline/panorama.py::analyze_panorama` deep-copies its `cfg` argument before mutating it (`cfg = copy.deepcopy(cfg)`) — never mutate the shared `@lru_cache`'d `Config` from `loader.get_config()`. `test_panorama.py::test_panorama_does_not_mutate_shared_config` already guards this; don't break it.
- Run tests with `cd backend && .venv/bin/pytest -q` (absolute venv path — this repo's zsh does not word-split unquoted `$VARS`, and there is no other interpreter on PATH guaranteed to have the right deps).
- Every new/modified test must pass in **this** sandbox, where `torch` and `segmentation_models_pytorch` are **not installed** (verified via `.venv/bin/python -c "import torch"` → `ModuleNotFoundError`). Any test that needs real U-Net inference must be written so it exercises the graceful-`None` fallback here, not real inference — that only runs on the organizer's GPU VM.
- Follow existing file conventions exactly: `backend/app/shlif/*.py` files are framework-agnostic and vendored from `hakaton_nornikel`; new divergences get recorded in `backend/app/shlif/VENDORED.md` (see Task 3, Step 10).

---

### Task 1: Wire ensemble-uncertainty into the panorama pipeline

**Files:**
- Modify: `backend/app/pipeline/panorama.py:19-27` (imports), `:29-32` (constants), `:79-83` (preamble), `:97-102` (tile loop), `:133-139` (`_run_panorama` return), `:154-161` (`analyze_panorama` return)
- Test: `backend/tests/test_panorama_uncertainty.py` (new)

**Interfaces:**
- Consumes: `app.shlif.uncertainty.ensemble_uncertainty(rgb: np.ndarray, cfg) -> dict` (keys: `confidence`, `entropy`, `low_conf`, `undetermined_fraction: float`, `labels`) and `app.shlif.uncertainty.find_low_conf_zones(result: dict, min_area: int = 64) -> list[dict]` (each zone: `{"bbox": [x, y, w, h], "area": int, "phase_a": str, "phase_b": str}`) — both already exist, unchanged.
- Produces: `_run_panorama(...)` return dict gains two keys: `"undetermined_fraction": float` (pixel-count-weighted mean across all non-empty tiles) and `"low_conf_zones": list[dict]` (same per-zone shape as above, but `bbox` remapped into **display-overlay pixel coordinates** — the same coordinate space as `overlay_url`'s image). `analyze_panorama(...)`'s return dict gains `"low_conf_zones"` at the top level and `"undetermined_fraction"` inside `verdict.metrics` — this is what Tasks 2–3 and any future caller read.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_panorama_uncertainty.py`:

```python
"""_run_panorama surfaces the ensemble uncertainty (undetermined_fraction +
low_conf_zones) the same way closeup.py already does, catching the classical
segmenter's exposure-driven magnetite<->sulfide flip instead of silently
mislabeling it."""
import numpy as np
import pytest
from PIL import Image

from app.pipeline import panorama, loader

CFG = loader.get_config()


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_reports_uncertainty(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)
    img[100:500, 100:500] = 220     # confident sulfide block
    img[700:1100, 700:1100] = 170   # borderline magnetite block -> disputed
                                     # under the ensemble's mild gamma/gain jitter
    p = tmp_path / "pano.png"
    Image.fromarray(img).save(p, "PNG")   # PNG, not JPEG: JPEG's block compression
                                            # smooths the flat blocks enough to hide
                                            # the dispute at these exact values

    r = panorama.analyze_panorama(str(p), CFG, "unctest")

    metrics = r["verdict"]["metrics"]
    assert "undetermined_fraction" in metrics
    assert metrics["undetermined_fraction"] > 0.0

    zones = r["low_conf_zones"]
    assert isinstance(zones, list)
    assert len(zones) >= 1
    assert any({"магнетит", "сульфид"} <= {z["phase_a"], z["phase_b"]} for z in zones)
    for z in zones:
        assert set(z) == {"bbox", "area", "phase_a", "phase_b"}
        assert len(z["bbox"]) == 4


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_uncertainty_does_not_mutate_shared_config(tmp_path):
    before = loader.get_config().talc.detect_dark_frac
    rng = np.random.default_rng(2)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)
    img[100:500, 100:500] = 220
    p = tmp_path / "pano.png"
    Image.fromarray(img).save(p, "PNG")
    panorama.analyze_panorama(str(p), loader.get_config(), "unccfgtest")
    assert loader.get_config().talc.detect_dark_frac == before
```

This exact fixture (seed 1, blocks at `[100:500,100:500]=220` and `[700:1100,700:1100]=170` in a
1200×2400 image, saved as **PNG**) is verified (outside this repo, during planning) to produce
`undetermined_fraction ≈ 0.0033` and 4 zones, all `магнетит`/`сульфид` pairs, once Steps 3–5 below
are implemented. The assertions above don't hardcode the exact count (fragile) but do assert the
qualitative behavior that matters: a real dispute is surfaced.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama_uncertainty.py -v`
Expected: FAIL — `KeyError: 'undetermined_fraction'` (the field doesn't exist yet in `analyze_panorama`'s return dict).

- [ ] **Step 3: Add the import and the downscale constant**

In `backend/app/pipeline/panorama.py`, change:

```python
from app.shlif.tiling import iter_tiles, tile_blend_weight, tile_grid
from app.pipeline import loader
from app.core import paths

SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
DISPLAY_MP = 4_000_000
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate
```

to:

```python
from app.shlif.tiling import iter_tiles, tile_blend_weight, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader
from app.core import paths

SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
DISPLAY_MP = 4_000_000
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate
_UNC_MAX_SIDE = 1024  # cap the ensemble-uncertainty resolution per tile (mirrors closeup.py)
```

- [ ] **Step 4: Add the accumulators and the per-tile ensemble call**

In `_run_panorama`, change:

```python
    talc_disp = np.zeros((dh, dw), bool)
    records = []
    talc_px = matrix_px = 0
    n_tiles = n_ore = n_matrix = 0
```

to:

```python
    talc_disp = np.zeros((dh, dw), bool)
    records = []
    low_conf_zones = []
    talc_px = matrix_px = 0
    undet_weighted_sum = 0.0
    undet_px_total = 0
    n_tiles = n_ore = n_matrix = 0
```

Then, in the tile loop, change:

```python
        dx0, dy0 = int(tile.x * rx), int(tile.y * ry)
        dx1, dy1 = min(int((tile.x + rgb.shape[1]) * rx), dw), min(int((tile.y + rgb.shape[0]) * ry), dh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue

        if ore_frac >= min_ore:
```

to:

```python
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
```

(`bx`/`by` convert a zone bbox from the *downscaled-tile-local* coordinates `find_low_conf_zones`
returns — the same convention `closeup.py::_uncertainty` already uses — through the tile's own
downscale factor and then through the panorama-native-to-display scale `rx`/`ry`, landing in the
same pixel space as the returned `overlay_url` image.)

- [ ] **Step 5: Return the new fields**

In `_run_panorama`, change:

```python
    return {
        "overlay": out, "verdict": verdict, "conf": conf,
        "proba": {classes[i]: float(sec[i]) for i in range(len(classes))},
        "talc_frac": talc_px / max(talc_px + matrix_px, 1),
        "n_ore": n_ore, "n_matrix": n_matrix, "n_tiles": n_tiles,
        "seconds": time.time() - t0, "factor": factor,
    }
```

to:

```python
    return {
        "overlay": out, "verdict": verdict, "conf": conf,
        "proba": {classes[i]: float(sec[i]) for i in range(len(classes))},
        "talc_frac": talc_px / max(talc_px + matrix_px, 1),
        "n_ore": n_ore, "n_matrix": n_matrix, "n_tiles": n_tiles,
        "seconds": time.time() - t0, "factor": factor,
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
    }
```

In `analyze_panorama`, change:

```python
    return {
        "mode": "panorama",
        "verdict": {"ore_class": r["verdict"], "text": "",
                    "metrics": {"talc_frac": r["talc_frac"], "confidence": r["conf"],
                                "sort_proba": r["proba"]}},
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": r["n_ore"], "n_tiles": r["n_tiles"], "talc_frac": r["talc_frac"],
    }
```

to:

```python
    return {
        "mode": "panorama",
        "verdict": {"ore_class": r["verdict"], "text": "",
                    "metrics": {"talc_frac": r["talc_frac"], "confidence": r["conf"],
                                "sort_proba": r["proba"],
                                "undetermined_fraction": r["undetermined_fraction"]}},
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": r["n_ore"], "n_tiles": r["n_tiles"], "talc_frac": r["talc_frac"],
        "low_conf_zones": r["low_conf_zones"],
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama_uncertainty.py -v`
Expected: PASS (both tests). This was independently verified during planning: `undetermined_fraction ≈ 0.0033`, 4 zones, all `магнетит`/`сульфид` pairs.

- [ ] **Step 7: Run the full existing panorama test suite to check for regressions**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama.py tests/test_panorama_aggregate.py tests/test_tiling_feather.py -v`
Expected: all PASS, unchanged from before this task (verified during planning — no output shape or behavior of the existing fields changed).

- [ ] **Step 8: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: `39 passed` (or more, now including the 2 new tests → `41 passed`), 0 failed.

- [ ] **Step 9: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama_uncertainty.py
git commit -m "feat(panorama): surface ensemble-uncertainty low-confidence zones

The classical segment_phases split has no exposure anchor — a brightened tile
can flip magnetite pixels into the sulfide band (verified empirically: 11%->46%
of true magnetite area flips as simulated gain increases 1.3x->2.0x). closeup.py
already runs the perturbation-ensemble uncertainty check; panorama.py never did.
Wires it in per-tile (downscaled for cost) so disputed magnetite/sulfide zones
surface as 'на проверку' instead of a confident wrong label."
```

---

### Task 2: Ore/matrix U-Net module + loader wiring

**Files:**
- Create: `backend/app/shlif/ore_unet.py`
- Modify: `backend/app/pipeline/loader.py`
- Test: `backend/tests/test_ore_unet.py` (new)
- Modify: `backend/tests/test_loader.py` (add one test)

**Interfaces:**
- Consumes: nothing new (only stdlib `pathlib`, `cv2`, `numpy`, and guarded `torch`/`segmentation_models_pytorch`); `app.shlif.preprocess.wb_clahe(rgb: np.ndarray, clahe_clip: float = 0.01) -> np.ndarray` (existing, unchanged).
- Produces: `app.shlif.ore_unet.build_ore_unet(ckpt: str = "unet_ore.pt", device: str | None = None) -> tuple[model, str] | None` and `app.shlif.ore_unet.ore_unet_mask(rgb: np.ndarray, model, device: str, tile: int = 512) -> np.ndarray` (bool `(H, W)`, `True` = ore). `app.pipeline.loader.load_ore_unet() -> tuple[model, str] | None`, `@lru_cache`'d like `load_classifier()`. Task 3 consumes all three names directly.

- [ ] **Step 1: Write the failing test for the missing-checkpoint path**

Create `backend/tests/test_ore_unet.py`:

```python
"""build_ore_unet guards on the checkpoint existing (and on torch/smp being
importable) so panorama.py can fall back to the classical segmenter cleanly
when neither is available -- exactly the case in this dev sandbox today."""
from app.shlif.ore_unet import build_ore_unet


def test_build_ore_unet_missing_checkpoint_returns_none(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    assert build_ore_unet(str(missing)) is None


def test_build_ore_unet_returns_none_without_torch_or_smp(tmp_path):
    # A checkpoint path that exists but isn't a real torch state dict still
    # must degrade to None, not raise -- covers "file present, torch/smp
    # absent or load fails" without needing real weights in the test.
    fake_ckpt = tmp_path / "unet_ore.pt"
    fake_ckpt.write_bytes(b"not a real checkpoint")
    assert build_ore_unet(str(fake_ckpt)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest -q tests/test_ore_unet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.shlif.ore_unet'`.

- [ ] **Step 3: Write `ore_unet.py`**

Create `backend/app/shlif/ore_unet.py`:

```python
"""Trained ore/matrix U-Net (``models/unet_ore.pt``) for the panorama ore gate.

Binary ore-vs-matrix segmentation (IoU 0.975 vs the classical multi-Otsu+Lab
segmenter's 0.81 on LumenStone), illumination-invariant by construction --
trained on gray-world-WB + CLAHE-normalised tiles, so an over/under-exposed
capture maps to the same decision (unlike ``segment_phases``'s per-image-
relative Otsu split -- see ``uncertainty.py`` for how that instability is
flagged for the finer magnetite/sulfide split this model does NOT make).

Ported from ``hakaton_nornikel/scripts/sam2_prelabel.py::build_unet`` /
``unet_ore_decision``. Guarded import: returns ``None`` when the checkpoint
or torch/segmentation_models_pytorch are unavailable, so CPU-only/no-model
machines fall back to the classical ``segment_phases`` cleanly.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ORE_CKPT = "unet_ore.pt"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_ore_unet(ckpt: str = ORE_CKPT, device: str | None = None):
    """Load the trained ore/matrix U-Net -> ``(model, device)``, or ``None``.

    Returns ``None`` when the checkpoint file is missing or torch/smp fail to
    import or load -- the caller then keeps the classical ``segment_phases``.
    """
    if not Path(ckpt).exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=2)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        return model.to(dev).eval(), dev
    except Exception:
        return None


def ore_unet_mask(rgb: np.ndarray, model, device: str, tile: int = 512) -> np.ndarray:
    """Bool (H, W): True = ore (sulfide+magnetite), tiled U-Net inference.

    Applies gray-world WB + CLAHE per sub-tile before the ImageNet
    normalisation -- IDENTICAL to training (``wb_clahe``). This MUST stay on
    for this checkpoint (unlike the talc U-Net, which trained on raw RGB).
    """
    import torch

    from .preprocess import wb_clahe

    H, W = rgb.shape[:2]
    ore = np.zeros((H, W), bool)
    with torch.inference_mode():
        for y in range(0, H, tile):
            for x in range(0, W, tile):
                crop = rgb[y:y + tile, x:x + tile]
                ch, cw = crop.shape[:2]
                cp = cv2.copyMakeBorder(crop, 0, tile - ch, 0, tile - cw, cv2.BORDER_REFLECT)
                cp = wb_clahe(cp)
                t = ((cp.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)
                t = torch.from_numpy(t).unsqueeze(0).to(device)
                p = model(t).argmax(1)[0].cpu().numpy()
                ore[y:y + ch, x:x + cw] = (p[:ch, :cw] != 0)
    return ore
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest -q tests/test_ore_unet.py -v`
Expected: PASS (both tests — verified during planning; the second test hits the `except Exception`
branch because `torch.load` on a non-checkpoint byte string raises, and separately because
`segmentation_models_pytorch` isn't installed in this sandbox at all, so the `import` inside the
`try` fails first either way).

- [ ] **Step 5: Write the failing test for `loader.load_ore_unet()`**

In `backend/tests/test_loader.py`, add (matching the existing `test_classifier_absent_returns_none` pattern):

```python
def test_load_ore_unet_absent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.settings, "models_dir", tmp_path)
    loader.load_ore_unet.cache_clear()
    assert loader.load_ore_unet() is None
    loader.load_ore_unet.cache_clear()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest -q tests/test_loader.py::test_load_ore_unet_absent_returns_none -v`
Expected: FAIL — `AttributeError: module 'app.pipeline.loader' has no attribute 'load_ore_unet'`.

- [ ] **Step 7: Add `load_ore_unet` to `loader.py`**

In `backend/app/pipeline/loader.py`, change:

```python
from __future__ import annotations
import os, pickle
from functools import lru_cache
from app.core.settings import settings
from app.shlif import load_config
from app.shlif.config import Config

@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()

@lru_cache(maxsize=1)
def load_classifier():
    p = settings.models_dir / "classifier.pkl"
    if not p.exists():
        return None
    m = pickle.load(open(p, "rb"))
    return m["clf"], list(m["feature_names"]), [str(c) for c in m["classes"]]
```

to:

```python
from __future__ import annotations
import os, pickle
from functools import lru_cache
from app.core.settings import settings
from app.shlif import load_config
from app.shlif.config import Config
from app.shlif.ore_unet import build_ore_unet

@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()

@lru_cache(maxsize=1)
def load_classifier():
    p = settings.models_dir / "classifier.pkl"
    if not p.exists():
        return None
    m = pickle.load(open(p, "rb"))
    return m["clf"], list(m["feature_names"]), [str(c) for c in m["classes"]]

@lru_cache(maxsize=1)
def load_ore_unet():
    """``(model, device)`` from ``models/unet_ore.pt``, or ``None`` when the
    checkpoint or torch/segmentation_models_pytorch are unavailable."""
    p = settings.models_dir / "unet_ore.pt"
    return build_ore_unet(str(p))
```

(`app.shlif.ore_unet` has no top-level `torch`/`smp` imports, so this top-level `from
app.shlif.ore_unet import build_ore_unet` is safe without torch installed — verified in Step 4.)

- [ ] **Step 8: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest -q tests/test_loader.py -v`
Expected: PASS, all tests in the file (including the pre-existing ones — `test_config_loads`,
`test_classifier_absent_returns_none`, `test_gpu_false_without_torch`, `test_model_status_shape`,
and the new `test_load_ore_unet_absent_returns_none`).

- [ ] **Step 9: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all passing, no regressions (this task only adds a new module and a new loader function;
nothing existing calls them yet).

- [ ] **Step 10: Commit**

```bash
git add backend/app/shlif/ore_unet.py backend/app/pipeline/loader.py backend/tests/test_ore_unet.py backend/tests/test_loader.py
git commit -m "feat(shlif): add guarded ore/matrix U-Net loader (unet_ore.pt)

Ports build_unet/unet_ore_decision from hakaton_nornikel's sam2_prelabel.py
into a standalone, CPU-safe module (mirrors the existing talc_unet.py
guarded-import pattern — never imports torch at module top level, returns
None when the checkpoint or torch/segmentation-models-pytorch aren't
available). Not wired into any pipeline yet — that's the next task."
```

---

### Task 3: Route the panorama ore/matrix gate through the U-Net, with classical fallback

**Files:**
- Modify: `backend/app/pipeline/panorama.py` (import; `_run_panorama` ore-bundle fetch + tile-loop branch)
- Modify: `backend/app/shlif/VENDORED.md`
- Modify: `README.md`
- Test: `backend/tests/test_panorama_unet_gate.py` (new)

**Interfaces:**
- Consumes: `app.shlif.ore_unet.ore_unet_mask` and `app.pipeline.loader.load_ore_unet` from Task 2 (exact names/signatures as produced there).
- Produces: one new field, `"ore_source": "unet" | "classical"`, in both `_run_panorama`'s and `analyze_panorama`'s return dicts (a transparency flag for the jury/frontend — which code path decided the ore/matrix split for this run). Otherwise no new return fields: this task changes *which code path* computes the `matrix` boolean inside `_run_panorama`'s tile loop — the `ore_frac` gate, the display overlay, and everything downstream keep the exact same shape as before.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_panorama_unet_gate.py`:

```python
"""When loader.load_ore_unet() returns a model bundle, _run_panorama must route
the ore/matrix decision through ore_unet_mask instead of the classical
segment_phases split. When it returns None (as in this dev sandbox, where
torch/segmentation-models-pytorch aren't installed), the classical path must
keep working exactly as before -- covered by the existing test_panorama.py."""
import numpy as np
import pytest
from PIL import Image

from app.pipeline import panorama, loader

CFG = loader.get_config()


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_routes_ore_gate_through_unet_when_available(tmp_path, monkeypatch):
    calls = []

    def fake_mask(rgb, model, device, tile=512):
        calls.append(rgb.shape[:2])
        return np.zeros(rgb.shape[:2], bool)   # deterministic: "nothing is ore"

    monkeypatch.setattr(panorama, "ore_unet_mask", fake_mask)
    monkeypatch.setattr(panorama.loader, "load_ore_unet", lambda: (object(), "cpu"))

    rng = np.random.default_rng(4)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)
    img[100:500, 100:500] = 220   # with the REAL classical segmenter this tile
                                   # is ore-gated (n_ore == 1, verified during
                                   # planning) -- proving the fake mask's
                                   # "nothing is ore" answer actually won means
                                   # the wiring, not the classical path, decided.
    p = tmp_path / "pano.png"
    Image.fromarray(img).save(p, "PNG")

    r = panorama.analyze_panorama(str(p), CFG, "unetwiring")

    assert calls, "ore_unet_mask must be invoked when load_ore_unet() returns a model"
    assert r["n_ore"] == 0
    assert r["ore_source"] == "unet"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama_unet_gate.py -v`
Expected: FAIL — `AttributeError: <module 'app.pipeline.panorama'> does not have the attribute
'ore_unet_mask'` (monkeypatch fails because `panorama.py` doesn't import that name yet).

- [ ] **Step 3: Add the import**

In `backend/app/pipeline/panorama.py`, change:

```python
from app.shlif.tiling import iter_tiles, tile_blend_weight, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader
```

to:

```python
from app.shlif.ore_unet import ore_unet_mask
from app.shlif.tiling import iter_tiles, tile_blend_weight, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader
```

- [ ] **Step 4: Fetch the ore bundle once, before the tile loop**

In `_run_panorama`, change:

```python
    ore_pct = float(getattr(cfg.tiling, "ore_density_pct", ORE_DENSITY_PCT))
    bright_thr = float(np.percentile(cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY), ore_pct))
```

to:

```python
    ore_pct = float(getattr(cfg.tiling, "ore_density_pct", ORE_DENSITY_PCT))
    bright_thr = float(np.percentile(cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY), ore_pct))
    # trained ore/matrix U-Net when available (IoU 0.975 vs classical 0.81);
    # None (missing checkpoint or torch/smp) -> classical segment_phases fallback
    ore_bundle = loader.load_ore_unet()
    ore_source = "unet" if ore_bundle is not None else "classical"
```

- [ ] **Step 5: Branch the tile-loop matrix decision**

In the tile loop, change:

```python
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        matrix = segment_phases(pre, cfg.segment).labels == 0
        talc = detect_talc(pre, matrix, cfg.talc)
```

to:

```python
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        if ore_bundle is not None:
            ore_model, ore_device = ore_bundle
            matrix = ~ore_unet_mask(rgb, ore_model, ore_device)
        else:
            matrix = segment_phases(pre, cfg.segment).labels == 0
        talc = detect_talc(pre, matrix, cfg.talc)
```

- [ ] **Step 6: Propagate `ore_source` into both return dicts**

In `_run_panorama`'s return (as left by Task 1, Step 5), change:

```python
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
    }
```

to:

```python
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
        "ore_source": ore_source,
    }
```

In `analyze_panorama`'s return (as left by Task 1, Step 5), change:

```python
        "n_ore": r["n_ore"], "n_tiles": r["n_tiles"], "talc_frac": r["talc_frac"],
        "low_conf_zones": r["low_conf_zones"],
    }
```

to:

```python
        "n_ore": r["n_ore"], "n_tiles": r["n_tiles"], "talc_frac": r["talc_frac"],
        "low_conf_zones": r["low_conf_zones"], "ore_source": r["ore_source"],
    }
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama_unet_gate.py -v`
Expected: PASS — verified during planning (`calls` is non-empty, `r["n_ore"] == 0` and
`r["ore_source"] == "unet"`, vs. `n_ore == 1` with the real classical path on the same fixture
when `load_ore_unet`/`ore_unet_mask` aren't monkeypatched).

- [ ] **Step 8: Run the full existing panorama + uncertainty suites to confirm the classical fallback still works untouched**

Run: `cd backend && .venv/bin/pytest -q tests/test_panorama.py tests/test_panorama_uncertainty.py tests/test_panorama_aggregate.py tests/test_tiling_feather.py -v`
Expected: all PASS. `loader.load_ore_unet()` returns `None` in this sandbox (torch/smp absent), so
every one of these tests exercises the untouched classical `segment_phases` branch — verified during
planning to produce byte-for-byte the same `n_ore`/`n_tiles`/`undetermined_fraction` results as
before this task.

- [ ] **Step 9: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all passing (now `39 + 2 (Task 1) + 2 (Task 2) + 1 (Task 3) = 44 passed`, 0 failed).

- [ ] **Step 10: Update `VENDORED.md` to record the new divergence**

In `backend/app/shlif/VENDORED.md`, under the existing `## Divergence from origin (2026-07-04)`
section, add one bullet after the `uncertainty.py` line:

```markdown
- new `ore_unet.py` — guarded loader/inference for the trained ore/matrix U-Net
  (`unet_ore.pt`), ported from `hakaton_nornikel/scripts/sam2_prelabel.py::build_unet` /
  `unet_ore_decision`. Not present as a standalone module in origin (origin's version lives
  inline in a CLI script); wired into `panorama.py`'s ore/matrix gate with a classical fallback.
```

- [ ] **Step 11: Update `README.md`'s Models table**

In `README.md`, the `unet_ore.pt` row currently reads:

```markdown
| `unet_ore.pt` *(planned)* | GPU ore/matrix segmentation — **not yet wired in this milestone** | No behaviour change: the pipeline always runs the classical multi-Otsu + Lab-colour segmenter (CPU) |
```

Change it to:

```markdown
| `unet_ore.pt` | Ore/matrix segmentation for the panorama ore gate (IoU 0.975 vs classical 0.81) | Panorama runs the classical multi-Otsu + Lab-colour segmenter (CPU) as a graceful fallback whenever the checkpoint or torch/segmentation-models-pytorch aren't available |
```

And just below the existing `> **U-Net wiring is deferred.**` paragraph, add:

```markdown
> **Update:** the panorama ore/matrix gate now uses `unet_ore.pt` when it — and
> `torch`/`segmentation-models-pytorch` — are available (`backend/app/shlif/ore_unet.py`,
> wired in `backend/app/pipeline/panorama.py::_run_panorama`). Neither package is a hard
> dependency (not in `backend/pyproject.toml`, matching the existing `talc_unet.py`
> convention) — install them only on a box that will actually run inference. The
> magnetite/sulfide split inside the ore region, and `unet_talc.pt`/`unet_s2.pt`, remain
> unwired as before.
```

- [ ] **Step 12: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama_unet_gate.py backend/app/shlif/VENDORED.md README.md
git commit -m "feat(panorama): route the ore/matrix gate through unet_ore.pt when available

Replaces the classical segment_phases ore/matrix split in the panorama tile
loop with the trained U-Net (IoU 0.975 vs 0.81) when the checkpoint and
torch/segmentation-models-pytorch are present; falls back to the classical
path otherwise (the case in this dev sandbox, and the only path exercised by
CI). The magnetite/sulfide split within the ore region is unchanged — its
instability is what Task 1's ensemble-uncertainty now flags instead."
```

---

## Non-goals (explicitly out of scope for this plan)

- **Frontend rendering** of `low_conf_zones` / `undetermined_fraction` for panorama mode. This plan
  only makes the backend surface honest data (matching what `closeup.py` already returns); wiring
  it into the UI is a separate, later task if wanted.
- **Anchoring `segment_phases` itself** against exposure drift (the design doc's item 3 / "stretch"
  option). Touches `features.py`'s RF-classifier feature extraction (F1 0.84 baseline) and needs its
  own re-validation — explicitly deferred.
- **The 5-class LumenStone U-Net (`unet_s2.pt`)** for a true magnetite-vs-sulfide split. Its own
  documented ceiling (magnetite IoU ~0.36) and lack of any existing inference wrapper beyond
  train/eval scripts make it a separate, larger effort — not part of the two tasks the user approved.
- **Docker/deployment wiring** (installing `torch`/`segmentation-models-pytorch` in the container
  image, GPU passthrough in `docker-compose.yml`). Out of scope; `talc_unet.py` already established
  that these packages are optional-at-runtime, not part of the build.
