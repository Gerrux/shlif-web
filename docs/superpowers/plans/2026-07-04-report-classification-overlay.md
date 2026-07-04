# Отчёт: попиксельная маска обычные/тонкие/тальк — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the already-computed but discarded per-pixel `normal`/`fine` intergrowth-liberation masks through to a persisted `intergrowth.png` layer (both closeup and panorama), render it as the ТЗ-required green/red/blue classification overlay in `Corrector.tsx`'s «Отчёт» tab, and remove the legacy per-tile `SORT_RGB` paint system in `panorama.py` that was the actual source of the "tile-based" look.

**Architecture:** Backend: a new `intergrowth_label_map(normal, fine)` helper packs the two boolean masks `verdict_from_masks` already returns into one uint8 label map (0/1/2); `verdict_from_masks_dict` — the single function both the panorama-analyze path and the universal mask-save/recompute endpoint already call — starts returning it; both call sites persist it via the existing `persist_editor_artifacts`, popping it out of the JSON-bound dict first (it's a raw `np.ndarray`, never JSON-serializable). Frontend: `Corrector.tsx` loads this new layer alongside `phases`/`talc` and branches its per-pixel canvas compositor on the active sidebar tab — «Отчёт» always paints all 3 colors from `intergrowth`/`talc`, ignoring the edit-tab's per-layer show/hide toggles; «Редактирование» keeps today's raw phase-color rendering unchanged.

**Tech Stack:** FastAPI/Python (pytest) backend, Next.js/React/TypeScript frontend (`node:test`), OpenCV (`cv2.resize`, nearest-neighbor for label maps).

## Global Constraints

- «Отчёт» always shows all three colors (green=обычные, red=тонкие, blue=тальк) — never gated by the edit-tab's сульфид/магнетит/тальк visibility toggles.
- PDF report (`backend/app/pipeline/report.py`) is explicitly out of scope for this plan.
- Canvas colors must match `backend/app/shlif/phases.py`'s existing constants exactly: `COLOR_NORMAL=(63,174,107)` green, `COLOR_FINE=(224,85,78)` red. `COLOR_TALC=(79,143,240)` already matches the frontend's existing `TALC_RGB` — reuse it, don't redefine.
- The RF texture classifier that feeds the `sort` card (`extract_features`, `clf.predict_proba`, `aggregate_section` in `panorama.py`) is a separate mechanism from the removed `SORT_RGB` tile-paint — it must be preserved untouched.
- `intergrowth.png` must always match `phases.png`/`talc.png`'s resolution exactly for the same job — a silent resolution mismatch would make the browser canvas stretch the label map with bilinear interpolation, corrupting class boundaries (0/1/2 are discrete labels, never intermediate values).
- No new npm/pip dependencies.
- Spec: `docs/superpowers/specs/2026-07-04-report-classification-overlay-design.md`.

---

## Task 1: `intergrowth_label_map` + `verdict_from_masks_dict` returns it

**Files:**
- Modify: `backend/app/pipeline/masks.py`
- Test: `backend/tests/test_masks.py`

**Interfaces:**
- Produces: `intergrowth_label_map(normal: np.ndarray, fine: np.ndarray) -> np.ndarray` (uint8, values `{0,1,2}`).
- Produces: `verdict_from_masks_dict(...)` now returns a dict with an extra `"intergrowth"` key (raw `np.ndarray`, uint8) alongside the existing `ore_class`/`text`/`metrics`.
- Consumed by: Task 2 (`persist_editor_artifacts`), Task 3 (`analyze_closeup`), Task 4 (`analyze_panorama`), Task 5 (`save_masks`).

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_masks.py` (after `test_verdict_from_masks_reacts_to_talc`):

```python
def test_intergrowth_label_map_values():
    normal = np.zeros((10, 10), bool); normal[0:3, 0:3] = True
    fine = np.zeros((10, 10), bool); fine[5:8, 5:8] = True
    im = masks.intergrowth_label_map(normal, fine)
    assert im.dtype == np.uint8
    assert set(np.unique(im)) <= {0, 1, 2}
    assert (im[0:3, 0:3] == 1).all()
    assert (im[5:8, 5:8] == 2).all()
    assert (im[8:, 8:] == 0).all()

def test_verdict_from_masks_dict_includes_intergrowth():
    cfg = loader.get_config()
    s = np.zeros((100, 100), bool); s[:10] = True
    m = np.zeros((100, 100), bool)
    mx = ~(s | m)
    v = masks.verdict_from_masks_dict(s, m, mx, np.zeros((100, 100), bool), cfg)
    assert "intergrowth" in v
    assert v["intergrowth"].shape == s.shape
    assert v["intergrowth"].dtype == np.uint8
    assert set(np.unique(v["intergrowth"])) <= {0, 1, 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_masks.py -v -k "intergrowth"`
Expected: FAIL — `AttributeError: module 'app.pipeline.masks' has no attribute 'intergrowth_label_map'` (and the second test fails on the missing `"intergrowth"` key).

- [ ] **Step 3: Write the implementation**

In `backend/app/pipeline/masks.py`, find:

```python
def split_phase_map(pm: np.ndarray):
    return pm == phases.SULFIDE, pm == phases.MAGNETITE, pm == phases.MATRIX

def verdict_from_masks_dict(sulfide, magnetite, matrix, talc, cfg) -> dict:
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg)
    return {"ore_class": v["ore_class"], "text": v["text"], "metrics": v["metrics"]}
```

Replace with:

```python
def split_phase_map(pm: np.ndarray):
    return pm == phases.SULFIDE, pm == phases.MAGNETITE, pm == phases.MATRIX

def intergrowth_label_map(normal: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """0 = not sulfide (magnetite/matrix/talc), 1 = normal (обычные), 2 = fine (тонкие)."""
    im = np.zeros(normal.shape, np.uint8)
    im[np.asarray(fine, dtype=bool)] = 2
    im[np.asarray(normal, dtype=bool)] = 1
    return im

def verdict_from_masks_dict(sulfide, magnetite, matrix, talc, cfg) -> dict:
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg)
    return {"ore_class": v["ore_class"], "text": v["text"], "metrics": v["metrics"],
            "intergrowth": intergrowth_label_map(v["normal"], v["fine"])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_masks.py -v`
Expected: all PASS, including the two new tests.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/masks.py backend/tests/test_masks.py
git commit -m "feat(masks): add intergrowth_label_map, wire it through verdict_from_masks_dict"
```

---

## Task 2: Persist and serve the `intergrowth` layer

**Files:**
- Modify: `backend/app/pipeline/masks.py` (`persist_editor_artifacts`)
- Modify: `backend/app/api/masks.py` (`get_mask` allowlist)
- Test: `backend/tests/test_masks.py`

**Interfaces:**
- Consumes: `intergrowth_label_map` from Task 1 (indirectly — callers now must supply `r["intergrowth"]`).
- Produces: `GET /api/masks/{jid}/intergrowth.png` — same shape/encoding contract as `phases.png` (uint8 label PNG, values as pixel intensity, not 0/255).

- [ ] **Step 1: Write the failing test**

In `backend/tests/test_masks.py`, find `test_persist_editor_artifacts_writes_all_files`:

```python
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

Replace with:

```python
def test_persist_editor_artifacts_writes_all_files(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    r = {
        "phase_map": np.zeros((8, 8), np.uint8),
        "talc": np.zeros((8, 8), bool),
        "intergrowth": np.zeros((8, 8), np.uint8),
        "superpixels": np.zeros((8, 8), np.uint16),
        "darkness": np.zeros((8, 8), np.uint8),
        "confidence": np.ones((8, 8), np.float32),
    }
    masks.persist_editor_artifacts("jobx", r)
    assert (tmp_path / "masks" / "jobx" / "phases.png").exists()
    assert (tmp_path / "masks" / "jobx" / "talc.png").exists()
    assert (tmp_path / "masks" / "jobx" / "intergrowth.png").exists()
    assert (tmp_path / "maps" / "jobx" / "superpixels.png").exists()
    assert (tmp_path / "maps" / "jobx" / "darkness.png").exists()
    assert (tmp_path / "maps" / "jobx" / "confidence.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_masks.py -v -k persist_editor_artifacts`
Expected: FAIL — `KeyError: 'intergrowth'`.

- [ ] **Step 3: Write the implementation**

In `backend/app/pipeline/masks.py`, find:

```python
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

Replace with:

```python
def persist_editor_artifacts(jid: str, r: dict) -> None:
    """Write the phase/talc/intergrowth masks + superpixel/darkness/confidence
    maps a finished job needs for the Corrector editor. Shared by the closeup
    and panorama result assembly so both produce identically-shaped, equally
    editable artifacts."""
    md = paths.masks_dir(jid)
    mp = paths.maps_dir(jid)
    (md / "phases.png").write_bytes(encode_png_gray(r["phase_map"]))
    (md / "talc.png").write_bytes(encode_png_gray((r["talc"].astype(np.uint8) * 255)))
    (md / "intergrowth.png").write_bytes(encode_png_gray(r["intergrowth"]))
    (mp / "superpixels.png").write_bytes(encode_png_label_rgb(r["superpixels"]))
    (mp / "darkness.png").write_bytes(encode_png_gray(r["darkness"]))
    (mp / "confidence.png").write_bytes(
        encode_png_gray(np.clip(r["confidence"] * 255.0, 0, 255).astype(np.uint8)))
```

In `backend/app/api/masks.py`, find:

```python
@router.get("/masks/{jid}/{layer}.png")
def get_mask(jid: str, layer: str):
    p = paths.masks_dir(jid) / f"{layer}.png"
    if layer not in {"phases", "talc"} or not p.exists():
        raise HTTPException(404, "mask not found")
    return FileResponse(p, media_type="image/png")
```

Replace with:

```python
@router.get("/masks/{jid}/{layer}.png")
def get_mask(jid: str, layer: str):
    p = paths.masks_dir(jid) / f"{layer}.png"
    if layer not in {"phases", "talc", "intergrowth"} or not p.exists():
        raise HTTPException(404, "mask not found")
    return FileResponse(p, media_type="image/png")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_masks.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/masks.py backend/app/api/masks.py backend/tests/test_masks.py
git commit -m "feat(masks): persist and serve the intergrowth.png layer"
```

---

## Task 3: Wire `analyze_closeup` to produce `intergrowth`

**Files:**
- Modify: `backend/app/pipeline/closeup.py`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `masks.intergrowth_label_map` (Task 1).
- Produces: `analyze_closeup(...)`'s returned dict now includes `"intergrowth"` (uint8 `np.ndarray`, same shape as `"phase_map"`) — consumed by `backend/app/api/analyze.py`'s existing `masks.persist_editor_artifacts(jid, r)` call (no change needed there — `r` already flows through unmodified).

- [ ] **Step 1: Write the failing test**

In `backend/tests/test_pipeline.py`, find:

```python
def test_analyze_closeup_structure(tiny_rgb):
    cfg = loader.get_config()
    r = closeup.analyze_closeup(tiny_rgb, cfg)
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
    assert r["phase_map"].shape == tiny_rgb.shape[:2]
    assert set(np.unique(r["phase_map"])) <= {0, 1, 2}
    assert r["talc"].shape == tiny_rgb.shape[:2]
    assert r["superpixels"].shape == tiny_rgb.shape[:2]
    assert r["darkness"].shape == tiny_rgb.shape[:2]
    assert r["sort"] is None or set(r["sort"]["classes"]) <= {"ordinary", "hard", "talcose"}
```

Replace with:

```python
def test_analyze_closeup_structure(tiny_rgb):
    cfg = loader.get_config()
    r = closeup.analyze_closeup(tiny_rgb, cfg)
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
    assert r["phase_map"].shape == tiny_rgb.shape[:2]
    assert set(np.unique(r["phase_map"])) <= {0, 1, 2}
    assert r["talc"].shape == tiny_rgb.shape[:2]
    assert r["intergrowth"].shape == tiny_rgb.shape[:2]
    assert set(np.unique(r["intergrowth"])) <= {0, 1, 2}
    assert r["superpixels"].shape == tiny_rgb.shape[:2]
    assert r["darkness"].shape == tiny_rgb.shape[:2]
    assert r["sort"] is None or set(r["sort"]["classes"]) <= {"ordinary", "hard", "talcose"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_pipeline.py -v -k test_analyze_closeup_structure`
Expected: FAIL — `KeyError: 'intergrowth'`.

- [ ] **Step 3: Write the implementation**

In `backend/app/pipeline/closeup.py`, find:

```python
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

Replace with:

```python
    m = res.masks
    phase_map = masks.phase_label_map(m["sulfide"], m["magnetite"])
    intergrowth = masks.intergrowth_label_map(m["normal"], m["fine"])

    unc = masks.uncertainty_for_editor(rgb, cfg)
    metrics = dict(res.metrics)
    metrics["undetermined_fraction"] = unc["undetermined_fraction"]

    return {
        "verdict": {"ore_class": res.ore_class, "text": res.text, "metrics": metrics},
        "sort": _sort_card(rgb, cfg),
        "phase_map": phase_map,
        "talc": m["talc"].astype(bool),
        "intergrowth": intergrowth,
        "superpixels": masks.build_superpixel_map(rgb),
        "darkness": masks.build_darkness_map(rgb),
        "confidence": unc["confidence"],
        "low_conf_zones": unc["low_conf_zones"],
        "text": res.text,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/closeup.py backend/tests/test_pipeline.py
git commit -m "feat(closeup): forward the already-computed normal/fine masks as intergrowth"
```

---

## Task 4: Wire `analyze_panorama` to produce and persist `intergrowth`

**Files:**
- Modify: `backend/app/pipeline/panorama.py`
- Test: `backend/tests/test_panorama.py`

**Interfaces:**
- Consumes: `verdict_from_masks_dict`'s new `"intergrowth"` key (Task 1), `persist_editor_artifacts` (Task 2).
- Produces: nothing new consumed by later tasks — this task must NOT touch `_run_panorama`'s tile-paint internals (that's Task 6).

**Important:** `verdict` (the dict returned by `verdict_from_masks_dict`) is later placed as-is into the JSON response (`"verdict": verdict` in the final return dict) — its `"intergrowth"` key (a raw `np.ndarray`) MUST be popped out before that happens, or the response will fail to serialize.

- [ ] **Step 1: Write the failing test**

In `backend/tests/test_panorama.py`, add (after `test_panorama_does_not_mutate_shared_config`):

```python
@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_persists_intergrowth_mask(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    from app.pipeline import masks as M
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    img = (np.random.default_rng(4).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()
    r = panorama.analyze_panorama(str(p), cfg, "igtest")
    assert "intergrowth" not in r["verdict"]  # popped before returning, must never leak to the JSON verdict
    ig = M.decode_png_gray((core_paths.masks_dir("igtest") / "intergrowth.png").read_bytes())
    assert ig.shape == (r["size"][1], r["size"][0])
    assert set(np.unique(ig)) <= {0, 1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_panorama.py -v -k persists_intergrowth`
Expected: FAIL — `AssertionError` on `"intergrowth" not in r["verdict"]` being false is NOT what fails first; it actually fails earlier because `"intergrowth"` is missing (not yet produced) — the read of `intergrowth.png` raises `FileNotFoundError` since the file was never written.

- [ ] **Step 3: Write the implementation**

In `backend/app/pipeline/panorama.py`, find:

```python
    assembled = _assemble_masks(path, cfg, arr)
    verdict = masks.verdict_from_masks_dict(
        assembled["sulfide"], assembled["magnetite"], assembled["matrix"], assembled["talc"], cfg)
    verdict["metrics"]["talc_share_est"] = float(assembled["dg"].mean())
```

Replace with:

```python
    assembled = _assemble_masks(path, cfg, arr)
    verdict = masks.verdict_from_masks_dict(
        assembled["sulfide"], assembled["magnetite"], assembled["matrix"], assembled["talc"], cfg)
    intergrowth = verdict.pop("intergrowth")
    verdict["metrics"]["talc_share_est"] = float(assembled["dg"].mean())
```

Then find:

```python
    phase_small = masks.phase_label_map(sulfide_small, magnetite_small)
    # confidence MAP for the editor overlay only (single downscaled pass);
    # low_conf_zones/undetermined_fraction above use _run_panorama's finer,
    # per-tile aggregation instead of this call's own (coarser) values.
    unc = masks.uncertainty_for_editor(edit, cfg)

    masks.persist_editor_artifacts(jid, {
        "phase_map": phase_small, "talc": talc_small,
        "superpixels": masks.build_superpixel_map(edit),
        "darkness": masks.build_darkness_map(edit),
        "confidence": unc["confidence"],
    })
```

Replace with:

```python
    phase_small = masks.phase_label_map(sulfide_small, magnetite_small)
    intergrowth_small = cv2.resize(intergrowth, (ew, eh), interpolation=cv2.INTER_NEAREST)
    # confidence MAP for the editor overlay only (single downscaled pass);
    # low_conf_zones/undetermined_fraction above use _run_panorama's finer,
    # per-tile aggregation instead of this call's own (coarser) values.
    unc = masks.uncertainty_for_editor(edit, cfg)

    masks.persist_editor_artifacts(jid, {
        "phase_map": phase_small, "talc": talc_small, "intergrowth": intergrowth_small,
        "superpixels": masks.build_superpixel_map(edit),
        "darkness": masks.build_darkness_map(edit),
        "confidence": unc["confidence"],
    })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_panorama.py -v`
Expected: all PASS (may skip if `models/classifier.pkl` is absent in this environment — that's the pre-existing `skipif` guard, not a new failure).

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama.py
git commit -m "feat(panorama): compute, resize, and persist the intergrowth mask"
```

---

## Task 5: Fix `save_masks` to persist `intergrowth.png` at the correct resolution

**Files:**
- Modify: `backend/app/api/masks.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: `verdict_from_masks_dict`'s `"intergrowth"` key (Task 1).
- Produces: `POST /masks/{jid}` now also writes an up-to-date `intergrowth.png`, guaranteed to match `phases.png`/`talc.png`'s resolution (the editor resolution the browser uploaded), never the (possibly larger, panorama-only) native analysis resolution.

**Why this task exists on its own:** `phases.png`/`talc.png` are written at editor resolution (whatever the browser uploaded) *before* the optional native-size upscale a few lines later; `verdict_from_masks_dict` is called *after* that upscale. For panorama jobs (where native resolution is larger than editor resolution), the `intergrowth` array coming back is at native resolution — persisting it as-is would silently create a resolution mismatch against `phases.png`/`talc.png` for the same job. The browser fetches all three layers into a canvas of a fixed `[w,h]` (`ctx.drawImage(img,0,0,w,h)`), which stretches a mismatched PNG with bilinear interpolation — corrupting the discrete 0/1/2 labels at class boundaries. The fix: resize `intergrowth` back down (nearest-neighbor) to the pre-upscale shape before writing.

- [ ] **Step 1: Write the failing tests**

In `backend/tests/test_api.py`, find:

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

Replace with:

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

    # layers + maps are fetchable
    assert c.get(f"/api/masks/{jid}/phases.png").status_code == 200
    assert c.get(f"/api/masks/{jid}/intergrowth.png").status_code == 200
    assert c.get(f"/api/maps/{jid}/superpixels.png").status_code == 200
    assert c.get(f"/api/maps/{jid}/darkness.png").status_code == 200

    # intergrowth.png must match phases.png's resolution exactly
    phases_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/phases.png").content)))
    ig_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/intergrowth.png").content)))
    assert ig_arr.shape == phases_arr.shape

    # edit: mark everything talc → verdict recomputes to talcose
    h, w = tiny_rgb.shape[:2]
    all_talc = np.full((h, w), 255, np.uint8)
    phases_png = c.get(f"/api/masks/{jid}/phases.png").content
    r = c.post(f"/api/masks/{jid}",
               files={"talc": ("talc.png", _png_bytes(all_talc), "image/png"),
                      "phases": ("phases.png", phases_png, "image/png")})
    assert r.status_code == 200
    assert r.json()["ore_class"] == "talcose"
    assert "intergrowth" not in r.json()  # popped server-side, must not leak into the Verdict JSON

    # resolution still matches after the recompute
    ig_arr2 = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/intergrowth.png").content)))
    assert ig_arr2.shape == phases_arr.shape


def test_edit_resizes_intergrowth_back_to_editor_resolution_when_native_differs(tiny_rgb):
    """save_masks computes intergrowth at whatever resolution the uploaded
    phases/talc arrive at, upscaled to native_size for accurate verdict metrics
    (mirroring how panorama jobs are edited) — but must persist intergrowth.png
    back down at the SAME resolution as phases.png/talc.png (the editor
    resolution), not native. Forges a panorama-shaped native_size on an
    ordinary closeup job so the mismatch path is exercised without needing an
    actual 50+ megapixel image."""
    c = TestClient(app)
    up = c.post("/api/analyze", files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    _poll(c, jid)

    from app.runtime import get_runtime
    job = get_runtime().store.get(jid)
    result = dict(job.result)
    result["native_size"] = [tiny_rgb.shape[1] * 3, tiny_rgb.shape[0] * 3]  # pretend native >> editor
    get_runtime().store.set_result(jid, result)

    h, w = tiny_rgb.shape[:2]
    all_talc = np.full((h, w), 255, np.uint8)
    phases_png = c.get(f"/api/masks/{jid}/phases.png").content
    r = c.post(f"/api/masks/{jid}",
               files={"talc": ("talc.png", _png_bytes(all_talc), "image/png"),
                      "phases": ("phases.png", phases_png, "image/png")})
    assert r.status_code == 200

    phases_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/phases.png").content)))
    ig_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/intergrowth.png").content)))
    assert ig_arr.shape == phases_arr.shape == (h, w)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: FAIL — `test_closeup_analyze_and_edit` fails on the `intergrowth.png` 404; `test_edit_resizes_intergrowth_back_to_editor_resolution_when_native_differs` fails the same way (the endpoint doesn't write `intergrowth.png` at all yet).

- [ ] **Step 3: Write the implementation**

In `backend/app/api/masks.py`, find:

```python
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

Replace with:

```python
@router.post("/masks/{jid}")
async def save_masks(jid: str, phases: UploadFile = File(...), talc: UploadFile = File(...)):
    pm = M.decode_png_gray(await phases.read()).astype(np.uint8)
    tk = M.decode_png_gray(await talc.read()) > 127
    paths.masks_dir(jid).joinpath("phases.png").write_bytes(M.encode_png_gray(pm))
    paths.masks_dir(jid).joinpath("talc.png").write_bytes(M.encode_png_gray(tk.astype(np.uint8) * 255))
    orig_h, orig_w = pm.shape  # editor resolution — intergrowth.png must match this, not native

    job = get_runtime().store.get(jid)
    native = (job.result or {}).get("native_size") if job else None
    if native and tuple(native) != (pm.shape[1], pm.shape[0]):
        nw, nh = int(native[0]), int(native[1])
        pm = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_NEAREST)
        tk = cv2.resize(tk.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0

    su, mg, mx = M.split_phase_map(pm)
    cfg = loader.get_config()
    v = M.verdict_from_masks_dict(su, mg, mx, tk & mx, cfg)
    intergrowth = v.pop("intergrowth")
    if intergrowth.shape != (orig_h, orig_w):
        intergrowth = cv2.resize(intergrowth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    paths.masks_dir(jid).joinpath("intergrowth.png").write_bytes(M.encode_png_gray(intergrowth))
    get_runtime().store.log_correction(jid, "phases+talc", int(pm.size))
    return v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_api.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/masks.py backend/tests/test_api.py
git commit -m "fix(masks): keep intergrowth.png resolution in sync with phases.png/talc.png on recompute"
```

---

## Task 6: Remove the legacy per-tile `SORT_RGB` paint system

**Files:**
- Modify: `backend/app/pipeline/panorama.py`
- Modify: `frontend/lib/api/types.ts`
- Test: `backend/tests/test_panorama.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_run_panorama(...)`'s returned dict no longer has an `"overlay"` key (only `"edit_rgb"`, the clean stitched photo); `analyze_panorama(...)`'s returned dict no longer has `"overlay_url"`.

**What must survive untouched:** the RF texture classifier that feeds the `sort` card (`extract_features`, `clf.predict_proba`, `records`, `aggregate_section`, `sort_proba`/`sort_top`), the U-Net/classical ore/matrix per-tile decision (`ore_bundle`, `ore_unet_mask`, `matrix`, `ore_frac`, `n_ore`/`n_matrix`/`n_tiles`), and the per-tile uncertainty/`low_conf_zones` aggregation. Only the *visual tile-painting* (`SORT_RGB`, `TALC_RGB`, `tile_blend_weight`-feathered blending into `overlay`) goes away — it was a coarse, tile-granular stand-in for a "verdict overlay" that the real per-pixel `intergrowth` mask (Tasks 1–5) now replaces properly.

- [ ] **Step 1: Remove the now-untested per-tile talc U-Net test**

The test `test_panorama_uses_talc_unet_when_available` in `backend/tests/test_panorama.py` exists specifically to verify that `_run_panorama`'s per-tile talc decision (used only to feed the now-removed `talc_disp`/tile-paint) comes from the U-Net. That code path is being deleted in this task, so this test has nothing left to test — remove the whole function:

```python
@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_uses_talc_unet_when_available(tmp_path, monkeypatch):
    """When loader.load_talc_unet() has weights, _run_panorama's per-tile talc
    decision (which drives the display overlay + ore-density weighting) must
    come from the U-Net, not the classical detect_talc.

    This checks only that path, not a ban on detect_talc anywhere in
    analyze_panorama: _assemble_masks (which produces the *reported* verdict)
    is classical-only regardless of U-Net availability (see this module's
    docstring — wiring U-Net into _assemble_masks too, mirroring
    shlif.analyze.analyze_image's ore_mask pattern, is a reasonable follow-up
    but is new, unreviewed work, not something to fold into this merge) — so
    it legitimately still calls detect_talc, and a blanket ban would fail for
    a reason unrelated to what this test is actually checking."""
    img = (np.random.default_rng(3).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()

    calls = []
    def fake_talc_unet(rgb, model, device, thr=None):
        calls.append(1)
        return np.ones(rgb.shape[:2], bool)
    monkeypatch.setattr(panorama.loader, "load_talc_unet", lambda: ("fake-model", "cpu"))
    monkeypatch.setattr(panorama, "talc_unet_mask", fake_talc_unet)

    r = panorama.analyze_panorama(str(p), cfg, "unettest")
    assert r["mode"] == "panorama"
    assert len(calls) > 0  # the U-Net talc path was actually exercised, not skipped
```

Delete this entire function (nothing replaces it — the behavior it protected no longer exists).

- [ ] **Step 2: Write the new regression test**

Add to `backend/tests/test_panorama.py` (in its place):

```python
@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_run_panorama_no_longer_builds_a_tile_painted_overlay(tmp_path):
    """SORT_RGB tile-painting was the source of the tile-based look users saw —
    removed per report-classification-overlay design §4.3. _run_panorama must
    no longer return an "overlay" key; edit_rgb is the plain stitched photo."""
    from app.shlif.tiling import load_working_array
    img = (np.random.default_rng(5).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()
    clf, feat, classes = loader.load_classifier()
    arr = load_working_array(str(p), cfg.tiling)
    run = panorama._run_panorama(str(p), clf, feat, classes, cfg, arr)
    assert "overlay" not in run
    assert "edit_rgb" in run
```

- [ ] **Step 3: Run the new test to verify it fails**

Run: `cd backend && python -m pytest tests/test_panorama.py -v -k no_longer_builds`
Expected: FAIL — `assert "overlay" not in run` fails because `_run_panorama` still returns `"overlay"` today.

- [ ] **Step 4: Update the module docstring**

In `backend/app/pipeline/panorama.py`, find:

```python
"""Panorama product flow — tile a whole-section scan, classify ore-rich tiles,
aggregate an ore-area-weighted section verdict, and stitch a display overlay.

Ported from ``hakaton_nornikel/scripts/analyze_panorama.py::run_panorama``. The
ore/matrix gate routes through the trained U-Net (``ore_unet_mask``, see
``backend/app/shlif/ore_unet.py``) when its checkpoint and torch are available,
falling back to the classical segmenter otherwise; talc per tile similarly
comes from the trained talc U-Net when its weights are loadable, else the
classical ``detect_talc``. Torch is never imported at this module's top
level — only lazily, inside the U-Net loaders/mask functions when a U-Net
path actually runs — so `import app.pipeline.panorama` still works without
torch installed.

Note: `_assemble_masks` (the whole-canvas mask reconstruction that feeds the
reported verdict) still uses the classical segmenter only, not the U-Net gate
`_run_panorama` uses for its own matrix/talc decisions below. Wiring U-Net into
`_assemble_masks` too — mirroring how `shlif.analyze.analyze_image` combines an
`ore_mask` with the classical sulfide/magnetite split — is a reasonable
follow-up, but is new, undesigned work; deliberately left classical-only here
rather than improvised during this merge.
"""
```

Replace with:

```python
"""Panorama product flow — tile a whole-section scan, classify ore-rich tiles,
and aggregate an ore-area-weighted section verdict.

Ported from ``hakaton_nornikel/scripts/analyze_panorama.py::run_panorama``. The
ore/matrix gate routes through the trained U-Net (``ore_unet_mask``, see
``backend/app/shlif/ore_unet.py``) when its checkpoint and torch are available,
falling back to the classical segmenter otherwise. Torch is never imported at
this module's top level — only lazily, inside the U-Net loaders/mask functions
when a U-Net path actually runs — so `import app.pipeline.panorama` still works
without torch installed.

Note: `_assemble_masks` (the whole-canvas mask reconstruction that feeds the
reported verdict) still uses the classical segmenter only, not the U-Net gate
`_run_panorama` uses for its own matrix decision below. Wiring U-Net into
`_assemble_masks` too — mirroring how `shlif.analyze.analyze_image` combines an
`ore_mask` with the classical sulfide/magnetite split — is a reasonable
follow-up, but is new, undesigned work; deliberately left classical-only here
rather than improvised during this merge.

The old per-tile display overlay (``SORT_RGB`` painting whole tiles one flat
colour, blended with ``tile_blend_weight`` feathering) is gone
(report-classification-overlay design §4.3) — it was a coarse, tile-granular
stand-in that predates the real per-pixel normal/fine/talc classification, and
it was the actual source of the "tile-based" look users were seeing. The
served image is now the plain stitched photo (``edit_rgb``); the client
renders its own precise overlay on top of it.
"""
```

- [ ] **Step 5: Remove the now-unused imports**

Find:

```python
from app.shlif import load_config, phases  # noqa: F401 (load_config kept for parity)
from app.shlif.features import extract_features
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import dark_gray_phase, detect_talc
from app.shlif.ore_unet import ore_unet_mask
from app.shlif.talc_unet import talc_unet_mask
from app.shlif.tiling import axis_core_bounds, iter_tiles, load_working_array, tile_blend_weight, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader, masks
from app.core import paths
```

Replace with:

```python
from app.shlif import load_config, phases  # noqa: F401 (load_config kept for parity)
from app.shlif.features import extract_features
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import dark_gray_phase, detect_talc
from app.shlif.ore_unet import ore_unet_mask
from app.shlif.tiling import axis_core_bounds, iter_tiles, load_working_array, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader, masks
from app.core import paths
```

(dropped `from app.shlif.talc_unet import talc_unet_mask`; dropped `tile_blend_weight` from the tiling import — both become unused by the end of this task.)

- [ ] **Step 6: Remove the module-level paint constants**

Find:

```python
SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate
```

Replace with:

```python
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate
```

- [ ] **Step 7: Remove the now-unused talc U-Net load**

Find:

```python
    unet = loader.load_talc_unet()
    ore_bundle = loader.load_ore_unet()
    ore_source = "unet" if ore_bundle is not None else "classical"
```

Replace with:

```python
    ore_bundle = loader.load_ore_unet()
    ore_source = "unet" if ore_bundle is not None else "classical"
```

- [ ] **Step 8: Remove the paint-accumulation setup**

Find:

```python
    base = edit.astype(np.float32)
    # Feathered stitch: accumulate weight*colour per tile and normalise, so
    # overlapping tiles blend seamlessly in the *display* overlay (no double-
    # darkened overlap band, no hard seam) — cosmetic only, unrelated to the
    # whole-canvas mask assembly above.
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
```

Replace with:

```python
    records = []
    low_conf_zones = []
    undet_weighted_sum = 0.0
    undet_px_total = 0
    n_tiles = n_ore = n_matrix = 0
    t0 = time.time()
```

- [ ] **Step 9: Remove the per-tile talc decision and paint accumulation from the tile loop**

Find:

```python
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
```

Replace with:

```python
        if ore_bundle is not None:
            ore_model, ore_device = ore_bundle
            matrix = ~ore_unet_mask(rgb, ore_model, ore_device)
        else:
            matrix = segment_phases(pre, cfg.segment).labels == phases.MATRIX
        ore_px = int((~matrix).sum())
        ore_frac = ore_px / max(matrix.size, 1)

        dx0, dy0 = int(tile.x * rx), int(tile.y * ry)

        th, tw = rgb.shape[:2]
```

Then find:

```python
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
```

Replace with:

```python
        if ore_frac >= min_ore:
            n_ore += 1
            feats = extract_features(rgb, cfg)
            proba = clf.predict_proba(np.array([[feats[k] for k in feat_names]], float))[0]
            pd = {classes[i]: float(proba[i]) for i in range(len(classes))}
            dens = ore_density(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), bright_thr)
            records.append((pd, dens))
        else:
            n_matrix += 1
```

- [ ] **Step 10: Remove the paint blending and clean up `_run_panorama`'s return**

Find:

```python
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

Replace with:

```python
    sec = aggregate_section(records, classes)
    sort_proba = {classes[i]: float(sec[i]) for i in range(len(classes))}
    sort_top = classes[int(sec.argmax())] if records else classes[0]

    return {
        "edit_rgb": edit, "sort": {"classes": sort_proba, "top": sort_top},
        "n_ore": n_ore, "n_matrix": n_matrix, "n_tiles": n_tiles,
        "seconds": time.time() - t0, "factor": factor,
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
        "ore_source": ore_source,
    }
```

- [ ] **Step 11: Serve the clean photo instead of the tinted overlay**

Find:

```python
    run = _run_panorama(path, clf, feat, classes, cfg, arr)
    verdict["metrics"]["undetermined_fraction"] = run["undetermined_fraction"]
    Image.fromarray(run["overlay"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)
```

Replace with:

```python
    run = _run_panorama(path, clf, feat, classes, cfg, arr)
    verdict["metrics"]["undetermined_fraction"] = run["undetermined_fraction"]
    Image.fromarray(run["edit_rgb"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)
```

- [ ] **Step 12: Drop the now-dead `overlay_url` field**

Find:

```python
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

Replace with:

```python
    return {
        "mode": "panorama",
        "verdict": verdict,
        "sort": run["sort"],
        "text": verdict["text"],
        "size": [ew, eh],
        "native_size": [W, H],
        "low_conf_zones": run["low_conf_zones"],
        "n_ore": run["n_ore"], "n_tiles": run["n_tiles"],
        "ore_source": run["ore_source"],
    }
```

In `frontend/lib/api/types.ts`, find:

```ts
export interface AnalyzeResult {
  mode: Mode; verdict: Verdict; sort: SortCard | null; text?: string;
  size?: [number, number]; overlay_url?: string; n_ore?: number; n_tiles?: number;
  low_conf_zones?: LowConfZone[];
}
```

Replace with:

```ts
export interface AnalyzeResult {
  mode: Mode; verdict: Verdict; sort: SortCard | null; text?: string;
  size?: [number, number]; n_ore?: number; n_tiles?: number;
  low_conf_zones?: LowConfZone[];
}
```

- [ ] **Step 13: Run backend tests to verify everything passes**

Run: `cd backend && python -m pytest tests/ -v`
Expected: all PASS (the deleted test is gone, the new `test_run_panorama_no_longer_builds_a_tile_painted_overlay` passes, `test_panorama_runs`/`test_panorama_does_not_mutate_shared_config`/`test_panorama_persists_intergrowth_mask` from Task 4 all still pass unmodified since none of them reference `overlay`/`overlay_url`/`talc_disp`).

- [ ] **Step 14: Run frontend build to verify the type change is clean**

Run: `cd frontend && npm run build`
Expected: succeeds — `overlay_url` isn't read anywhere in any `.tsx` file (verify with `grep -rn "overlay_url" frontend/app frontend/components frontend/lib` returning no matches before relying on this).

- [ ] **Step 15: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama.py frontend/lib/api/types.ts
git commit -m "refactor(panorama): remove the legacy per-tile SORT_RGB paint overlay"
```

---

## Task 7: Render the classification overlay in `Corrector.tsx`

**Files:**
- Modify: `frontend/lib/api/client.ts`
- Modify: `frontend/components/corrector/Corrector.tsx`

**Interfaces:**
- Consumes: `GET /api/masks/{jid}/intergrowth.png` (Tasks 2, 4, 5).
- Produces: nothing consumed by later tasks — this is the last task.

- [ ] **Step 1: Add `"intergrowth"` to the mask layer type**

In `frontend/lib/api/client.ts`, find:

```ts
export const maskUrl = (id: string, layer: "phases" | "talc") => `${base}/api/masks/${id}/${layer}.png`;
```

Replace with:

```ts
export const maskUrl = (id: string, layer: "phases" | "talc" | "intergrowth") => `${base}/api/masks/${id}/${layer}.png`;
```

- [ ] **Step 2: Add the green/red color constants**

In `frontend/components/corrector/Corrector.tsx`, find:

```tsx
const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const DARK_RGB: [number, number, number] = [200, 60, 220];
```

Replace with:

```tsx
const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const DARK_RGB: [number, number, number] = [200, 60, 220];
const NORMAL_RGB: [number, number, number] = [63, 174, 107];
const FINE_RGB: [number, number, number] = [224, 85, 78];
```

- [ ] **Step 3: Add the `intergrowth` ref and the `sideTab` mirror ref**

Find:

```tsx
  const darkRef = useRef<Uint8Array | null>(null);
```

Replace with:

```tsx
  const darkRef = useRef<Uint8Array | null>(null);
  const intergrowthRef = useRef<Uint8Array | null>(null);
```

Find:

```tsx
  const [sideTab, setSideTab] = useState<"edit" | "report">("report");
  const zp = useZoomPan();
```

Replace with:

```tsx
  const [sideTab, setSideTab] = useState<"edit" | "report">("report");
  const sideTabRef = useRef(sideTab); sideTabRef.current = sideTab;
  const zp = useZoomPan();
```

- [ ] **Step 4: Load the `intergrowth` layer at mount**

Find:

```tsx
      const phasesGray = await pngToArray(maskUrl(jobId, "phases"), w, h);
      const talc = await pngToArray(maskUrl(jobId, "talc"), w, h);
      spRef.current = await loadSuperpixels(mapUrl(jobId, "superpixels"), w, h);
      darkRef.current = await pngToArray(mapUrl(jobId, "darkness"), w, h);
```

Replace with:

```tsx
      const phasesGray = await pngToArray(maskUrl(jobId, "phases"), w, h);
      const talc = await pngToArray(maskUrl(jobId, "talc"), w, h);
      spRef.current = await loadSuperpixels(mapUrl(jobId, "superpixels"), w, h);
      darkRef.current = await pngToArray(mapUrl(jobId, "darkness"), w, h);
      intergrowthRef.current = await pngToArray(maskUrl(jobId, "intergrowth"), w, h);
```

- [ ] **Step 5: Branch `composePixel` on the active tab**

Find:

```tsx
  function composePixel(pm: Uint8Array, tc: Uint8Array, i: number) {
    const b = (enhancedRGBA.current ?? baseRGBA.current)!, o = outRef.current!.data; const j = i * 4;
    let r = b[j], g = b[j + 1], bl = b[j + 2];
    const cls = pm[i], v = visRef.current, a = maskAlphaRef.current;
    if ((cls === 2 && v.sulfide) || (cls === 1 && v.magnetite)) {
      const c = PHASE_RGB[cls];
      r = (1 - a) * r + a * c[0]; g = (1 - a) * g + a * c[1]; bl = (1 - a) * bl + a * c[2];
    }
    if (tc[i] && v.talc) {
      r = (1 - a) * r + a * TALC_RGB[0]; g = (1 - a) * g + a * TALC_RGB[1]; bl = (1 - a) * bl + a * TALC_RGB[2];
    }
    const dm = darkMaskRef.current;
    if (dm && dm[i] && !tc[i]) {
      r = 0.5 * r + 0.5 * DARK_RGB[0]; g = 0.5 * g + 0.5 * DARK_RGB[1]; bl = 0.5 * bl + 0.5 * DARK_RGB[2];
    }
    o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
  }
```

Replace with:

```tsx
  function composePixel(pm: Uint8Array, tc: Uint8Array, i: number) {
    const b = (enhancedRGBA.current ?? baseRGBA.current)!, o = outRef.current!.data; const j = i * 4;
    let r = b[j], g = b[j + 1], bl = b[j + 2];
    const a = maskAlphaRef.current;
    if (sideTabRef.current === "report") {
      // Отчёт: всегда все 3 цвета по ТЗ (обычные/тонкие/тальк) — фиксированный
      // интерпретируемый вид, не зависит от глазков видимости редактора.
      const ig = intergrowthRef.current;
      const cls = ig ? ig[i] : 0;
      if (cls === 1) {
        r = (1 - a) * r + a * NORMAL_RGB[0]; g = (1 - a) * g + a * NORMAL_RGB[1]; bl = (1 - a) * bl + a * NORMAL_RGB[2];
      } else if (cls === 2) {
        r = (1 - a) * r + a * FINE_RGB[0]; g = (1 - a) * g + a * FINE_RGB[1]; bl = (1 - a) * bl + a * FINE_RGB[2];
      }
      if (tc[i]) {
        r = (1 - a) * r + a * TALC_RGB[0]; g = (1 - a) * g + a * TALC_RGB[1]; bl = (1 - a) * bl + a * TALC_RGB[2];
      }
    } else {
      const cls = pm[i], v = visRef.current;
      if ((cls === 2 && v.sulfide) || (cls === 1 && v.magnetite)) {
        const c = PHASE_RGB[cls];
        r = (1 - a) * r + a * c[0]; g = (1 - a) * g + a * c[1]; bl = (1 - a) * bl + a * c[2];
      }
      if (tc[i] && v.talc) {
        r = (1 - a) * r + a * TALC_RGB[0]; g = (1 - a) * g + a * TALC_RGB[1]; bl = (1 - a) * bl + a * TALC_RGB[2];
      }
      const dm = darkMaskRef.current;
      if (dm && dm[i] && !tc[i]) {
        r = 0.5 * r + 0.5 * DARK_RGB[0]; g = 0.5 * g + 0.5 * DARK_RGB[1]; bl = 0.5 * bl + 0.5 * DARK_RGB[2];
      }
    }
    o[j] = r; o[j + 1] = g; o[j + 2] = bl; o[j + 3] = 255;
  }
```

- [ ] **Step 6: Redraw immediately on tab switch**

Find:

```tsx
  useEffect(() => {
    if (state && baseRGBA.current && !strokeRef.current) {
      srcRef.current = { pm: state.phaseMap, tc: state.talc };
      requestDraw();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phaseMap, state?.talc, vis, maskAlpha]);
```

Replace with:

```tsx
  useEffect(() => {
    if (state && baseRGBA.current && !strokeRef.current) {
      srcRef.current = { pm: state.phaseMap, tc: state.talc };
      requestDraw();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state?.phaseMap, state?.talc, vis, maskAlpha, sideTab]);
```

- [ ] **Step 7: Refetch `intergrowth` after saving edits**

Find:

```tsx
  async function save() {
    if (!state) return;
    setSaving(true);
    try {
      const phaseBlob = await rawMaskToPngBlob(state.phaseMap, w, h);
      const talcBlob = await maskToPngBlob(state.talc, w, h);
      const v = await saveMasks(jobId, phaseBlob, talcBlob);
      onVerdict(v);
    } finally { setSaving(false); }
  }
```

Replace with:

```tsx
  async function save() {
    if (!state) return;
    setSaving(true);
    try {
      const phaseBlob = await rawMaskToPngBlob(state.phaseMap, w, h);
      const talcBlob = await maskToPngBlob(state.talc, w, h);
      const v = await saveMasks(jobId, phaseBlob, talcBlob);
      intergrowthRef.current = await pngToArray(maskUrl(jobId, "intergrowth"), w, h);
      requestDraw();
      onVerdict(v);
    } finally { setSaving(false); }
  }
```

- [ ] **Step 8: Verify it builds and the existing suite still passes**

Run: `cd frontend && npm test`
Expected: all existing tests still PASS (this task adds no new `.test.mjs` files — `Corrector.tsx` has no component-test harness in this repo, same constraint as the earlier mask-editor-sidebar feature).

Run: `cd frontend && npm run build`
Expected: succeeds, no TypeScript errors.

- [ ] **Step 9: Manual verification**

Run: `cd backend && <your usual way of starting the API>` and `cd frontend && npm run dev`, open a closeup or panorama job.
Check:
- «Отчёт» tab (now default) shows green/red sulfide coloring (+ blue talc where present) instead of brass/steel — matches the legend chips already in `VerdictPanel`.
- Switching to «Редактирование» immediately reverts to the raw editing colors; switching back to «Отчёт» immediately shows the classification colors again.
- Editing sulfide/magnetite and clicking «Сохранить», then checking «Отчёт» — the green/red distribution has visibly updated to match the edit.
- For a panorama job: the photo under the mask is no longer blotched with flat per-tile color patches.

- [ ] **Step 10: Commit**

```bash
git add frontend/lib/api/client.ts frontend/components/corrector/Corrector.tsx
git commit -m "feat(corrector): render green/red/blue classification overlay in Отчёт tab"
```
