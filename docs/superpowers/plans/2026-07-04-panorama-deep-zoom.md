# Panorama Deep-Zoom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the panorama viewer show real additional detail on zoom by generating a tile pyramid from the already-decoded working array and serving it through a standalone, view-only OpenSeadragon viewer.

**Architecture:** `analyze_panorama()` already decodes the full scan into `arr` (up to `cfg.tiling.max_pixels`, 150 MP) before shrinking it to the ~2400px edit copy used by `Corrector.tsx`. This plan adds one call — `tiles.build_pyramid(arr, jid)`, wrapped in try/except — that slices `arr` into a JPEG tile pyramid on disk before it's discarded, plus two read-only endpoints to serve it, plus a small self-contained frontend component that shows a "view at full resolution" button only when a pyramid exists. Nothing about editing, closeup jobs, or the PDF report changes.

**Tech Stack:** FastAPI + Pillow/OpenCV/NumPy (backend, matches existing pipeline stack), Next.js 15 / React 19 + OpenSeadragon (frontend, new dependency).

**Spec:** `docs/superpowers/specs/2026-07-04-panorama-deep-zoom-design.md`

## Global Constraints

- Tile size: 256px, fixed (not configurable).
- Tile JPEG quality: 82.
- No new backend config value — pyramid resolution is exactly `arr`'s resolution, already governed by the existing `cfg.tiling.max_pixels` (150,000,000, `backend/app/config/default.yaml:36`).
- Scoped to panorama-mode jobs only. Closeup is out of scope (see spec §Non-goals).
- `Corrector.tsx`, `useZoomPan.ts`, `_run_panorama`, `_assemble_masks` — all unchanged. This work is purely additive.
- New endpoints follow the existing `FileResponse` + 404-on-missing pattern from `backend/app/api/masks.py`.
- New frontend dependencies: `openseadragon`, `@types/openseadragon`.
- Pyramid build failures must never fail the analysis job (try/except around the one call site).

---

### Task 1: Tile pyramid builder (`build_pyramid`)

**Files:**
- Create: `backend/app/pipeline/tiles.py`
- Modify: `backend/app/core/paths.py` (add `tiles_dir` helper)
- Test: `backend/tests/test_panorama_tiles.py`

**Interfaces:**
- Produces: `tiles.build_pyramid(arr: np.ndarray, jid: str) -> None` — writes `data/tiles/{jid}/{level}/{col}_{row}.jpg` + `data/tiles/{jid}/manifest.json` (`{"width": int, "height": int, "tileSize": 256, "maxLevel": int}`). Level 0 = lowest resolution, `maxLevel` = `arr`'s own resolution (OpenSeadragon's own numbering convention).
- Produces: `paths.tiles_dir(job_id: str) -> Path` — same pattern as the existing `paths.masks_dir`/`paths.maps_dir`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_panorama_tiles.py`:

```python
import json
import numpy as np
from PIL import Image
from app.pipeline import tiles


def test_build_pyramid_writes_expected_levels_and_manifest(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)

    arr = (np.random.default_rng(0).integers(0, 255, (600, 1000, 3))).astype(np.uint8)
    tiles.build_pyramid(arr, "jobA")

    out_dir = tmp_path / "tiles" / "jobA"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest == {"width": 1000, "height": 600, "tileSize": 256, "maxLevel": 2}

    # level 0 (150x250): fits in a single tile
    assert sorted(p.name for p in (out_dir / "0").iterdir()) == ["0_0.jpg"]
    assert Image.open(out_dir / "0" / "0_0.jpg").size == (250, 150)

    # level 1 (300x500): 2x2 tiles, edges cropped to 244/44
    assert sorted(p.name for p in (out_dir / "1").iterdir()) == ["0_0.jpg", "0_1.jpg", "1_0.jpg", "1_1.jpg"]
    assert Image.open(out_dir / "1" / "1_1.jpg").size == (244, 44)  # last col x last row

    # level 2 (maxLevel, full resolution 600x1000): 4 cols x 3 rows
    level2 = out_dir / "2"
    assert len(list(level2.iterdir())) == 12
    assert Image.open(level2 / "3_2.jpg").size == (232, 88)  # last col (x=768..1000) x last row (y=512..600)

    # reconstructing maxLevel's tiles must match the source array's dimensions
    recon = np.zeros((600, 1000, 3), np.uint8)
    for row in range(3):
        for col in range(4):
            tile = np.asarray(Image.open(level2 / f"{col}_{row}.jpg"))
            th, tw = tile.shape[:2]
            recon[row * 256: row * 256 + th, col * 256: col * 256 + tw] = tile
    assert recon.shape == arr.shape


def test_build_pyramid_single_level_when_already_small(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)

    arr = (np.random.default_rng(1).integers(0, 255, (100, 150, 3))).astype(np.uint8)
    tiles.build_pyramid(arr, "jobB")

    out_dir = tmp_path / "tiles" / "jobB"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest == {"width": 150, "height": 100, "tileSize": 256, "maxLevel": 0}
    assert sorted(p.name for p in (out_dir / "0").iterdir()) == ["0_0.jpg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_panorama_tiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.tiles'`

- [ ] **Step 3: Add `tiles_dir` to `paths.py`**

In `backend/app/core/paths.py`, add after `maps_dir`:

```python
def tiles_dir(job_id: str) -> Path: return _ensure(settings.data_dir / "tiles" / job_id)
```

- [ ] **Step 4: Write `build_pyramid`**

Create `backend/app/pipeline/tiles.py`:

```python
"""Zoomable tile pyramid for the panorama viewer — built once from the
already-decoded working array (`arr` in `analyze_panorama`), before it's
discarded. Independent of the mask/verdict pipeline: takes a plain RGB
array and a job id, nothing more."""

from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from app.core import paths

TILE_SIZE = 256
JPEG_QUALITY = 82


def build_pyramid(arr: np.ndarray, jid: str) -> None:
    """Slice `arr` into a zoomable tile pyramid on disk:
    `data/tiles/{jid}/{level}/{col}_{row}.jpg` + `data/tiles/{jid}/manifest.json`.
    Level 0 is the lowest-resolution level (whole image fits in ~1 tile);
    `maxLevel` is `arr`'s own resolution — OpenSeadragon's own level
    numbering, so the frontend needs no translation."""
    h, w = arr.shape[:2]
    levels = [arr]
    while max(levels[-1].shape[:2]) > TILE_SIZE:
        prev = levels[-1]
        ph, pw = prev.shape[:2]
        nh, nw = max(1, (ph + 1) // 2), max(1, (pw + 1) // 2)
        levels.append(cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA))
    levels.reverse()  # levels[0] = smallest, levels[-1] = full resolution
    max_level = len(levels) - 1

    out_dir = paths.tiles_dir(jid)
    for level, level_arr in enumerate(levels):
        lh, lw = level_arr.shape[:2]
        level_dir = out_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        for row, y in enumerate(range(0, lh, TILE_SIZE)):
            for col, x in enumerate(range(0, lw, TILE_SIZE)):
                tile = level_arr[y:y + TILE_SIZE, x:x + TILE_SIZE]
                Image.fromarray(tile).save(level_dir / f"{col}_{row}.jpg", "JPEG", quality=JPEG_QUALITY)

    (out_dir / "manifest.json").write_text(json.dumps({
        "width": w, "height": h, "tileSize": TILE_SIZE, "maxLevel": max_level,
    }))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_panorama_tiles.py -v`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/app/pipeline/tiles.py backend/app/core/paths.py backend/tests/test_panorama_tiles.py
git commit -m "feat(panorama): add tile pyramid builder for deep-zoom"
```

---

### Task 2: Wire pyramid generation into `analyze_panorama`

**Files:**
- Modify: `backend/app/pipeline/panorama.py`
- Test: `backend/tests/test_panorama.py` (extend)

**Interfaces:**
- Consumes: `tiles.build_pyramid(arr, jid)` from Task 1.
- Produces: nothing new for later tasks — this task's own deliverable (pyramid files on disk after a real panorama job) is what Task 3's endpoints serve, but Task 3 doesn't call anything from this task directly, it just reads `paths.tiles_dir(jid)`.

- [ ] **Step 1: Write the failing test**

In `backend/tests/test_panorama.py`, add (keep existing tests unchanged):

```python
@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_survives_tile_pyramid_failure(tmp_path, monkeypatch):
    """A broken tile pyramid must never take down the whole analysis — it's
    a display enhancement, not part of the verdict."""
    img = (np.random.default_rng(4).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()

    def boom(arr, jid):
        raise RuntimeError("disk full")
    monkeypatch.setattr(panorama.tiles, "build_pyramid", boom)

    r = panorama.analyze_panorama(str(p), cfg, "pyramidfailtest")
    assert r["mode"] == "panorama"
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_panorama.py::test_panorama_survives_tile_pyramid_failure -v`
Expected: FAIL — `AttributeError: <module 'app.pipeline.panorama'> does not have the attribute 'tiles'` (panorama.py hasn't imported `tiles` yet)

- [ ] **Step 3: Wire `build_pyramid` into `analyze_panorama`**

In `backend/app/pipeline/panorama.py`, change the import line:

```python
from app.pipeline import loader, masks
```
to:
```python
from app.pipeline import loader, masks, tiles
```

Then, in `analyze_panorama`, right after the existing overlay save (`Image.fromarray(run["overlay"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)`), add:

```python
    try:
        tiles.build_pyramid(arr, jid)
    except Exception as e:
        print(f"panorama tile pyramid failed for job {jid}: {e}")
```

So that section of `analyze_panorama` reads:

```python
    report(0.85, "сохранение оверлея")
    Image.fromarray(run["overlay"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)

    try:
        tiles.build_pyramid(arr, jid)
    except Exception as e:
        print(f"panorama tile pyramid failed for job {jid}: {e}")

    edit = run["edit_rgb"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_panorama.py -v`
Expected: all tests pass, including `test_panorama_survives_tile_pyramid_failure`

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/panorama.py backend/tests/test_panorama.py
git commit -m "feat(panorama): build tile pyramid during analysis, non-fatal on failure"
```

---

### Task 3: Tile-serving API endpoints

**Files:**
- Create: `backend/app/api/tiles.py`
- Modify: `backend/main.py` (register router)
- Test: `backend/tests/test_tiles_api.py`

**Interfaces:**
- Consumes: `paths.tiles_dir(jid)` from Task 1.
- Produces: `GET /api/tiles/{jid}/manifest.json`, `GET /api/tiles/{jid}/{level}/{col}_{row}.jpg` — consumed by the frontend in Task 5.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_tiles_api.py`:

```python
import json
from fastapi.testclient import TestClient
from main import app
from app.core import paths as core_paths


def test_get_manifest_and_tile_returns_200(tmp_path, monkeypatch):
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    out = core_paths.tiles_dir("jobA")
    (out / "manifest.json").write_text(json.dumps({"width": 10, "height": 10, "tileSize": 256, "maxLevel": 0}))
    (out / "0").mkdir()
    (out / "0" / "0_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal fake JPEG bytes

    c = TestClient(app)
    r = c.get("/api/tiles/jobA/manifest.json")
    assert r.status_code == 200
    assert r.json() == {"width": 10, "height": 10, "tileSize": 256, "maxLevel": 0}

    r = c.get("/api/tiles/jobA/0/0_0.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_get_manifest_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    c = TestClient(app)
    assert c.get("/api/tiles/missingjob/manifest.json").status_code == 404
    assert c.get("/api/tiles/missingjob/0/0_0.jpg").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_tiles_api.py -v`
Expected: FAIL — 404 routes don't exist yet (`c.get(...)` returns 404 from FastAPI's own "not found" for an unregistered route, but the manifest-content assertion fails since there's no route at all yet — you'll see a clear failure either on the 200 assertion or a connection/route error)

- [ ] **Step 3: Create the router**

Create `backend/app/api/tiles.py`:

```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core import paths

router = APIRouter()

@router.get("/tiles/{jid}/manifest.json")
def get_manifest(jid: str):
    p = paths.tiles_dir(jid) / "manifest.json"
    if not p.exists():
        raise HTTPException(404, "tile pyramid not found")
    return FileResponse(p, media_type="application/json")

@router.get("/tiles/{jid}/{level}/{col}_{row}.jpg")
def get_tile(jid: str, level: int, col: int, row: int):
    p = paths.tiles_dir(jid) / str(level) / f"{col}_{row}.jpg"
    if not p.exists():
        raise HTTPException(404, "tile not found")
    return FileResponse(p, media_type="image/jpeg")
```

- [ ] **Step 4: Register the router**

In `backend/main.py`, change:

```python
    from app.api import health, analyze, jobs, masks, report
    api = FastAPI(title="Шлиф-Web API")
    for r in (health.router, analyze.router, jobs.router, masks.router, report.router):
```
to:
```python
    from app.api import health, analyze, jobs, masks, report, tiles
    api = FastAPI(title="Шлиф-Web API")
    for r in (health.router, analyze.router, jobs.router, masks.router, report.router, tiles.router):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_tiles_api.py -v`
Expected: `2 passed`

- [ ] **Step 6: Run the full backend test suite for regressions**

Run: `cd backend && python -m pytest -q`
Expected: all tests pass (same pass/skip counts as before this plan, plus the new tests from Tasks 1–3)

- [ ] **Step 7: Commit**

```bash
git add backend/app/api/tiles.py backend/main.py backend/tests/test_tiles_api.py
git commit -m "feat(api): serve panorama tile pyramid via /api/tiles"
```

---

### Task 4: Frontend dependency + URL helpers

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Modify: `frontend/lib/api/client.ts`
- Test: `frontend/tests/client.test.mjs` (extend)

**Interfaces:**
- Produces: `tileManifestUrl(id: string): string`, `tileUrl(id: string, level: number, x: number, y: number): string` — consumed by `DeepZoomViewer.tsx` and `PanoramaZoomModal.tsx` in Task 5.
- Produces: `openseadragon` + `@types/openseadragon` installed — consumed by Task 5.

- [ ] **Step 1: Install dependencies**

Run: `cd frontend && npm install openseadragon && npm install -D @types/openseadragon`
Expected: `package.json` gains `openseadragon` under `dependencies` and `@types/openseadragon` under `devDependencies`; `package-lock.json` updates.

- [ ] **Step 2: Write the failing test**

In `frontend/tests/client.test.mjs`, change the import line:

```js
import { maskUrl, mapUrl, imageUrl, reportUrl } from "../lib/api/client.ts";
```
to:
```js
import { maskUrl, mapUrl, imageUrl, reportUrl, tileManifestUrl, tileUrl } from "../lib/api/client.ts";
```

Add a new test in the same file:

```js
test("tile url builders", () => {
  assert.strictEqual(tileManifestUrl("abc"), "/api/tiles/abc/manifest.json");
  assert.strictEqual(tileUrl("abc", 2, 3, 5), "/api/tiles/abc/2/3_5.jpg");
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL — `tileManifestUrl is not defined` (or a TS import error naming the missing export)

- [ ] **Step 4: Add the URL helpers**

In `frontend/lib/api/client.ts`, add after the existing `reportUrl` export:

```ts
export const tileManifestUrl = (id: string) => `${base}/api/tiles/${id}/manifest.json`;
export const tileUrl = (id: string, level: number, x: number, y: number) => `${base}/api/tiles/${id}/${level}/${x}_${y}.jpg`;
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd frontend && npm test`
Expected: all tests pass, including `tile url builders`

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/lib/api/client.ts frontend/tests/client.test.mjs
git commit -m "feat(frontend): add openseadragon dependency and tile URL helpers"
```

---

### Task 5: Deep-zoom viewer components

**Files:**
- Create: `frontend/components/DeepZoomViewer.tsx`
- Create: `frontend/components/PanoramaZoomModal.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Consumes: `tileManifestUrl`, `tileUrl` from Task 4.
- Produces: `PanoramaZoomModal({ jobId: string })` React component — consumed by `page.tsx` in Task 6. Exports `TileManifest` type (`{ width: number; height: number; tileSize: number; maxLevel: number }`) from `DeepZoomViewer.tsx`.

No automated test in this task: the codebase has no DOM/component test harness (`frontend/tests/*.test.mjs` only covers pure logic, e.g. `client.test.mjs`'s URL builders from Task 4) — this matches how `Corrector.tsx` and the deleted `PanoramaWorkspace.tsx` were never unit-tested either. This task is verified manually in Task 7.

- [ ] **Step 1: Create the OpenSeadragon-mounting component**

Create `frontend/components/DeepZoomViewer.tsx`:

```tsx
"use client";
import { useEffect, useRef } from "react";
import OpenSeadragon from "openseadragon";
import { tileUrl } from "@/lib/api/client";

export interface TileManifest { width: number; height: number; tileSize: number; maxLevel: number }

export function DeepZoomViewer({ jobId, manifest }: { jobId: string; manifest: TileManifest }) {
  const elRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!elRef.current) return;
    const viewer = OpenSeadragon({
      element: elRef.current,
      showNavigationControl: false,
      tileSources: {
        width: manifest.width,
        height: manifest.height,
        tileSize: manifest.tileSize,
        minLevel: 0,
        maxLevel: manifest.maxLevel,
        getTileUrl: (level: number, x: number, y: number) => tileUrl(jobId, level, x, y),
      },
    });
    return () => viewer.destroy();
  }, [jobId, manifest]);

  return <div ref={elRef} className="osd-container" />;
}
```

This file is only ever loaded client-side (via the dynamic import in Step 2), so its top-level `import OpenSeadragon from "openseadragon"` never runs during Next.js server rendering.

- [ ] **Step 2: Create the button + modal wrapper**

Create `frontend/components/PanoramaZoomModal.tsx`:

```tsx
"use client";
import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { tileManifestUrl } from "@/lib/api/client";
import { IconZoomIn } from "@/components/icons";
import type { TileManifest } from "@/components/DeepZoomViewer";

const DeepZoomViewer = dynamic(
  () => import("@/components/DeepZoomViewer").then((m) => m.DeepZoomViewer),
  { ssr: false }
);

export function PanoramaZoomModal({ jobId }: { jobId: string }) {
  const [manifest, setManifest] = useState<TileManifest | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch(tileManifestUrl(jobId))
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => { if (!cancelled) setManifest(m); })
      .catch(() => { if (!cancelled) setManifest(null); });
    return () => { cancelled = true; };
  }, [jobId]);

  if (!manifest) return null;

  return (
    <>
      <button type="button" className="btn ghost" onClick={() => setOpen(true)}>
        <IconZoomIn /> Открыть в максимальном разрешении
      </button>
      {open ? (
        <div className="deepzoom-modal">
          <button type="button" className="btn dark sm icon deepzoom-close" aria-label="Закрыть" onClick={() => setOpen(false)}>×</button>
          <DeepZoomViewer jobId={jobId} manifest={manifest} />
        </div>
      ) : null}
    </>
  );
}
```

- [ ] **Step 3: Add modal styles**

In `frontend/app/globals.css`, add after the `.zoom-hint` rule:

```css
.deepzoom-modal { position: fixed; inset: 0; z-index: 50; background: oklch(10% 0.01 258 / .92); }
.deepzoom-modal .osd-container { position: absolute; inset: 0; }
.deepzoom-close { position: absolute; top: 16px; right: 18px; z-index: 2; }
```

- [ ] **Step 4: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/components/DeepZoomViewer.tsx frontend/components/PanoramaZoomModal.tsx frontend/app/globals.css
git commit -m "feat(frontend): add deep-zoom viewer and open-at-full-resolution modal"
```

---

### Task 6: Mount the modal in the main page

**Files:**
- Modify: `frontend/app/page.tsx`

**Interfaces:**
- Consumes: `PanoramaZoomModal` from Task 5.

- [ ] **Step 1: Import and mount**

In `frontend/app/page.tsx`, add the import next to the other component imports:

```tsx
import { PanoramaZoomModal } from "@/components/PanoramaZoomModal";
```

Then, inside `infoNode`, right after the "Скачать протокол (PDF)" link block, add:

```tsx
{shown?.mode === "panorama" && jobId ? <PanoramaZoomModal jobId={jobId} /> : null}
```

So that block of `infoNode` reads:

```tsx
      {shown && jobId ? (
        <a className="btn ghost" href={reportUrl(jobId)} target="_blank" rel="noopener noreferrer">
          <IconDownload /> Скачать протокол (PDF)
        </a>
      ) : null}
      {shown?.mode === "panorama" && jobId ? <PanoramaZoomModal jobId={jobId} /> : null}
```

- [ ] **Step 2: Type-check**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat(frontend): show deep-zoom button on panorama results"
```

---

### Task 7: Manual verification

**Files:** none (verification only)

- [ ] **Step 1: Start both dev servers**

Use the `/run` skill (or manually: `cd backend && uvicorn main:app --reload` and `cd frontend && npm run dev`) to launch the app.

- [ ] **Step 2: Verify the panorama path**

Upload a large (>50 MP, per `cfg.tiling.direct_max_pixels`) sample image. Confirm:
- The job completes and shows the verdict as before.
- A "Открыть в максимальном разрешении" button appears in the sidebar.
- Clicking it opens a full-screen viewer; scrolling to zoom in shows sharper detail rather than blurring — the actual bug this plan fixes.
- Closing the modal returns to the normal report/editor view without errors.

- [ ] **Step 3: Verify the closeup path is untouched**

Upload a small (<50 MP) sample image. Confirm the "Открыть в максимальном разрешении" button does **not** appear, and editing still works exactly as before.

- [ ] **Step 4: Verify the PDF report is unaffected**

For the panorama job from Step 2, download the PDF report and confirm it renders the same as before this change.

- [ ] **Step 5: Sanity-check memory/time on the largest available real scan**

If a large real panorama sample is available (per the spec, the dataset survey's largest file is ~574 MP native, decoded to the 150 MP working-array ceiling), run it through and watch for excessive memory growth or a stall during tile generation — this can't be caught by unit tests (no gigapixel fixture in the repo).

---

## Self-Review Notes

- **Spec coverage:** every §Architecture/§Components item in the spec maps to a task — pyramid builder (Task 1), non-fatal wiring (Task 2), endpoints (Task 3), frontend dependency (Task 4), viewer + modal (Task 5), mount point (Task 6), manual checks from §Testing (Task 7).
- **Deviation from spec worth noting:** the spec described one new frontend file (`PanoramaZoomModal.tsx`); this plan splits it into two (`PanoramaZoomModal.tsx` + `DeepZoomViewer.tsx`) so the `next/dynamic(..., { ssr: false })` boundary wraps only the module that actually imports `openseadragon` — otherwise the import would still be evaluated during Next's SSR pass of the wrapping component. Same behavior and endpoints as approved; purely a file-organization detail.
- **Type consistency checked:** `TileManifest` shape (`width`, `height`, `tileSize`, `maxLevel`) is identical across the Task 1 manifest writer, the Task 3 test fixtures, and the Task 5 TypeScript interface. `tileUrl`/`tileManifestUrl` signatures match their call sites in `DeepZoomViewer.tsx`/`PanoramaZoomModal.tsx`. `panorama.tiles.build_pyramid` naming used in the Task 2 test matches the `from app.pipeline import loader, masks, tiles` import added in that same task.
