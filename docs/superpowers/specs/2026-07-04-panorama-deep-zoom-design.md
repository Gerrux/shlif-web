# Panorama deep-zoom — design

- **Date:** 2026-07-04
- **Status:** draft — pending review
- **Repo:** `shlif-web`
- **Relates to:** `docs/superpowers/specs/2026-07-04-panorama-closeup-unification-design.md` (§2 explicitly
  leaves "panorama view at maximum resolution with zoom" as an unblocked, separate follow-up — this is
  that follow-up). That spec has already landed in the backend (`panorama.py`, `masks.py`,
  `Corrector.tsx` unification); this design is written against that current state, not the pre-unification
  code.

## Problem

Zooming into the panorama view does not reveal more detail. Both closeup and panorama jobs now share one
viewer, `frontend/components/corrector/Corrector.tsx`, which draws onto a `<canvas>` fixed at
`size = [w, h]` — the "editor resolution" — and zooms it purely via CSS `transform: scale()`
(`frontend/lib/useZoomPan.ts`), up to 12x. For panorama jobs, `size` is capped at
`EDIT_MAX_SIDE = 2400` px per side (`backend/app/pipeline/masks.py:50`, ~5.76 MP) — deliberately, so
editing stays fast and identical across both modes.

The backend already decodes the full scan into a working array (`arr`, via
`load_working_array(path, cfg.tiling)` in `analyze_panorama`) at up to the existing
`cfg.tiling.max_pixels` ceiling (150,000,000 px, `backend/app/config/default.yaml:36`) — the same
ceiling classification already trusts. That array is used to build the ≤2400px edit copy and then
discarded; nothing above edit resolution is ever persisted. So zooming in the UI just magnifies pixels
that were thrown away before the browser ever saw them — no frontend fix can address this without also
serving more resolution.

## Goals

- Give the panorama viewer a way to show real additional detail on zoom, up to the resolution the
  pipeline already decodes for classification (`cfg.tiling.max_pixels`, 150 MP) — no new decode ceiling
  is introduced.
- Zero regression to editing: `EDIT_MAX_SIDE`, `Corrector.tsx`, and the mask-editing flow are untouched.
- Zero regression for closeup jobs, the PDF report, or jobs analyzed before this ships.

## Non-goals

- **Closeup deep-zoom.** Closeup sources top out at ~26 MP (per the unification spec's dataset survey)
  against a 2400px edit copy — a much smaller gap than panorama's 126–574 MP sources, and not what was
  reported broken. Not built now; the mechanism below is intentionally structured so adding it later is
  a small, separate change (see §Architecture).
- **True unbounded/streaming gigapixel decode** (pyvips region reads). The pyramid is built from `arr`,
  which is already capped at `cfg.tiling.max_pixels` — the same limitation classification already has
  today. Not a new constraint introduced by this work.
- **Color-coded ore/talc overlay in the deep-zoom view.** The pyramid is built from the raw stitched
  `arr`, before `_run_panorama`'s sort-tint/talc-tint blending. That overlay is a coarse, feathered
  per-tile-blob tint — the reason to zoom in is to see actual grain/texture detail (relevant to the
  sulfide grain-size liberation rule), which the tint would only obscure.
- **Editing at high zoom.** View-only, same as the panorama viewing experience always was.
- **Tile retention/cleanup policy.** `data/uploads`, `data/images`, `data/masks`, `data/maps` have no
  retention policy today either; not new debt introduced here.

## Architecture

`analyze_panorama()` (`backend/app/pipeline/panorama.py`) already loads `arr` once and threads it through
`_assemble_masks` and `_run_panorama`. After both complete successfully, it calls a new
`tiles.build_pyramid(arr, jid)`, wrapped in `try/except` — a failure here is logged and swallowed, never
fails the job or drops the verdict. No changes to `_run_panorama`, `_assemble_masks`, or their signatures;
this is a single additive call site using data that's already in memory, so there's no second decode of
the source file.

Scoped to panorama-mode jobs only (§Non-goals). The pyramid builder takes a plain `np.ndarray` and a
`jid` — it has no idea whether the caller is panorama or closeup, so if closeup deep-zoom is wanted
later, it's the same function called from `analyze.py`'s closeup branch with its own array, not a
redesign.

**Backend**

- `backend/app/pipeline/tiles.py` (new) — `build_pyramid(arr: np.ndarray, jid: str) -> None`. Builds
  successive half-scale levels starting from the smallest (the longest side fits in one tile) and
  doubling resolution each level up to full `arr` resolution, slices each level into non-overlapping
  `256x256` tiles, writes `data/tiles/{jid}/{level}/{col}_{row}.jpg` (JPEG, quality=82) and
  `data/tiles/{jid}/manifest.json`: `{"width": int, "height": int, "tileSize": 256, "maxLevel": int}`.
  Level 0 is the lowest-resolution level (whole image in ~1 tile); `maxLevel` is the highest-resolution
  level and matches `arr`'s full pixel dimensions — this is OpenSeadragon's own level numbering
  convention, so the frontend needs no translation layer.
- `backend/app/pipeline/panorama.py` — one new import, one new call:
  `try: tiles.build_pyramid(arr, jid) except Exception: log + continue`, placed after `run = _run_panorama(...)`.
- `backend/app/api/tiles.py` (new) — mirrors `backend/app/api/masks.py`'s existing pattern:
  - `GET /api/tiles/{jid}/manifest.json` → `FileResponse` or 404.
  - `GET /api/tiles/{jid}/{level}/{col}_{row}.jpg` → `FileResponse` or 404.
- No config changes. The pyramid's finest level is simply `arr` as decoded — already governed by
  `cfg.tiling.max_pixels`, which is the one existing knob that controls both classification and (now)
  display detail.

**Frontend**

- `frontend/components/PanoramaZoomModal.tsx` (new, self-contained) — takes `jobId`. On mount, fetches
  `/api/tiles/{jobId}/manifest.json`; renders nothing if it 404s or errors (covers: closeup jobs, jobs
  analyzed before this ships, jobs where pyramid generation failed). On success, renders a button
  ("Открыть в максимальном разрешении"); clicking it opens a full-screen overlay with OpenSeadragon
  (dynamically imported, `next/dynamic` with `ssr: false`), configured with a custom tile source whose
  `getTileUrl(level, x, y)` points at the endpoints above.
- `frontend/app/page.tsx` — mount `<PanoramaZoomModal jobId={jobId} />` inside the existing `infoNode`,
  next to the "Скачать протокол (PDF)" link, gated on `shown?.mode === "panorama"` (a cheap client-side
  pre-check so closeup jobs never issue the manifest fetch at all — not required for correctness, just
  avoids a permanently-404ing request for every closeup job).
- `Corrector.tsx` and `useZoomPan.ts` — **unchanged**. This is fully additive; the editing canvas and its
  zoom never interact with the pyramid.
- New dependency: `openseadragon`.

## Data flow

```
analyze_panorama()
  → arr = load_working_array(path, cfg.tiling)        (unchanged, up to cfg.tiling.max_pixels)
  → _assemble_masks(...), _run_panorama(...)            (unchanged)
  → try: tiles.build_pyramid(arr, jid)                  (new — writes data/tiles/{jid}/**)
  → save data/images/{jid}.jpg from run["overlay"]      (unchanged path/size)
  → JobStore.set_status(jid, "done", ...)

GET /api/tiles/{jid}/manifest.json      → 200 + JSON, or 404 (no pyramid: closeup / pre-feature / failed build)
GET /api/tiles/{jid}/{level}/{col}_{row}.jpg → 200 + JPEG, or 404

frontend: PanoramaZoomModal fetches manifest on mount
  → 200 ⇒ show "открыть в максимальном разрешении" button ⇒ OpenSeadragon on click
  → 404 / error ⇒ render nothing
```

## Error handling

- `build_pyramid` failure (disk I/O, PIL/cv2 error) → caught in `analyze_panorama`, logged; job still
  completes with its verdict and `{jid}.jpg`. No `manifest.json` ⇒ modal renders nothing.
- Tile/manifest requested for a `jid` with no pyramid → 404, same pattern as `masks.py`'s `get_mask`/`get_image`.
- Manifest fetch failure on the frontend (network error, non-404 5xx) → treated the same as 404: render
  nothing rather than a broken button.

## Testing

- `backend/tests/test_panorama_tiles.py` (new): synthetic RGB array (same `np.random.default_rng`
  pattern as `test_panorama.py`), isolated via `monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)`
  (same pattern already used in `test_masks.py::test_persist_editor_artifacts_writes_all_files`). Assert:
  level count, tile count per level, correct pixel dimensions of edge (non-square) tiles, manifest field
  values, and that the `maxLevel` tiles reconstruct to the original array's dimensions.
- `backend/tests/test_panorama.py` (extend): monkeypatch `tiles.build_pyramid` to raise, assert
  `analyze_panorama` still returns a normal result (verifies the try/except actually protects the job).
- Manual verification via the `/run` skill in a real browser: confirm zoom actually sharpens instead of
  blurring on a real panorama upload, confirm the button doesn't appear for a closeup upload, confirm the
  PDF report still renders unchanged. No gigapixel sample exists in the repo's test fixtures, so peak
  memory/time on a realistically large scan (the dataset survey's `Панорамы/16.jpg`, 574 MP native) can
  only be checked this way, not by unit test — this was already a known cost of `arr`'s existing 150 MP
  ceiling, not something this feature adds, but worth eyeballing since we now also write ~1.33x that many
  pixels back out as tiles.
