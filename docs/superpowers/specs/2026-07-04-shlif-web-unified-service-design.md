# «Шлиф-Web» — unified analysis + correction service — design

- **Date:** 2026-07-04
- **Status:** approved (design), pre-implementation
- **Repo:** `github.com/Gerrux/shlif-web` (new, private) — spun out of `hakaton_nornikel`
- **Supersedes:** the two Streamlit apps `app.py` (processing) and `annotate.py` (mask corrector)

---

## 1. Context & goal

The Nornickel hackathon MVP is a Python package (`shlif`) driving two separate Streamlit
apps: one for **processing** a полированный шлиф (segment phases → detect talc → classify ore
sort → verdict) and one for **correcting** pseudo-masks (superpixel-toggle editor). The goal is
to fold both into **one proper web service on better rails**: a **granian FastAPI** backend +
**Next.js** frontend behind **Traefik**, in one docker-compose — modelled on the `../arboweb`
stack (which already uses granian + Next.js App Router + Traefik).

The unifying product idea is a single continuous flow:

> **upload шлиф → automatic result (сорт + фазы + тальк) → «Доработать» opens an inline,
> multi-layer mask editor on the same image → edits recompute the verdict and are saved.**

«Обработка и доработка при помощи инструмента» — process and refine with a tool, on one screen.

No authentication (explicitly descoped by the owner). The organizer VM (L4 GPU) is temporarily
offline, so the stack must run and be demoable **locally on CPU** today, and light up the GPU
path unchanged when the VM returns.

## 2. Non-goals (YAGNI)

Auth · Postgres · Redis · external queue/worker (RabbitMQ/Celery) · monitoring (Grafana/Loki) ·
TLS/letsencrypt (env-gated stub only) · multi-user/collaboration · in-app model training. The
training `scripts/` stay in `hakaton_nornikel`; only the **runtime** pipeline moves here.

## 3. Architecture overview

Monorepo `shlif-web/`:

```
shlif-web/
  docker-compose.yml            # traefik + api + web
  docker-compose.override.yml   # local/CPU dev (no GPU runtime, bind mounts, hot reload)
  traefik/
    traefik.yml                 # static: entrypoints, providers
    dynamic/                    # (optional) file-provider middlewares
  backend/
    Dockerfile                  # uv + granian
    pyproject.toml
    main.py                     # granian entry: `main:app`
    app/
      api/                      # routers: analyze, jobs, masks, health
      core/                     # settings, paths, gpu detection
      jobs/                     # SQLite job store + threadpool executor
      pipeline/                 # thin wrappers over vendored shlif (closeup + panorama)
      schemas/                  # pydantic models (job, verdict, layer)
      shlif/                    # VENDORED runtime package (copied from hakaton_nornikel)
    models/                     # gitignored: classifier.pkl, unet_ore.pt, unet_talc.pt
    data/                       # gitignored: uploads/, masks/, maps/, shlif.db
    tests/
  frontend/
    Dockerfile                  # node build → next start
    package.json
    app/                        # Next.js App Router
      (analyze)/page.tsx        # the single primary screen
    components/
      corrector/                # canvas multi-layer mask editor
      verdict/                  # sort card, phase bars, «на проверку»
    lib/api/                    # typed client + TanStack Query hooks
    tests/
  README.md
```

**ML reuse = vendoring.** Copy the runtime `shlif/` package into `backend/app/shlif/` and port
`scripts/analyze_panorama.py::run_panorama` into `backend/app/pipeline/panorama.py`. Rationale:
the new repo must be self-contained and deployable without a checkout of `hakaton_nornikel`; a
git-submodule or `pip install -e ../hakaton_nornikel` would couple the two repos and break the
Docker build context. Cost: the vendored copy can drift from the original — acceptable, since the
pipeline is feature-frozen for the hackathon and only the runtime subset moves.

**Models** live in `backend/models/` (gitignored, volume-mounted). `classifier.pkl` (10 MB, CPU)
is copied over now and enables the sort card locally. `unet_*.pt` (GPU) are added on the VM later;
their absence degrades to the classical segmentation path, not a crash.

## 4. Backend (granian FastAPI)

Granian runs `main:app` with **a single worker** (the pipeline is GPU-bound — one GPU, avoid
contention). Heavy compute runs off the event loop in a **threadpool** (`run_in_executor`).

### 4.1 Endpoints

| Method + path | Purpose |
|---|---|
| `POST /api/analyze` | multipart image + `mode` (`closeup`\|`panorama`) + params → `{job_id}`. Stores upload, enqueues async job. |
| `GET /api/jobs/{id}` | poll `{status: queued\|running\|done\|error, progress, message?, result?}`. `result` = verdict + layer refs (see §4.3). |
| `GET /api/masks/{id}/{layer}.png` | a single mask layer as PNG (talc/sulfide/magnetite/matrix/normal/fine/ore). |
| `GET /api/maps/{id}/superpixels.png` | SLIC label map (16-bit PNG or packed binary) — fetched once, enables client-side superpixel toggling. |
| `GET /api/maps/{id}/darkness.png` | 8-bit grayscale darkness map — fetched once, enables the client-side threshold ("тёмные области") tool. |
| `GET /api/images/{id}.jpg` | source image, downscaled for display. |
| `POST /api/masks/{id}/{layer}` | save an edited mask PNG → persist → **recompute verdict from current masks** → return updated verdict. Correction logged. |
| `GET /api/health` | liveness + `{gpu: bool, models: {classifier, unet_ore, unet_talc}}`. |

### 4.2 Job model

- **Store:** SQLite table `jobs(id, mode, status, progress, message, created_at, result_json)` at
  `data/shlif.db`. Survives restart; status is readable even though compute is in-process.
- **Execution:** on `POST /api/analyze`, insert a `queued` row, submit the compute callable to a
  bounded `ThreadPoolExecutor`; the callable updates `status`/`progress` as it runs. Panorama
  tiling reports coarse progress. On exception → `status=error`, `message` set.
- **Why not synchronous:** panoramas take up to ~2 min → HTTP/gateway timeouts behind
  Traefik+granian. Poll instead.

### 4.3 Pipeline wrappers (`app/pipeline/`)

- `closeup.py`: `analyze_image(rgb, cfg, ore_mask?, talc_mask?, detect_talc_flag)` + the RF sort
  classifier → verdict. GPU U-Nets used when present (`unet_ore`, `unet_talc`), else classical.
- `panorama.py`: port of `run_panorama` — tile → RF sort per tile + U-Net matrix + talc candidates
  → section verdict + overlay.
- `masks.py`: produce the per-layer masks + the SLIC label map + darkness map for the editor;
  `recompute_verdict(masks)` re-runs the rule/fractions from (possibly edited) masks.
- GPU auto-detect via `torch.cuda.is_available()` guarded by a try/import so torch is optional.

### 4.4 Persistence & data layout

```
backend/data/
  shlif.db                      # sqlite: jobs, corrections
  uploads/{id}.{ext}            # original upload
  images/{id}.jpg               # downscaled display copy
  masks/{id}/{layer}.png        # per-layer masks (edited overwrite the pipeline output)
  maps/{id}/{superpixels,darkness}.png
```
Corrections logged to a `corrections` table (id, layer, n_pixels_changed, ts) for later retraining
— the flywheel the annotator was already feeding.

## 5. Frontend (Next.js App Router)

One primary screen, `app/(analyze)/page.tsx`, styled with the **«Шлиф» design tokens** (OKLCH
palette, IBM Plex Sans/Mono, dark scene, brass accent, semantic phase colours: green=обычные
срастания, red=тонкие, blue=тальк).

- **Left — image stage:** source image or overlay; mode toggle (Крупный план / Панорама);
  upload / sample picker.
- **Right — verdict panel:** RF sort card (probs), phase fractions, talc frac, «на проверку»
  badge — the same information the current `app.py` right column shows.
- **After a result:** a **«Доработать»** button switches the stage into correction mode.

### 5.1 The corrector (HTML `<canvas>`, multi-layer, all minerals)

Edits **all** mask/mineral layers, not just talc.

- **Layer selector:** тальк · сульфид · магнетит · матрица · обычные срастания · тонкие
  срастания · руда/матрица. The active layer is editable; the others render faint for context.
- **Tools:**
  - **Суперпиксель** — click toggles a SLIC cell in/out of the active layer. The label map is
    fetched **once** (`/api/maps/{id}/superpixels.png`); lookups + toggles are client-side, so it
    feels as snappy as the Streamlit `annotate.py`.
  - **Кисть / Ластик** — freehand paint/erase, size slider, instant on canvas.
  - **Тёмные области (порог)** — the darkness map is fetched once; a slider adds pixels darker
    than the threshold **within the active region** (the "тальк = рассеянная тёмная фаза"
    recipe), also usable per layer.
  - **Авто-заполнение** — seed the active layer from the pipeline's detector output.
  - Undo/redo, layer opacity, display-only brightness/filter controls (never mutate the mask,
    matching `annotate.py`), reset.
- **All edits are client-side** after the few maps load. Only **Save** round-trips
  (`POST /api/masks/{id}/{layer}`) → verdict recomputed → panel updates.

### 5.2 Data fetching

TanStack Query. `upload → job_id`, poll `GET /api/jobs/{id}` with backoff until `done`, then load
verdict + mask PNGs + superpixel/darkness maps onto canvas layers. Same-origin behind Traefik →
no CORS handling.

## 6. Data flow (end-to-end)

```
upload ─POST /analyze→ {job_id} ─poll /jobs→ done
       → result: verdict + mask PNGs + superpixel map + darkness map
       → user edits on canvas (client-side, instant)
       → Save ─POST /masks→ persist + recompute verdict → panel updates
       → correction logged (retraining flywheel)
```

## 7. Infra: Traefik + docker-compose

- **Services:** `traefik` (v3), `api` (granian), `web` (Next.js).
- **Routing (labels, trimmed arboweb pattern):** `PathPrefix(/api)` → `api`; everything else →
  `web`. Entrypoint `web` on `:80`. No TLS/letsencrypt locally; a commented/env-gated block for
  the VM later.
- **GPU gating:** the `api` service's `runtime: nvidia` / `deploy.resources` is set only in the
  production compose; `docker-compose.override.yml` (local) omits it and bind-mounts source for
  hot reload. `torch` import is optional so CPU-only images run.
- **No auth**, same-origin.

## 8. Error handling

Upload type/size validation · huge-panorama OOM guard (reuse `load_rgb(max_pixels=)` + the tiler) ·
job failure → `status=error` + `message` surfaced in the UI · model/GPU absent → classical/CPU
badge (not a crash) · idempotent mask overwrite on save · poll backoff on the client.

## 9. Testing

- **Backend (pytest):** job lifecycle (enqueue → poll → done); `analyze` on a tiny fixture image
  on the **classical path** (no GPU, no `.pt`); mask save → verdict recompute; health/GPU
  detection reports correctly with models absent.
- **Frontend (node --test, arboweb style):** API client + mask-encode/decode util; corrector
  reducer (tool switching, undo/redo, layer toggle) as a pure function.
- **End-to-end:** later, drive upload → result → edit → save locally on the classical path via the
  `verify` skill.

## 10. Risks & open questions

- **Superpixel label map transport:** ~600 labels per image → 16-bit PNG or packed `Uint16`
  binary; must decode losslessly in the browser (verify PNG re-encode doesn't dither). Fallback: a
  per-click backend round-trip if client-side lookup proves fiddly.
- **Vendored `shlif` drift:** acceptable for a frozen hackathon pipeline; note the source commit in
  `backend/app/shlif/VENDORED.md`.
- **Local model coverage:** only `classifier.pkl` is local; U-Net masks (matrix/talc) are
  unavailable until the VM returns, so the classical masks are what the corrector starts from
  today. The editor is model-agnostic, so this only affects seed quality, not the flow.

## 11. Milestones (for the implementation plan)

1. Repo scaffold + Traefik + compose skeleton (both compose files) — “hello” api + web behind Traefik.
2. Backend: vendored `shlif`, `/health`, closeup `analyze` job + poll, mask + map endpoints.
3. Frontend: upload → poll → verdict panel (design tokens).
4. Corrector: canvas + layers + the four tools + undo/redo; Save → recompute verdict.
5. Panorama mode wired through the same job/verdict path.
6. Tests + local end-to-end verify + README.
