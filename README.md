# Шлиф-Web

Unified web service for **automatic ore classification from reflected-light optical microscopy**
(полированные шлифы / аншлифы) with an **inline expert correction tool** — one continuous flow:

> upload шлиф → automatic result (сорт + фазы + тальк) → **«Доработать»** opens a multi-layer
> mask editor on the same image → edits recompute the verdict and are saved.

Replaces the two Streamlit MVP apps from the hackathon prototype (`app.py` for processing,
`annotate.py` for mask correction) with a **granian FastAPI** backend + **Next.js** frontend
behind **Traefik**, in one `docker-compose` stack. No authentication (descoped).

## Quickstart — Docker Compose

```bash
docker compose up -d --build
```

This builds and starts three containers on a shared network:

- **traefik** (port `80`, published to the host) — routes `/api/*` → `api`, everything else → `web`.
- **api** — the FastAPI backend (granian ASGI server).
- **web** — the Next.js frontend.

`docker-compose.override.yml` is picked up automatically and forces `SHLIF_FORCE_CPU=1` for local
dev (no GPU assumed). On the organizer L4 VM (or any CUDA box), run the base file only —
`docker compose -f docker-compose.yml up -d --build` — to let the GPU auto-detect.

Once it's up, visit `http://localhost/` in a browser, or check the API directly:

```bash
curl http://localhost/api/health
# {"status":"ok","gpu":false,"models":{"classifier":true,"unet_ore":false,"unet_talc":false}}
```

> **Sandboxed-environment note:** in some sandboxes (e.g. the one this was developed in), the host
> cannot reach a container's published port via `curl localhost` even though the port mapping is
> correct — the loopback path is blocked at the sandbox network layer, not by the app. If
> `curl localhost/api/...` hangs or refuses from the host, verify from *inside* the compose network
> instead, e.g.:
> ```bash
> docker run --rm --network shlif-web_shlif-net curlimages/curl:latest -fsS http://traefik/api/health
> ```
> This talks to the same Traefik router a browser would hit, just via the Docker network instead of
> the published host port — a real client (a browser on the host, or the judges' machine) is
> unaffected and uses `http://localhost/...` normally.

Stop the stack with `docker compose down`.

## Local development (without Docker)

### Backend

```bash
cd backend
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
uv pip install --python .venv/bin/python 'scikit-learn==1.7.2'  # match the deployed classifier.pkl
.venv/bin/granian --interface asgi --reload --host 0.0.0.0 --port 8000 main:app
```

(`--reload` needs the `granian[reload]` extra — `uv pip install --python .venv/bin/python 'granian[reload]'`
— if it's not already pulled in; drop the flag for a plain one-shot run.)

> `pyproject.toml` only requires `scikit-learn>=1.4` (no lock file, no hard pin — see
> `backend/app/shlif/VENDORED.md`), so a fresh `uv pip install` resolves to whatever's latest. Pin
> to `1.7.2` locally to match what's actually used to train/pickle `backend/models/classifier.pkl`
> — otherwise loading it prints a harmless but noisy `InconsistentVersionWarning` per estimator.

Run the test suite:

```bash
cd backend && .venv/bin/pytest -q
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

Run the tests / production build:

```bash
cd frontend
npm test           # node --test over tests/*.test.mjs — pure-logic unit tests
npm run build       # next build — type-checks + produces the standalone server
```

## Models

Trained model artifacts are **not committed** (gitignored: `backend/models/`, `backend/data/`,
`*.pt`, `*.pkl`, `*.onnx`) — they're large binaries that belong outside git. Drop them in
`backend/models/` (bind-mounted into the `api` container as `/app/models` via `docker-compose.yml`):

| File | Enables | Without it |
|---|---|---|
| `classifier.pkl` | The ore-sort card (RandomForest, F1 0.84 / AUC 0.92) on close-ups **and** the section verdict on panoramas | Sort card shows a "classifier недоступен" note; panorama analyze returns a surfaced error |
| `unet_ore.pt` | Ore/matrix segmentation for the panorama ore gate (IoU 0.975 vs classical 0.81) | Panorama runs the classical multi-Otsu + Lab-colour segmenter (CPU) as a graceful fallback whenever the checkpoint or torch/segmentation-models-pytorch aren't available |
| `unet_talc.pt` *(planned)* | GPU talc-zone detector — **not yet wired in this milestone** | No behaviour change: the pipeline always runs the classical darkness/texture talc heuristic |

> **U-Net wiring is deferred.** The close-up and panorama pipelines run the **classical path
> unconditionally** in this build. Dropping `unet_ore.pt` / `unet_talc.pt` into `backend/models/`
> today only flips their `/api/health` flags to `true` — it does **not** change segmentation
> output. The `analyze_image` hooks for injecting U-Net masks exist (`backend/app/shlif/analyze.py`,
> params `ore_mask`/`talc_mask`) but no caller populates them yet; GPU inference wiring is planned
> follow-up work. Only `classifier.pkl` affects results today (the sort card + panorama verdict).

> **Update:** the panorama ore/matrix gate now uses `unet_ore.pt` when it — and
> `torch`/`segmentation-models-pytorch` — are available (`backend/app/shlif/ore_unet.py`,
> wired in `backend/app/pipeline/panorama.py::_run_panorama`). Neither package is a hard
> dependency (not in `backend/pyproject.toml`, matching the existing `talc_unet.py`
> convention) — install them only on a box that will actually run inference. The
> magnetite/sulfide split inside the ore region, and `unet_talc.pt`/`unet_s2.pt`, remain
> unwired as before.

`GET /api/health` reports which of these files are **present on disk** (`"models": {"classifier":
bool, "unet_ore": bool, "unet_talc": bool}`, an existence check — not a load/deserialize check)
alongside `"gpu": bool`.

Source of the trained artifacts: the origin hackathon repo `hakaton_nornikel`, e.g.
`hakaton_nornikel/out/classifier.pkl` — training scripts stay there, only the runtime pipeline is
vendored here (`backend/app/shlif/`, see `backend/app/shlif/VENDORED.md`).

Models here: https://disk.yandex.ru/d/DXGQgHBYe7eYgA

## CPU / GPU auto-detect

The backend never hard-requires a GPU. `app.pipeline.loader.gpu_available()` checks
`torch.cuda.is_available()` at runtime (a failed/missing torch import is treated as "no GPU");
`SHLIF_FORCE_CPU=1` (set by the Compose override) short-circuits it to `False` regardless of
hardware, and `torch` is an optional import throughout — `import app.shlif` and the whole test
suite run with torch absent. In **this** milestone the analysis pipelines run the **classical
(CPU) methods unconditionally** — except for the panorama ore/matrix gate, which now conditionally
routes through the trained U-Net when it and torch are available (see the Models note above) — so
the service works end-to-end on CPU-only hardware with no model beyond `classifier.pkl`. The GPU
U-Net path (auto-detect a CUDA device + the matching `.pt` checkpoint, branch into U-Net inference,
else fall back to classical) is scaffolded — `gpu_available()`, the `/api/health` model flags, and
the `analyze_image` `ore_mask`/`talc_mask` hooks are all in place — but for close-up analysis and
talc detection the inference wiring itself is planned follow-up, not yet active (see the Models note
above).

## Architecture

- **Backend:** FastAPI served by granian (ASGI), SQLite job store + threadpool runner for
  long-running panorama jobs (queued → running → done/error), filesystem storage for
  uploads/masks/derived maps under `backend/data/`.
- **Frontend:** Next.js (App Router), TanStack Query for polling, an HTML-canvas multi-layer mask
  editor (superpixel/brush/eraser/threshold, undo/redo) in the «Шлиф» design tokens.
- **Infra:** Traefik v3 routes `/api/*` → `api`, everything else → `web`, all on one Docker
  Compose network.

Full design rationale, API contract, and data flow:
[`docs/superpowers/specs/2026-07-04-shlif-web-unified-service-design.md`](docs/superpowers/specs/2026-07-04-shlif-web-unified-service-design.md).
Task-by-task implementation plan (this repo was built against it, Tasks 1–14):
[`docs/superpowers/plans/2026-07-04-shlif-web-unified-service.md`](docs/superpowers/plans/2026-07-04-shlif-web-unified-service.md).

## Status

Implemented end-to-end and verified: backend pytest suite green, frontend unit tests + production
build green, and a full containerized round-trip (`docker compose up` → upload a close-up through
Traefik → poll the job → fetch the phase mask) confirmed working on the classical (CPU) path with
the trained sort classifier in place.
