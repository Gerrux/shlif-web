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
.venv/bin/granian --interface asgi --reload --host 0.0.0.0 --port 8000 main:app
```

(`--reload` needs the `granian[reload]` extra — `uv pip install --python .venv/bin/python 'granian[reload]'`
— if it's not already pulled in; drop the flag for a plain one-shot run.)

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
| `unet_ore.pt` *(optional)* | GPU-accelerated ore/matrix segmentation | Falls back to the classical multi-Otsu + Lab-colour segmenter (CPU, always available) |
| `unet_talc.pt` *(optional)* | GPU-accelerated talc-zone detector | Falls back to the classical darkness/texture talc heuristic |

`GET /api/health` reports which of these are actually loaded (`"models": {"classifier": bool,
"unet_ore": bool, "unet_talc": bool}`) alongside `"gpu": bool`.

Source of the trained artifacts: the origin hackathon repo `hakaton_nornikel`, e.g.
`hakaton_nornikel/out/classifier.pkl` — training scripts stay there, only the runtime pipeline is
vendored here (`backend/app/shlif/`, see `backend/app/shlif/VENDORED.md`).

## CPU / GPU auto-detect

The backend never hard-requires a GPU. `app.pipeline.loader.gpu_available()` checks
`torch.cuda.is_available()` at runtime (a failed/missing torch import is treated as "no GPU");
`SHLIF_FORCE_CPU=1` (set by the Compose override) short-circuits it to `False` regardless of
hardware. Code paths that can use a U-Net check for the corresponding `.pt` checkpoint at load
time and silently fall back to the classical (non-GPU) implementation when it's absent — so the
whole pipeline runs end-to-end on CPU-only hardware with the classical methods, and picks up GPU
acceleration automatically when both a CUDA device and the matching checkpoint are present (e.g.
redeployed on the organizer's L4 VM).

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
