# Шлиф-Web Unified Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two Streamlit MVP apps (processing + mask annotation) with one web service — a granian FastAPI backend + Next.js frontend behind Traefik — where you upload a шлиф, get an automatic verdict, and refine every mineral mask inline with a canvas tool that recomputes the verdict.

**Architecture:** Monorepo. The runtime `shlif` pipeline is **vendored** into `backend/app/shlif/`. The backend serves an async job API (SQLite job store + threadpool executor) so minute-long panoramas never time out; GPU is auto-detected with a classical/CPU fallback so it runs locally today. The frontend polls the job, renders the verdict with the «Шлиф» design tokens, and opens an HTML-canvas multi-layer mask editor whose edits POST back and recompute the phase-composition verdict. Traefik routes `/api → api`, everything else `→ web`.

**Tech Stack:** Python 3.12 · FastAPI · granian · numpy/opencv-headless/scikit-image/scikit-learn/scipy/pandas/pyyaml/pillow · SQLite (stdlib) · Next.js (App Router) · TanStack Query · TypeScript · Traefik v3 · Docker Compose.

## Global Constraints

- **No authentication** — descoped by the owner. Same-origin behind Traefik (no CORS config).
- **Python floor 3.12**; **torch is an OPTIONAL import** — `import shlif` and all tests must run with torch absent (classical/CPU path). Never import `shlif.talc_unet` or torch at module top level in vendored/pipeline code; import lazily inside the function that needs the GPU.
- **granian runs `main:app` with a single worker** (GPU-bound). Heavy compute runs in a threadpool, never inline in the request.
- **Phases are one exclusive label map** — every pixel is exactly one of `MATRIX=0 / MAGNETITE=1 / SULFIDE=2` (`shlif.phases`). **Talc is a separate binary overlay.** обычные/тонкие срастания and руда are **derived/computed, never hand-painted**.
- **A mask edit recomputes only the phase-composition verdict** (fractions + rule → рядовая/труднообог./оталькованная). The **RF sort card is texture-based on the source image and does NOT change** on mask edits.
- **Models live in `backend/models/` (gitignored)**: `classifier.pkl` (RF sort, CPU), `unet_ore.pt`, `unet_talc.pt` (GPU, optional). Absent models → classical path + a badge, never a crash.
- **Data lives in `backend/data/` (gitignored)**: `shlif.db`, `uploads/`, `images/`, `masks/{id}/`, `maps/{id}/`.
- **UI copy is Russian** (jury-facing). Design tokens: OKLCH palette, IBM Plex Sans/Mono, dark image scene, brass accent; phase colours green=обычные срастания, red=тонкие, blue=тальк.
- **Commit after every task.** Commit-trailer for this repo: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

```
shlif-web/
  docker-compose.yml               # traefik + api + web (Task 3, extended Task 12)
  docker-compose.override.yml      # local/CPU dev
  traefik/traefik.yml              # static config
  backend/
    Dockerfile                     # uv + granian
    pyproject.toml
    main.py                        # create_app(); granian entry main:app
    app/
      __init__.py
      core/{__init__,settings,paths}.py
      shlif/                       # VENDORED copy of hakaton_nornikel/shlif/ (+ VENDORED.md)
      config/default.yaml          # VENDORED copy of hakaton_nornikel/config/default.yaml
      schemas/{__init__,jobs}.py   # pydantic: JobRecord, Verdict, AnalyzeResult
      jobs/{__init__,store,runner}.py
      pipeline/{__init__,loader,closeup,panorama,masks}.py
      api/{__init__,health,analyze,jobs,masks}.py
    tests/
      conftest.py                  # synthetic fixture image
      test_health.py test_loader.py test_pipeline.py test_masks.py
      test_jobs.py test_api.py test_panorama.py
    models/  data/                 # gitignored (runtime)
  frontend/
    Dockerfile
    package.json  next.config.mjs  tsconfig.json  postcss.config.mjs
    app/
      layout.tsx  globals.css      # «Шлиф» design tokens
      page.tsx                     # the analyze screen
    components/
      verdict/{SortCard,PhaseBars,VerdictPanel}.tsx
      corrector/{Corrector,toolbar}.tsx  corrector/reducer.ts
    lib/
      api/{client,types,hooks}.ts
      mask/{encode,superpixel}.ts
    tests/                         # node --test on lib/
  README.md
  docs/superpowers/{specs,plans}/
```

---

## Task 1: Backend skeleton + health endpoint

**Files:**
- Create: `backend/pyproject.toml`, `backend/main.py`, `backend/app/__init__.py`, `backend/app/core/__init__.py`, `backend/app/core/settings.py`, `backend/app/core/paths.py`, `backend/app/api/__init__.py`, `backend/app/api/health.py`, `backend/Dockerfile`, `backend/tests/__init__.py`, `backend/tests/test_health.py`
- Modify: none

**Interfaces:**
- Produces: `create_app() -> FastAPI` in `main.py`; `app = create_app()` module global. Health route `GET /api/health` → `{"status":"ok","gpu":bool,"models":{"classifier":bool,"unet_ore":bool,"unet_talc":bool}}`. `Settings` in `core/settings.py` with `.data_dir`, `.models_dir`, `.max_display_px`. Path helpers in `core/paths.py`: `job_dir(id)`, `masks_dir(id)`, `maps_dir(id)`, `uploads_dir()`, `images_dir()`, `db_path()`.

- [ ] **Step 1: Write `backend/pyproject.toml`**

```toml
[project]
name = "shlif-web-backend"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "granian>=1.6",
    "python-multipart>=0.0.9",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "numpy>=1.26",
    "pillow>=10.3",
    "opencv-python-headless>=4.9",
    "scikit-image>=0.24",
    "scikit-learn>=1.4",
    "scipy>=1.11",
    "pandas>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `backend/app/core/settings.py`**

```python
from __future__ import annotations
from pathlib import Path

class Settings:
    """Runtime paths + limits. Env override via SHLIF_DATA_DIR / SHLIF_MODELS_DIR."""
    def __init__(self) -> None:
        import os
        root = Path(__file__).resolve().parents[2]  # backend/
        self.data_dir = Path(os.environ.get("SHLIF_DATA_DIR", root / "data"))
        self.models_dir = Path(os.environ.get("SHLIF_MODELS_DIR", root / "models"))
        self.max_display_px = int(os.environ.get("SHLIF_MAX_DISPLAY_PX", 4_000_000))

settings = Settings()
```

- [ ] **Step 3: Write `backend/app/core/paths.py`**

```python
from __future__ import annotations
from pathlib import Path
from .settings import settings

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def uploads_dir() -> Path: return _ensure(settings.data_dir / "uploads")
def images_dir() -> Path: return _ensure(settings.data_dir / "images")
def masks_dir(job_id: str) -> Path: return _ensure(settings.data_dir / "masks" / job_id)
def maps_dir(job_id: str) -> Path: return _ensure(settings.data_dir / "maps" / job_id)
def db_path() -> Path:
    _ensure(settings.data_dir)
    return settings.data_dir / "shlif.db"
```

- [ ] **Step 4: Write the failing test `backend/tests/test_health.py`**

```python
from fastapi.testclient import TestClient
from main import app

def test_health_ok():
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["gpu"] is False  # no torch in test env
    assert set(body["models"]) == {"classifier", "unet_ore", "unet_talc"}
```

- [ ] **Step 5: Write `backend/app/api/health.py`**

```python
from __future__ import annotations
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health() -> dict:
    # Task 4 replaces the stub bodies with real detection via pipeline.loader.
    return {
        "status": "ok",
        "gpu": False,
        "models": {"classifier": False, "unet_ore": False, "unet_talc": False},
    }
```

- [ ] **Step 6: Write `backend/main.py`**

```python
from __future__ import annotations
from fastapi import FastAPI
from app.api import health

def create_app() -> FastAPI:
    app = FastAPI(title="Шлиф-Web API")
    app.include_router(health.router, prefix="/api")
    return app

app = create_app()
```

- [ ] **Step 7: Install deps and run the test**

Run:
```bash
cd backend && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/pytest tests/test_health.py -v
```
Expected: `test_health_ok PASSED`.

- [ ] **Step 8: Write `backend/Dockerfile`**

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -e '.[dev]'
COPY . .
EXPOSE 8000
CMD ["granian","--interface","asgi","--host","0.0.0.0","--port","8000","--workers","1","--runtime-threads","2","--http","auto","--respawn-failed-workers","main:app"]
```

- [ ] **Step 9: Commit**

```bash
cd .. && git add backend && git commit -m "feat(backend): skeleton FastAPI app + /api/health + granian Dockerfile"
```

---

## Task 2: Frontend skeleton (Next.js hello)

**Files:**
- Create: `frontend/package.json`, `frontend/next.config.mjs`, `frontend/tsconfig.json`, `frontend/app/layout.tsx`, `frontend/app/page.tsx`, `frontend/Dockerfile`, `frontend/.dockerignore`

**Interfaces:**
- Produces: a Next.js App Router app that builds and serves a root page. Later tasks replace `app/page.tsx` with the analyze screen and add `app/globals.css`.

- [ ] **Step 1: Write `frontend/package.json`**

```json
{
  "name": "shlif-web-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start -p 3000",
    "test": "node --test tests/"
  },
  "dependencies": {
    "next": "^15.1.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "@tanstack/react-query": "^5.59.0"
  },
  "devDependencies": {
    "typescript": "^5.6.0",
    "@types/node": "^22",
    "@types/react": "^19",
    "@types/react-dom": "^19"
  }
}
```

- [ ] **Step 2: Write `frontend/next.config.mjs`**

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = { output: "standalone", reactStrictMode: true };
export default nextConfig;
```

- [ ] **Step 3: Write `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022", "lib": ["dom", "dom.iterable", "es2022"],
    "allowJs": true, "skipLibCheck": true, "strict": true, "noEmit": true,
    "esModuleInterop": true, "module": "esnext", "moduleResolution": "bundler",
    "resolveJsonModule": true, "isolatedModules": true, "jsx": "preserve",
    "incremental": true, "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 4: Write `frontend/app/layout.tsx`**

```tsx
export const metadata = { title: "Шлиф · классификация руд", description: "Обработка и доработка шлифов" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
```

- [ ] **Step 5: Write `frontend/app/page.tsx`**

```tsx
export default function Home() {
  return <main style={{ padding: 24, fontFamily: "system-ui" }}>Шлиф-Web — сервис готовится…</main>;
}
```

- [ ] **Step 6: Write `frontend/Dockerfile`**

```dockerfile
FROM node:20-bookworm-slim AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-bookworm-slim AS run
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

- [ ] **Step 7: Write `frontend/.dockerignore`**

```
node_modules
.next
```

- [ ] **Step 8: Verify the build**

Run:
```bash
cd frontend && npm install && mkdir -p public && npm run build
```
Expected: `✓ Compiled successfully` / `Route (app) /` listed. (The build is the test here — a broken page fails the build.)

- [ ] **Step 9: Commit**

```bash
cd .. && git add frontend && git commit -m "feat(frontend): Next.js App Router skeleton + standalone Dockerfile"
```

---

## Task 3: Traefik + docker-compose skeleton (rails end-to-end)

**Files:**
- Create: `traefik/traefik.yml`, `docker-compose.yml`, `docker-compose.override.yml`

**Interfaces:**
- Produces: Traefik on `:80` routing `PathPrefix(/api) → api:8000`, everything else `→ web:3000`. `docker compose up` brings all three up; `curl http://localhost/api/health` and `curl http://localhost/` both answer.

- [ ] **Step 1: Write `traefik/traefik.yml`**

```yaml
entryPoints:
  web:
    address: ":80"
providers:
  docker:
    exposedByDefault: false
api:
  dashboard: false
log:
  level: INFO
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  traefik:
    image: traefik:v3.2
    command:
      - "--configFile=/etc/traefik/traefik.yml"
    ports:
      - "80:80"
    volumes:
      - ./traefik/traefik.yml:/etc/traefik/traefik.yml:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks: [shlif-net]

  api:
    build: ./backend
    environment:
      - SHLIF_DATA_DIR=/app/data
      - SHLIF_MODELS_DIR=/app/models
    volumes:
      - ./backend/data:/app/data
      - ./backend/models:/app/models
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=PathPrefix(`/api`)"
      - "traefik.http.routers.api.entrypoints=web"
      - "traefik.http.routers.api.priority=100"
      - "traefik.http.services.api.loadbalancer.server.port=8000"
    networks: [shlif-net]

  web:
    build: ./frontend
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.web.rule=PathPrefix(`/`)"
      - "traefik.http.routers.web.entrypoints=web"
      - "traefik.http.routers.web.priority=1"
      - "traefik.http.services.web.loadbalancer.server.port=3000"
    networks: [shlif-net]

networks:
  shlif-net:
```

- [ ] **Step 3: Write `docker-compose.override.yml` (local/CPU dev, no GPU)**

```yaml
# Loaded automatically by `docker compose`. Keeps local dev CPU-only.
# For the GPU VM, run: docker compose -f docker-compose.yml up  (skip the override).
services:
  api:
    environment:
      - SHLIF_FORCE_CPU=1
```

- [ ] **Step 4: Validate compose config**

Run: `docker compose config >/dev/null && echo OK`
Expected: `OK` (no schema errors).

- [ ] **Step 5: Build + up + integration smoke**

Run:
```bash
docker compose up -d --build
sleep 8
curl -fsS http://localhost/api/health | grep -q '"status":"ok"' && echo API_OK
curl -fsS http://localhost/ | grep -q 'Шлиф-Web' && echo WEB_OK
docker compose down
```
Expected: `API_OK` and `WEB_OK`.

- [ ] **Step 6: Commit**

```bash
git add traefik docker-compose.yml docker-compose.override.yml && git commit -m "feat(infra): Traefik v3 + docker-compose (traefik+api+web), /api routed, CPU override"
```

---

## Task 4: Vendor the `shlif` pipeline + model/GPU loader

**Files:**
- Create: `backend/app/shlif/` (copied), `backend/app/config/default.yaml` (copied), `backend/app/shlif/VENDORED.md`, `backend/app/pipeline/__init__.py`, `backend/app/pipeline/loader.py`, `backend/tests/test_loader.py`
- Modify: `backend/app/api/health.py`

**Interfaces:**
- Consumes: `shlif.load_config`, `shlif.analyze_image`, `shlif.phases` (from the vendored package).
- Produces: `loader.py` — `get_config() -> Config` (cached), `load_classifier() -> tuple[clf, list[str], list[str]] | None` (reads `models/classifier.pkl`; pickle dict keys `clf`/`feature_names`/`classes`), `gpu_available() -> bool` (honours `SHLIF_FORCE_CPU`, lazy torch import), `model_status() -> dict` (`{"classifier":bool,"unet_ore":bool,"unet_talc":bool}`).

- [ ] **Step 1: Vendor the package and config**

Run:
```bash
cp -r ../hakaton_nornikel/shlif backend/app/shlif
cp ../hakaton_nornikel/config/default.yaml backend/app/config/default.yaml
rm -rf backend/app/shlif/__pycache__
( cd ../hakaton_nornikel && git rev-parse HEAD ) > /tmp/src_commit
```
Note the config path: `shlif/config.py` resolves `Path(__file__).parent.parent/"config"/"default.yaml"` → `backend/app/config/default.yaml`. The copy above puts it exactly there.

- [ ] **Step 2: Write `backend/app/shlif/VENDORED.md`**

```markdown
# Vendored `shlif` runtime package
Copied from `hakaton_nornikel/shlif/` at source commit (see below). Training
scripts stay in the origin repo. Only the runtime pipeline lives here.
Do NOT import `shlif.talc_unet` or torch at module top level — GPU is optional.

Source commit: <paste the hash printed by `git rev-parse HEAD` in Step 1>
```
Paste the hash from `/tmp/src_commit` into the file.

- [ ] **Step 3: Write the failing test `backend/tests/test_loader.py`**

```python
from app.pipeline import loader

def test_config_loads():
    cfg = loader.get_config()
    assert cfg.rule.talc_threshold is not None

def test_classifier_absent_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(loader.settings, "models_dir", tmp_path)
    loader.load_classifier.cache_clear()
    assert loader.load_classifier() is None

def test_gpu_false_without_torch():
    assert loader.gpu_available() is False

def test_model_status_shape():
    s = loader.model_status()
    assert set(s) == {"classifier", "unet_ore", "unet_talc"}
```

- [ ] **Step 4: Write `backend/app/pipeline/loader.py`**

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

def gpu_available() -> bool:
    if os.environ.get("SHLIF_FORCE_CPU") == "1":
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

def model_status() -> dict:
    md = settings.models_dir
    return {
        "classifier": (md / "classifier.pkl").exists(),
        "unet_ore": (md / "unet_ore.pt").exists(),
        "unet_talc": (md / "unet_talc.pt").exists(),
    }
```

- [ ] **Step 5: Update `backend/app/api/health.py` to report real status**

```python
from __future__ import annotations
from fastapi import APIRouter
from app.pipeline import loader

router = APIRouter()

@router.get("/health")
def health() -> dict:
    return {"status": "ok", "gpu": loader.gpu_available(), "models": loader.model_status()}
```

- [ ] **Step 6: Run tests**

Run: `cd backend && .venv/bin/pip install -e '.[dev]' >/dev/null; .venv/bin/pytest tests/test_loader.py tests/test_health.py -v`
Expected: all PASS (health still `gpu False`, models all False locally).

- [ ] **Step 7: Commit**

```bash
cd .. && git add backend && git commit -m "feat(backend): vendor shlif runtime + config; model/GPU loader; real /health status"
```

---

## Task 5: Closeup pipeline wrapper + verdict recompute + editor maps

**Files:**
- Create: `backend/app/pipeline/closeup.py`, `backend/app/pipeline/masks.py`, `backend/tests/conftest.py`, `backend/tests/test_pipeline.py`, `backend/tests/test_masks.py`
- Modify: `backend/app/shlif/analyze.py` (extract a reusable `verdict_from_masks`)

**Interfaces:**
- Consumes: `loader.get_config`, `loader.load_classifier`, `shlif.analyze_image`, `shlif.phases`, `shlif.features.extract_features`, `shlif.talc.talc_fraction`.
- Produces:
  - `masks.verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px=12) -> dict` (fractions + rule; the phase-composition half of the verdict).
  - `masks.phase_label_map(sulfide, magnetite) -> np.ndarray` (uint8 HxW, 0/1/2).
  - `masks.split_phase_map(phase_map) -> tuple[sulfide, magnetite, matrix]` (bool masks).
  - `masks.build_superpixel_map(rgb, n_segments=600) -> np.ndarray` (uint16 HxW SLIC labels).
  - `masks.build_darkness_map(rgb) -> np.ndarray` (uint8 HxW grayscale).
  - `masks.encode_png_gray(arr) -> bytes`, `masks.decode_png_gray(data) -> np.ndarray`, `masks.encode_png_u16(arr) -> bytes`.
  - `closeup.analyze_closeup(rgb, cfg) -> dict` with keys `verdict` (from `verdict_from_masks`), `sort` (`{classes: {name:prob}, top: name}` or `None` if no classifier), `phase_map` (uint8), `talc` (bool), `superpixels` (uint16), `darkness` (uint8), `text` (str).

- [ ] **Step 1: Refactor `backend/app/shlif/analyze.py` — extract `verdict_from_masks`**

Replace the body of `analyze_image` from the `normal, fine = _intergrowth_split(...)` line through the `metrics = {...}` / `text = _verdict_text(...)` block with a call to a new public function, and add that function. Insert **before** `analyze_image`:

```python
def verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px: int = 12) -> dict:
    """Phase-composition verdict from (already-decided) phase masks + talc overlay.
    Returns {ore_class, text, metrics}. Shared by analyze_image and the web recompute."""
    talc_frac = talc_fraction(talc)
    normal, fine = _intergrowth_split(sulfide, magnetite, dist_px)
    sulf_area = float(sulfide.sum())
    fine_share = float(fine.sum()) / sulf_area if sulf_area > 0 else 0.0
    normal_share = 1.0 - fine_share

    rule = cfg.rule
    talc_thr = float(rule.talc_threshold)
    dom_thr = float(rule.dominance_threshold)
    if talc_frac > talc_thr:
        ore = phases.ORE_TALCOSE
        confidence = min(1.0, (talc_frac - talc_thr) / max(talc_thr, 1e-6) + 0.5)
    else:
        margin = abs(fine_share - dom_thr) / max(dom_thr, 1e-6)
        confidence = min(1.0, 0.5 + margin)
        ore = phases.ORE_HARD if fine_share > dom_thr else phases.ORE_ORDINARY
        if confidence < float(rule.fine_min_confidence):
            ore = phases.ORE_REVIEW

    total = matrix.size
    metrics = {
        "sulfide_frac": float(sulfide.sum()) / total,
        "magnetite_frac": float(magnetite.sum()) / total,
        "matrix_frac": float(matrix.sum()) / total,
        "talc_frac": talc_frac,
        "normal_share": normal_share,
        "fine_share": fine_share,
        "confidence": confidence,
    }
    return {"ore_class": ore, "text": _verdict_text(ore, metrics),
            "metrics": metrics, "normal": normal, "fine": fine}
```

Then in `analyze_image`, replace the block that computed `normal, fine, ...metrics..., text` with:

```python
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px)
    ore, text, metrics, normal, fine = v["ore_class"], v["text"], v["metrics"], v["normal"], v["fine"]
```

(Leave the `masks = {...}` dict and `return Analysis(...)` unchanged.)

- [ ] **Step 2: Write `backend/tests/conftest.py` (synthetic fixture)**

```python
import numpy as np, pytest

@pytest.fixture
def tiny_rgb():
    """256x256 RGB: dark matrix with a couple of bright blobs (sulfide) and a grey blob."""
    rng = np.random.default_rng(0)
    img = (rng.integers(8, 28, (256, 256, 3))).astype(np.uint8)  # dark matrix
    img[40:110, 40:110] = 220   # bright sulfide blob
    img[150:210, 150:210] = 120  # mid-grey magnetite blob
    return img
```

- [ ] **Step 3: Write the failing test `backend/tests/test_masks.py`**

```python
import numpy as np
from app.pipeline import masks
from app.pipeline import loader

def test_phase_map_roundtrip():
    s = np.zeros((10, 10), bool); s[0:3, 0:3] = True
    m = np.zeros((10, 10), bool); m[5:8, 5:8] = True
    pm = masks.phase_label_map(s, m)
    assert pm.dtype == np.uint8 and set(np.unique(pm)) <= {0, 1, 2}
    su, mg, mx = masks.split_phase_map(pm)
    assert (su == s).all() and (mg == m).all() and (mx == ~(s | m)).all()

def test_verdict_from_masks_reacts_to_talc():
    cfg = loader.get_config()
    s = np.zeros((100, 100), bool); s[:10] = True
    m = np.zeros((100, 100), bool)
    mx = ~(s | m)
    no_talc = masks.verdict_from_masks_dict(s, m, mx, np.zeros((100,100), bool), cfg)
    lots = np.zeros((100, 100), bool); lots[50:] = True
    talcy = masks.verdict_from_masks_dict(s, m, mx & lots | mx, lots & mx, cfg)
    assert talcy["metrics"]["talc_frac"] > no_talc["metrics"]["talc_frac"]

def test_superpixel_and_darkness_maps(tiny_rgb):
    sp = masks.build_superpixel_map(tiny_rgb, n_segments=120)
    assert sp.dtype == np.uint16 and sp.shape == tiny_rgb.shape[:2] and sp.max() >= 50
    dk = masks.build_darkness_map(tiny_rgb)
    assert dk.dtype == np.uint8 and dk.shape == tiny_rgb.shape[:2]

def test_png_gray_roundtrip():
    a = (np.arange(256, dtype=np.uint8).reshape(16, 16))
    assert (masks.decode_png_gray(masks.encode_png_gray(a)) == a).all()
```

- [ ] **Step 4: Write `backend/app/pipeline/masks.py`**

```python
from __future__ import annotations
import cv2, numpy as np
from skimage.segmentation import slic
from app.shlif import phases
from app.shlif.analyze import verdict_from_masks

def phase_label_map(sulfide: np.ndarray, magnetite: np.ndarray) -> np.ndarray:
    pm = np.zeros(sulfide.shape, np.uint8)          # 0 = matrix
    pm[magnetite.astype(bool)] = phases.MAGNETITE   # 1
    pm[sulfide.astype(bool)] = phases.SULFIDE       # 2 (sulfide wins overlap)
    return pm

def split_phase_map(pm: np.ndarray):
    return pm == phases.SULFIDE, pm == phases.MAGNETITE, pm == phases.MATRIX

def verdict_from_masks_dict(sulfide, magnetite, matrix, talc, cfg, dist_px: int = 12) -> dict:
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px)
    return {"ore_class": v["ore_class"], "text": v["text"], "metrics": v["metrics"]}

def build_superpixel_map(rgb: np.ndarray, n_segments: int = 600) -> np.ndarray:
    seg = slic(rgb, n_segments=n_segments, compactness=12, start_label=0)
    return seg.astype(np.uint16)

def build_darkness_map(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def encode_png_gray(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr.astype(np.uint8))
    if not ok: raise RuntimeError("png encode failed")
    return buf.tobytes()

def decode_png_gray(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None: raise ValueError("png decode failed")
    return arr

def encode_png_u16(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr.astype(np.uint16))
    if not ok: raise RuntimeError("png u16 encode failed")
    return buf.tobytes()
```

- [ ] **Step 5: Write the failing test `backend/tests/test_pipeline.py`**

```python
import numpy as np
from app.pipeline import closeup, loader

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

- [ ] **Step 6: Write `backend/app/pipeline/closeup.py`**

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
    return {"classes": probs, "top": max(probs, key=probs.get)}

def analyze_closeup(rgb: np.ndarray, cfg) -> dict:
    """Classical/CPU path (GPU U-Net wiring is added later behind loader.gpu_available)."""
    res = analyze_image(rgb, cfg, detect_talc_flag=True)  # classical talc seed
    m = res.masks
    phase_map = masks.phase_label_map(m["sulfide"], m["magnetite"])
    return {
        "verdict": {"ore_class": res.ore_class, "text": res.text, "metrics": res.metrics},
        "sort": _sort_card(rgb, cfg),
        "phase_map": phase_map,
        "talc": m["talc"].astype(bool),
        "superpixels": masks.build_superpixel_map(rgb),
        "darkness": masks.build_darkness_map(rgb),
        "text": res.text,
    }
```

- [ ] **Step 7: Run tests**

Run: `cd backend && .venv/bin/pytest tests/test_masks.py tests/test_pipeline.py -v`
Expected: all PASS (classifier absent → `sort` is `None`, allowed by the assertion).

- [ ] **Step 8: Commit**

```bash
cd .. && git add backend && git commit -m "feat(pipeline): closeup analyze + verdict_from_masks refactor + editor maps (superpixel/darkness)"
```

---

## Task 6: Job store (SQLite) + threadpool runner

**Files:**
- Create: `backend/app/schemas/__init__.py`, `backend/app/schemas/jobs.py`, `backend/app/jobs/__init__.py`, `backend/app/jobs/store.py`, `backend/app/jobs/runner.py`, `backend/tests/test_jobs.py`

**Interfaces:**
- Produces:
  - `schemas/jobs.py`: `JobRecord` pydantic (`id, mode, status, progress, message, result`) with `status ∈ {"queued","running","done","error"}`.
  - `store.py`: `JobStore(db_path)` with `create(mode) -> str` (uuid hex id), `get(id) -> JobRecord | None`, `set_status(id, status, progress=None, message=None)`, `set_result(id, dict)`, `log_correction(id, layer, n_pixels)`.
  - `runner.py`: `JobRunner(store, max_workers=2)` with `submit(id, fn)` where `fn() -> dict` runs in a thread; on success `set_result` + `status=done`; on exception `status=error`, `message=str(e)`.

- [ ] **Step 1: Write `backend/app/schemas/jobs.py`**

```python
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel

Status = Literal["queued", "running", "done", "error"]

class JobRecord(BaseModel):
    id: str
    mode: str
    status: Status = "queued"
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[dict[str, Any]] = None
```

- [ ] **Step 2: Write the failing test `backend/tests/test_jobs.py`**

```python
import time
from app.jobs.store import JobStore
from app.jobs.runner import JobRunner

def test_job_lifecycle_success(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")
    assert store.get(jid).status == "queued"
    runner.submit(jid, lambda: {"ore_class": "ordinary"})
    for _ in range(50):
        if store.get(jid).status == "done": break
        time.sleep(0.05)
    rec = store.get(jid)
    assert rec.status == "done" and rec.result == {"ore_class": "ordinary"}

def test_job_lifecycle_error(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")
    def boom(): raise ValueError("nope")
    runner.submit(jid, boom)
    for _ in range(50):
        if store.get(jid).status == "error": break
        time.sleep(0.05)
    rec = store.get(jid)
    assert rec.status == "error" and "nope" in rec.message
```

- [ ] **Step 3: Write `backend/app/jobs/store.py`**

```python
from __future__ import annotations
import json, sqlite3, threading, uuid
from pathlib import Path
from app.schemas.jobs import JobRecord

class JobStore:
    def __init__(self, db_path: Path):
        self._path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, mode TEXT, status TEXT, progress REAL,
                message TEXT, result TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS corrections(
                id TEXT PRIMARY KEY, job_id TEXT, layer TEXT, n_pixels INTEGER, ts TEXT)""")

    def _conn(self):
        return sqlite3.connect(self._path, timeout=30, check_same_thread=False)

    def create(self, mode: str) -> str:
        jid = uuid.uuid4().hex
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                      (jid, mode, "queued", 0.0, None, None))
        return jid

    def get(self, jid: str) -> JobRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT id,mode,status,progress,message,result FROM jobs WHERE id=?",
                            (jid,)).fetchone()
        if not row: return None
        return JobRecord(id=row[0], mode=row[1], status=row[2], progress=row[3],
                         message=row[4], result=json.loads(row[5]) if row[5] else None)

    def set_status(self, jid, status, progress=None, message=None):
        with self._lock, self._conn() as c:
            if progress is None:
                c.execute("UPDATE jobs SET status=?,message=? WHERE id=?", (status, message, jid))
            else:
                c.execute("UPDATE jobs SET status=?,progress=?,message=? WHERE id=?",
                          (status, progress, message, jid))

    def set_result(self, jid, result: dict):
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), jid))

    def log_correction(self, job_id, layer, n_pixels):
        import time as _t
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO corrections VALUES(?,?,?,?,?)",
                      (uuid.uuid4().hex, job_id, layer, int(n_pixels),
                       _t.strftime("%Y-%m-%dT%H:%M:%S")))
```

- [ ] **Step 4: Write `backend/app/jobs/runner.py`**

```python
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from app.jobs.store import JobStore

class JobRunner:
    def __init__(self, store: JobStore, max_workers: int = 2):
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, jid: str, fn: Callable[[], dict]) -> None:
        self._store.set_status(jid, "running", progress=0.05)
        self._pool.submit(self._run, jid, fn)

    def _run(self, jid: str, fn: Callable[[], dict]) -> None:
        try:
            result = fn()
            self._store.set_result(jid, result)
            self._store.set_status(jid, "done", progress=1.0)
        except Exception as e:  # noqa: BLE001 — surfaced to the client as status=error
            self._store.set_status(jid, "error", message=str(e))
```

- [ ] **Step 5: Run tests**

Run: `cd backend && .venv/bin/pytest tests/test_jobs.py -v`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
cd .. && git add backend && git commit -m "feat(backend): SQLite job store + threadpool runner (queued→running→done/error)"
```

---

## Task 7: Analyze / jobs / masks API (closeup over HTTP)

**Files:**
- Create: `backend/app/api/analyze.py`, `backend/app/api/jobs.py`, `backend/app/api/masks.py`, `backend/tests/test_api.py`
- Modify: `backend/main.py` (wire routers + app-scoped store/runner)

**Interfaces:**
- Consumes: `JobStore`, `JobRunner`, `closeup.analyze_closeup`, `masks.*`, `loader.get_config`, `paths.*`.
- Produces the endpoints in the spec §4.1. `POST /api/analyze` returns `{"job_id": str}`. `GET /api/jobs/{id}` returns the `JobRecord`. `result` on a done closeup job = `{"mode":"closeup","verdict":{...},"sort":{...}|null,"text":str,"size":[w,h]}`. Masks/maps served from disk; `POST /api/masks/{id}` returns the recomputed verdict.

- [ ] **Step 1: Write the failing test `backend/tests/test_api.py`**

```python
import io, time, numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from main import app

def _png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()

def _poll(c, jid):
    for _ in range(100):
        r = c.get(f"/api/jobs/{jid}").json()
        if r["status"] in ("done", "error"): return r
        time.sleep(0.1)
    raise AssertionError("job did not finish")

def test_closeup_analyze_and_edit(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", data={"mode": "closeup"},
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    assert up.status_code == 200
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["status"] == "done"
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

- [ ] **Step 2: Write `backend/app/api/analyze.py`**

```python
from __future__ import annotations
import io, numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
from app.pipeline import closeup, panorama, loader, masks
from app.core import paths
from app import runtime  # app-scoped store/runner (set in main.py)

router = APIRouter()
Image.MAX_IMAGE_PIXELS = None

def _persist_maps(jid, r):
    md = paths.masks_dir(jid); mp = paths.maps_dir(jid)
    (md / "phases.png").write_bytes(masks.encode_png_gray(r["phase_map"]))
    (md / "talc.png").write_bytes(masks.encode_png_gray((r["talc"].astype(np.uint8) * 255)))
    (mp / "superpixels.png").write_bytes(masks.encode_png_u16(r["superpixels"]))
    (mp / "darkness.png").write_bytes(masks.encode_png_gray(r["darkness"]))

@router.post("/analyze")
async def analyze(image: UploadFile = File(...), mode: str = Form("closeup")):
    data = await image.read()
    jid = runtime.store.create(mode)
    up = paths.uploads_dir() / f"{jid}_{image.filename or 'up'}"
    up.write_bytes(data)

    def work():
        cfg = loader.get_config()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if mode == "panorama":
            return panorama.analyze_panorama(str(up), cfg, jid)
        im.thumbnail((2400, 2400))
        rgb = np.asarray(im)
        r = closeup.analyze_closeup(rgb, cfg)
        # save display image + editor layers/maps
        disp = paths.images_dir() / f"{jid}.jpg"
        Image.fromarray(rgb).save(disp, "JPEG", quality=90)
        _persist_maps(jid, r)
        h, w = rgb.shape[:2]
        return {"mode": "closeup", "verdict": r["verdict"], "sort": r["sort"],
                "text": r["text"], "size": [w, h]}

    runtime.runner.submit(jid, work)
    return {"job_id": jid}
```

- [ ] **Step 3: Write `backend/app/api/jobs.py`**

```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app import runtime

router = APIRouter()

@router.get("/jobs/{jid}")
def get_job(jid: str):
    rec = runtime.store.get(jid)
    if rec is None:
        raise HTTPException(404, "job not found")
    return rec
```

- [ ] **Step 4: Write `backend/app/api/masks.py`**

```python
from __future__ import annotations
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from app.core import paths
from app.pipeline import masks as M, loader
from app import runtime

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
    if name not in {"superpixels", "darkness"} or not p.exists():
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
    su, mg, mx = M.split_phase_map(pm)
    cfg = loader.get_config()
    v = M.verdict_from_masks_dict(su, mg, mx, tk & mx, cfg)
    runtime.store.log_correction(jid, "phases+talc", int(pm.size))
    return v
```

- [ ] **Step 5: Rewrite `backend/main.py` to wire runtime + routers**

```python
from __future__ import annotations
from fastapi import FastAPI
from app.core import paths

# app-scoped singletons (single granian worker; threadpool inside)
class _Runtime:
    def __init__(self):
        from app.jobs.store import JobStore
        from app.jobs.runner import JobRunner
        self.store = JobStore(paths.db_path())
        self.runner = JobRunner(self.store)

def create_app() -> FastAPI:
    import app as app_pkg
    app_pkg.runtime = _Runtime()          # published for routers: `from app import runtime`
    from app.api import health, analyze, jobs, masks
    api = FastAPI(title="Шлиф-Web API")
    for r in (health.router, analyze.router, jobs.router, masks.router):
        api.include_router(r, prefix="/api")
    return api

app = create_app()
```

Add to `backend/app/__init__.py`:
```python
runtime = None  # set by main.create_app(); routers import this
```

- [ ] **Step 6: Add the panorama stub so imports resolve (real impl in Task 8)**

Create `backend/app/pipeline/panorama.py`:
```python
from __future__ import annotations

def analyze_panorama(path: str, cfg, jid: str) -> dict:
    raise NotImplementedError("panorama wired in Task 8")
```

- [ ] **Step 7: Run tests**

Run: `cd backend && .venv/bin/pytest tests/test_api.py -v`
Expected: `test_closeup_analyze_and_edit PASSED` (marking all pixels talc drives `talc_frac` over `rule.talc_threshold` → `talcose`).

- [ ] **Step 8: Rebuild the api container and smoke through Traefik**

Run:
```bash
cd .. && docker compose up -d --build api traefik
sleep 6 && curl -fsS http://localhost/api/health | grep -q '"status":"ok"' && echo OK
docker compose down
```
Expected: `OK`.

- [ ] **Step 9: Commit**

```bash
git add backend && git commit -m "feat(api): closeup analyze job + jobs poll + mask/map serving + edit→recompute verdict"
```

---

## Task 8: Panorama pipeline wrapper + mode=panorama

**Files:**
- Create: `backend/tests/test_panorama.py`
- Modify: `backend/app/pipeline/panorama.py`

**Interfaces:**
- Consumes: vendored `run_panorama` logic, `loader.load_classifier`, `loader.get_config`, `paths.images_dir`.
- Produces: `panorama.analyze_panorama(path, cfg, jid) -> dict` with `{"mode":"panorama","verdict":{...},"overlay_url":str,"n_ore":int,"n_tiles":int,"talc_frac":float}`; writes the stitched overlay to `images_dir()/{jid}.jpg`. **torch imported lazily** — classifier-only (classical matrix) path must run without torch.

- [ ] **Step 1: Port `run_panorama` into `backend/app/pipeline/panorama.py`**

Copy the `run_panorama` function body from `hakaton_nornikel/scripts/analyze_panorama.py` (lines 42–122) into this module. Change the imports at the top to the **lazy** form — replace the module-level `from shlif.talc_unet import talc_unet_mask` with a lazy import inside the `if talc_unet is not None:` branch:

```python
from __future__ import annotations
import time, cv2, numpy as np
from PIL import Image
from app.shlif import load_config  # noqa: F401 (kept for parity)
from app.shlif.features import extract_features
from app.shlif.imageio import load_rgb
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import detect_talc
from app.shlif.tiling import iter_tiles, tile_grid
from app.pipeline import loader
from app.core import paths

SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)

def _run_panorama(path, clf, feat_names, classes, cfg, min_ore=0.04, display_mp=4_000_000):
    # ---- paste the body of run_panorama here, using talc_unet=None (classical) ----
    # (identical logic; the `if talc_unet is not None` branch, if kept, must do a
    #  local `from app.shlif.talc_unet import talc_unet_mask` import.)
    ...
```

(Paste the exact loop from the source, dropping the `unet`/`talc_unet` parameters — the local build has no `.pt`, so `matrix = segment_phases(pre, cfg.segment).labels == 0` and `talc = detect_talc(pre, matrix, cfg.talc)`.)

Then the public wrapper:

```python
def analyze_panorama(path: str, cfg, jid: str) -> dict:
    cfg.tiling.tile = 2048
    cfg.talc.detect_dark_frac = 0.15
    bundle = loader.load_classifier()
    if bundle is None:
        raise RuntimeError("classifier.pkl required for panorama sort")
    clf, feat, classes = bundle
    r = _run_panorama(path, clf, feat, classes, cfg)
    out = paths.images_dir() / f"{jid}.jpg"
    Image.fromarray(r["overlay"]).save(out, "JPEG", quality=88)
    return {
        "mode": "panorama",
        "verdict": {"ore_class": r["verdict"], "text": "",
                    "metrics": {"talc_frac": r["talc_frac"], "confidence": r["conf"],
                                "sort_proba": r["proba"]}},
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": r["n_ore"], "n_tiles": r["n_tiles"], "talc_frac": r["talc_frac"],
    }
```

- [ ] **Step 2: Write the failing test `backend/tests/test_panorama.py`**

```python
import numpy as np, pytest
from PIL import Image
from app.pipeline import panorama, loader

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
```

- [ ] **Step 3: Run the test**

Run: `cd backend && .venv/bin/pytest tests/test_panorama.py -v`
Expected: PASS if `models/classifier.pkl` is present, else **skipped** (the classifier is copied in Task 14; this test lights up then). Either way the module imports cleanly with torch absent.

- [ ] **Step 4: Commit**

```bash
cd .. && git add backend && git commit -m "feat(pipeline): panorama wrapper (lazy torch) wired into mode=panorama"
```

---

## Task 9: Frontend API client, types, and query hooks

**Files:**
- Create: `frontend/lib/api/types.ts`, `frontend/lib/api/client.ts`, `frontend/lib/api/hooks.ts`, `frontend/lib/mask/encode.ts`, `frontend/tests/encode.test.mjs`, `frontend/tests/client.test.mjs`

**Interfaces:**
- Produces:
  - `types.ts`: `Verdict`, `SortCard`, `AnalyzeResult`, `Job`, `Mode = "closeup"|"panorama"`.
  - `client.ts`: `analyze(file, mode) -> Promise<{job_id}>`, `getJob(id) -> Promise<Job>`, `maskUrl(id, layer)`, `mapUrl(id, name)`, `imageUrl(id)`, `saveMasks(id, phasesBlob, talcBlob) -> Promise<Verdict>`. Base URL `""` (same-origin via Traefik).
  - `hooks.ts`: `useAnalyze()` (mutation), `useJob(id)` (polling query, `refetchInterval` while not done).
  - `encode.ts`: `maskToPngBlob(mask: Uint8Array, w, h) -> Promise<Blob>` (0/255 grayscale PNG via canvas), `pngUrlToImageData(url) -> Promise<ImageData>`.

- [ ] **Step 1: Write `frontend/lib/api/types.ts`**

```typescript
export type Mode = "closeup" | "panorama";
export type OreClass = "ordinary" | "hard" | "talcose" | "review";

export interface Verdict {
  ore_class: OreClass;
  text: string;
  metrics: Record<string, number> & { talc_frac?: number; fine_share?: number; confidence?: number };
}
export interface SortCard { classes: Record<string, number>; top: string; }
export interface AnalyzeResult {
  mode: Mode; verdict: Verdict; sort: SortCard | null; text?: string;
  size?: [number, number]; overlay_url?: string; n_ore?: number; n_tiles?: number;
}
export interface Job {
  id: string; mode: string; status: "queued" | "running" | "done" | "error";
  progress: number; message: string | null; result: AnalyzeResult | null;
}
```

- [ ] **Step 2: Write `frontend/lib/api/client.ts`**

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
export async function getJob(id: string): Promise<Job> {
  const r = await fetch(`${base}/api/jobs/${id}`);
  if (!r.ok) throw new Error(`job failed: ${r.status}`);
  return r.json();
}
export const maskUrl = (id: string, layer: "phases" | "talc") => `${base}/api/masks/${id}/${layer}.png`;
export const mapUrl = (id: string, name: "superpixels" | "darkness") => `${base}/api/maps/${id}/${name}.png`;
export const imageUrl = (id: string) => `${base}/api/images/${id}.jpg`;

export async function saveMasks(id: string, phases: Blob, talc: Blob): Promise<Verdict> {
  const fd = new FormData();
  fd.append("phases", phases, "phases.png");
  fd.append("talc", talc, "talc.png");
  const r = await fetch(`${base}/api/masks/${id}`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`save failed: ${r.status}`);
  return r.json();
}
```

- [ ] **Step 3: Write `frontend/lib/api/hooks.ts`**

```typescript
import { useMutation, useQuery } from "@tanstack/react-query";
import { analyze, getJob } from "./client";
import type { Mode } from "./types";

export function useAnalyze() {
  return useMutation({ mutationFn: (v: { file: File; mode: Mode }) => analyze(v.file, v.mode) });
}
export function useJob(id: string | null) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => getJob(id as string),
    enabled: !!id,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "done" || s === "error" ? false : 800;
    },
  });
}
```

- [ ] **Step 4: Write `frontend/lib/mask/encode.ts`**

```typescript
// Convert a 0/1 mask to a 0/255 grayscale PNG blob using an offscreen canvas.
export async function maskToPngBlob(mask: Uint8Array, w: number, h: number): Promise<Blob> {
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i] ? 255 : 0;
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => cv.toBlob((b) => res(b as Blob), "image/png"));
}
// Pure helper (unit-testable without a DOM): pack a class label map to bytes.
export function labelMapToBytes(map: Uint8Array): Uint8Array {
  return Uint8Array.from(map); // already 0/1/2 per pixel
}
```

- [ ] **Step 5: Write `frontend/tests/encode.test.mjs`**

```javascript
import { test } from "node:test";
import assert from "node:assert";
import { labelMapToBytes } from "../lib/mask/encode.ts";

test("labelMapToBytes preserves class ids", () => {
  const out = labelMapToBytes(Uint8Array.from([0, 1, 2, 0]));
  assert.deepStrictEqual([...out], [0, 1, 2, 0]);
});
```

- [ ] **Step 6: Write `frontend/tests/client.test.mjs`**

```javascript
import { test } from "node:test";
import assert from "node:assert";
import { maskUrl, mapUrl, imageUrl } from "../lib/api/client.ts";

test("url builders", () => {
  assert.strictEqual(maskUrl("abc", "phases"), "/api/masks/abc/phases.png");
  assert.strictEqual(mapUrl("abc", "darkness"), "/api/maps/abc/darkness.png");
  assert.strictEqual(imageUrl("abc"), "/api/images/abc.jpg");
});
```

- [ ] **Step 7: Run the tests (Node 22 strips TS types natively)**

Run: `cd frontend && node --test --experimental-strip-types tests/`
Expected: 2 files, all tests pass. (If the Node version rejects `.ts` imports, add `"type":"module"` and run via `tsx`; Node ≥22.7 strips types with the flag above.)

- [ ] **Step 8: Commit**

```bash
cd .. && git add frontend && git commit -m "feat(frontend): typed API client + TanStack Query hooks + mask PNG encode util + tests"
```

---

## Task 10: Analyze screen — upload → poll → verdict panel («Шлиф» tokens)

**Files:**
- Create: `frontend/app/globals.css`, `frontend/app/providers.tsx`, `frontend/components/verdict/SortCard.tsx`, `frontend/components/verdict/PhaseBars.tsx`, `frontend/components/verdict/VerdictPanel.tsx`, `frontend/postcss.config.mjs`
- Modify: `frontend/app/layout.tsx` (import globals.css + wrap in providers), `frontend/app/page.tsx` (the screen)

**Interfaces:**
- Consumes: `useAnalyze`, `useJob`, `imageUrl`, types.
- Produces: `VerdictPanel({result})` rendering the sort card + phase bars + «на проверку» badge; `page.tsx` — a client component: file input + mode toggle → `useAnalyze` → poll `useJob` → show image + `VerdictPanel`.

- [ ] **Step 1: Write `frontend/app/globals.css` — port the «Шлиф» tokens**

Port the CSS custom properties + component classes from `hakaton_nornikel/app.py` (the `<style>` block, lines 37–94): `:root` OKLCH tokens, `.oreclass`, `.verdict`, `.mrow/.mbar`, `.note`, IBM Plex `@import`. Paste them verbatim into `globals.css` (they are plain CSS). Add:
```css
body { background: var(--bg); color: var(--text); font-family: var(--sans); margin: 0; }
.stage { background: var(--stage); border: 1px solid oklch(38% .014 258); border-radius: 12px; padding: 8px; }
.stage img { max-width: 100%; display: block; }
.grid2 { display: grid; grid-template-columns: 1.7fr 1fr; gap: 24px; align-items: start; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
```

- [ ] **Step 2: Write `frontend/app/providers.tsx`**

```tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
export default function Providers({ children }: { children: React.ReactNode }) {
  const [qc] = useState(() => new QueryClient());
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}
```

- [ ] **Step 3: Update `frontend/app/layout.tsx`**

```tsx
import "./globals.css";
import Providers from "./providers";
export const metadata = { title: "Шлиф · классификация руд" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="ru"><body><Providers>{children}</Providers></body></html>);
}
```

- [ ] **Step 4: Write `frontend/components/verdict/SortCard.tsx`**

```tsx
import type { SortCard as Sort } from "@/lib/api/types";
const RU: Record<string, string> = { ordinary: "рядовая руда", hard: "труднообогатимая руда", talcose: "оталькованная руда" };
const BAR: Record<string, string> = { ordinary: "rgb(80,190,120)", hard: "rgb(225,85,80)", talcose: "rgb(95,140,235)" };
export function SortCard({ sort }: { sort: Sort | null }) {
  if (!sort) return <div className="note">Классификатор сорта недоступен (нет models/classifier.pkl).</div>;
  const top = sort.top;
  return (
    <div className="verdict" style={{ marginBottom: 14 }}>
      <div className="vh"><div className="eye">Сорт руды · классификатор (RF · F1 0.84)</div>
        <div style={{ marginTop: 8 }}><span className={`oreclass ${top}`}>{RU[top]}</span></div></div>
      <div className="vb">
        {Object.entries(sort.classes).map(([k, v]) => (
          <div className="mrow" key={k}>
            <div className="top"><span>{RU[k]}</span><span className="pct">{Math.round(v * 100)}%</span></div>
            <div className="mbar"><i style={{ width: `${Math.min(v * 100, 100)}%`, background: BAR[k] }} /></div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Write `frontend/components/verdict/PhaseBars.tsx`**

```tsx
import type { Verdict } from "@/lib/api/types";
const ROWS: [string, string, string][] = [
  ["Доля талька", "talc_frac", "var(--phase-talc-ink)"],
  ["Тонкие срастания", "fine_share", "var(--phase-fine-ink)"],
  ["Обычные срастания", "normal_share", "var(--phase-normal-ink)"],
  ["Доля сульфидов", "sulfide_frac", "var(--text)"],
];
export function PhaseBars({ verdict }: { verdict: Verdict }) {
  const m = verdict.metrics;
  return (
    <div className="verdict">
      <div className="vh"><div className="eye">Фазовый состав · правило</div>
        <div style={{ marginTop: 8 }}><span className={`oreclass ${verdict.ore_class}`}>{verdict.text ? "" : ""}{oreRu(verdict.ore_class)}</span></div></div>
      <div className="vb">
        {ROWS.map(([label, key, col]) => (
          <div className="kv" key={key}><span className="k">{label}</span>
            <span className="v" style={{ color: col }}>{((m[key] ?? 0) * 100).toFixed(1)}%</span></div>
        ))}
      </div>
      <div className="vf"><span>уверенность {(m.confidence ?? 0).toFixed(2)}</span><span>seg+rule</span></div>
    </div>
  );
}
function oreRu(c: string) {
  return { ordinary: "рядовая руда", hard: "труднообогатимая руда", talcose: "оталькованная руда", review: "на проверку" }[c] ?? c;
}
```

- [ ] **Step 6: Write `frontend/components/verdict/VerdictPanel.tsx`**

```tsx
import type { AnalyzeResult } from "@/lib/api/types";
import { SortCard } from "./SortCard";
import { PhaseBars } from "./PhaseBars";
export function VerdictPanel({ result }: { result: AnalyzeResult }) {
  return (<div><SortCard sort={result.sort} /><PhaseBars verdict={result.verdict} />
    {result.text ? <div className="note" style={{ marginTop: 12 }}>{result.text}</div> : null}</div>);
}
```

- [ ] **Step 7: Write `frontend/postcss.config.mjs`** (no Tailwind — plain CSS)

```javascript
export default { plugins: {} };
```

- [ ] **Step 8: Rewrite `frontend/app/page.tsx`**

```tsx
"use client";
import { useState } from "react";
import { useAnalyze, useJob } from "@/lib/api/hooks";
import { imageUrl } from "@/lib/api/client";
import type { Mode } from "@/lib/api/types";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";

export default function Home() {
  const [mode, setMode] = useState<Mode>("closeup");
  const [jobId, setJobId] = useState<string | null>(null);
  const analyze = useAnalyze();
  const job = useJob(jobId);

  function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) analyze.mutate({ file: f, mode }, { onSuccess: (r) => setJobId(r.job_id) });
  }
  const result = job.data?.status === "done" ? job.data.result : null;
  const busy = analyze.isPending || (job.data && ["queued", "running"].includes(job.data.status));

  return (
    <main style={{ maxWidth: 1280, margin: "0 auto", padding: "22px 20px" }}>
      <div className="topbar"><div className="logo">◈</div>
        <div><div className="crumb">Шлиф · классификация руд</div><h1 style={{ margin: 0 }}>Скажи мне кто твой шлиф</h1></div></div>
      <div style={{ display: "flex", gap: 12, margin: "14px 0" }}>
        <label><input type="radio" checked={mode === "closeup"} onChange={() => setMode("closeup")} /> Крупный план</label>
        <label><input type="radio" checked={mode === "panorama"} onChange={() => setMode("panorama")} /> Панорама</label>
        <input type="file" accept="image/*" onChange={onFile} />
      </div>
      <div className="grid2">
        <div className="stage">
          {jobId && result ? <img src={result.overlay_url ?? imageUrl(jobId)} alt="шлиф" /> :
            <div style={{ padding: 40, color: "var(--muted)" }}>{busy ? "Анализ…" : "Загрузите снимок шлифа"}</div>}
        </div>
        <div>{result ? <VerdictPanel result={result} /> :
          job.data?.status === "error" ? <div className="note">Ошибка: {job.data.message}</div> : null}</div>
      </div>
    </main>
  );
}
```

- [ ] **Step 9: Build to verify**

Run: `cd frontend && npm run build`
Expected: `✓ Compiled successfully`; `/` route listed.

- [ ] **Step 10: Commit**

```bash
cd .. && git add frontend && git commit -m "feat(frontend): analyze screen — upload/mode/poll + verdict panel in «Шлиф» tokens"
```

---

## Task 11: Corrector state reducer (pure, unit-tested)

**Files:**
- Create: `frontend/components/corrector/reducer.ts`, `frontend/tests/reducer.test.mjs`

**Interfaces:**
- Produces: `reducer.ts` — pure functions the canvas component drives:
  - Types: `Tool = "superpixel"|"brush"|"eraser"|"threshold"|"autofill"`, `Layer = "matrix"|"magnetite"|"sulfide"|"talc"`, `CorrectorState { phaseMap: Uint8Array; talc: Uint8Array; w:number; h:number; tool:Tool; layer:Layer; brush:number; undo: Snapshot[]; redo: Snapshot[] }`.
  - `applyPhase(state, idxs: number[], layer): CorrectorState` — assign a phase (0/1/2) to pixel indices in `phaseMap` (talc layer untouched); pushes an undo snapshot.
  - `applyTalc(state, idxs, value: boolean): CorrectorState`.
  - `undo(state)`, `redo(state)`.
  - `layerToClass(layer): 0|1|2` mapping matrix→0/magnetite→1/sulfide→2.

- [ ] **Step 1: Write the failing test `frontend/tests/reducer.test.mjs`**

```javascript
import { test } from "node:test";
import assert from "node:assert";
import { initState, applyPhase, applyTalc, undo, redo, layerToClass } from "../components/corrector/reducer.ts";

const st0 = () => initState(new Uint8Array(4), new Uint8Array(4), 2, 2);

test("layerToClass maps phases", () => {
  assert.strictEqual(layerToClass("matrix"), 0);
  assert.strictEqual(layerToClass("magnetite"), 1);
  assert.strictEqual(layerToClass("sulfide"), 2);
});

test("applyPhase sets class ids and is undoable", () => {
  let s = st0();
  s = applyPhase(s, [0, 1], "sulfide");
  assert.deepStrictEqual([...s.phaseMap], [2, 2, 0, 0]);
  s = undo(s);
  assert.deepStrictEqual([...s.phaseMap], [0, 0, 0, 0]);
  s = redo(s);
  assert.deepStrictEqual([...s.phaseMap], [2, 2, 0, 0]);
});

test("applyTalc toggles the talc overlay independently", () => {
  let s = st0();
  s = applyTalc(s, [3], true);
  assert.deepStrictEqual([...s.talc], [0, 0, 0, 1]);
  assert.deepStrictEqual([...s.phaseMap], [0, 0, 0, 0]);
});
```

- [ ] **Step 2: Write `frontend/components/corrector/reducer.ts`**

```typescript
export type Tool = "superpixel" | "brush" | "eraser" | "threshold" | "autofill";
export type Layer = "matrix" | "magnetite" | "sulfide" | "talc";
interface Snapshot { phaseMap: Uint8Array; talc: Uint8Array; }
export interface CorrectorState {
  phaseMap: Uint8Array; talc: Uint8Array; w: number; h: number;
  tool: Tool; layer: Layer; brush: number; undoStack: Snapshot[]; redoStack: Snapshot[];
}
export function layerToClass(l: Layer): 0 | 1 | 2 {
  return l === "sulfide" ? 2 : l === "magnetite" ? 1 : 0;
}
export function initState(phaseMap: Uint8Array, talc: Uint8Array, w: number, h: number): CorrectorState {
  return { phaseMap, talc, w, h, tool: "brush", layer: "sulfide", brush: 12, undoStack: [], redoStack: [] };
}
function snap(s: CorrectorState): Snapshot {
  return { phaseMap: Uint8Array.from(s.phaseMap), talc: Uint8Array.from(s.talc) };
}
export function applyPhase(s: CorrectorState, idxs: number[], layer: Layer): CorrectorState {
  const cls = layerToClass(layer);
  const phaseMap = Uint8Array.from(s.phaseMap);
  for (const i of idxs) phaseMap[i] = cls;
  return { ...s, phaseMap, undoStack: [...s.undoStack, snap(s)], redoStack: [] };
}
export function applyTalc(s: CorrectorState, idxs: number[], value: boolean): CorrectorState {
  const talc = Uint8Array.from(s.talc);
  for (const i of idxs) talc[i] = value ? 1 : 0;
  return { ...s, talc, undoStack: [...s.undoStack, snap(s)], redoStack: [] };
}
export function undo(s: CorrectorState): CorrectorState {
  if (!s.undoStack.length) return s;
  const prev = s.undoStack[s.undoStack.length - 1];
  return { ...s, phaseMap: Uint8Array.from(prev.phaseMap), talc: Uint8Array.from(prev.talc),
    undoStack: s.undoStack.slice(0, -1), redoStack: [...s.redoStack, snap(s)] };
}
export function redo(s: CorrectorState): CorrectorState {
  if (!s.redoStack.length) return s;
  const next = s.redoStack[s.redoStack.length - 1];
  return { ...s, phaseMap: Uint8Array.from(next.phaseMap), talc: Uint8Array.from(next.talc),
    redoStack: s.redoStack.slice(0, -1), undoStack: [...s.undoStack, snap(s)] };
}
```

- [ ] **Step 3: Run the test**

Run: `cd frontend && node --test --experimental-strip-types tests/reducer.test.mjs`
Expected: all 3 tests pass.

- [ ] **Step 4: Commit**

```bash
cd .. && git add frontend && git commit -m "feat(corrector): pure reducer — phase label map + talc overlay + undo/redo"
```

---

## Task 12: Corrector canvas component + tools + Save→recompute

**Files:**
- Create: `frontend/components/corrector/Corrector.tsx`, `frontend/lib/mask/superpixel.ts`
- Modify: `frontend/app/page.tsx` (add «Доработать» toggle → mount `Corrector`)

**Interfaces:**
- Consumes: reducer (`initState/applyPhase/applyTalc/undo/redo`), `maskUrl/mapUrl/imageUrl/saveMasks`, `maskToPngBlob`, superpixel decode.
- Produces: `Corrector({ jobId, size, onVerdict })` — loads the source image + phase/talc masks + superpixel/darkness maps, renders an editable canvas with tool controls (Суперпиксель / Кисть / Ластик / Тёмные области / Авто), and a Save button that POSTs both masks and calls `onVerdict(verdict)`. `superpixel.ts`: `loadSuperpixels(url,w,h) -> Promise<Uint16Array>` (decode the 16-bit PNG label map), `cellIndices(labels, seedIdx) -> number[]`.

- [ ] **Step 1: Write `frontend/lib/mask/superpixel.ts`**

```typescript
// Decode the SLIC label map PNG into a Uint16Array of per-pixel segment ids.
export async function loadSuperpixels(url: string, w: number, h: number): Promise<Uint16Array> {
  const img = await createImageBitmap(await (await fetch(url)).blob());
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint16Array(w * h);
  // OpenCV wrote a single-channel 16-bit PNG; the browser expands it to RGBA8.
  // Reconstruct the id from the red+green byte pair is unreliable across encoders,
  // so we instead treat equal-colour neighbourhoods as one cell: pack R,G,B.
  for (let i = 0; i < w * h; i++) out[i] = (data[i * 4] << 8) | data[i * 4 + 1];
  return out;
}
export function cellIndices(labels: Uint16Array, seedIdx: number): number[] {
  const id = labels[seedIdx];
  const out: number[] = [];
  for (let i = 0; i < labels.length; i++) if (labels[i] === id) out.push(i);
  return out;
}
```

> Note for the implementer: 16-bit single-channel PNG decoding in-browser is the risk flagged in the spec (§10). If `loadSuperpixels` yields wrong cells, switch the backend `/maps/{id}/superpixels.png` to an **8-bit RGB** encoding where the segment id is packed as `(id>>8, id&255, 0)` and decode `R<<8|G` here — that pairing is lossless through canvas. Verify with the manual smoke in Step 6 before moving on.

- [ ] **Step 2: Write `frontend/components/corrector/Corrector.tsx`**

```tsx
"use client";
import { useEffect, useRef, useState } from "react";
import { initState, applyPhase, applyTalc, undo, redo, type CorrectorState, type Tool, type Layer } from "./reducer";
import { imageUrl, maskUrl, mapUrl, saveMasks } from "@/lib/api/client";
import { maskToPngBlob } from "@/lib/mask/encode";
import { loadSuperpixels, cellIndices } from "@/lib/mask/superpixel";
import type { Verdict } from "@/lib/api/types";

const PHASE_RGB: Record<number, [number, number, number]> = { 1: [150, 160, 182], 2: [201, 180, 95] };
const TALC_RGB: [number, number, number] = [79, 143, 240];
const TOOLS: [Tool, string][] = [["superpixel", "Суперпиксель"], ["brush", "Кисть"], ["eraser", "Ластик"], ["threshold", "Тёмные области"], ["autofill", "Авто-заполнение"]];
const LAYERS: [Layer, string][] = [["sulfide", "сульфид"], ["magnetite", "магнетит"], ["matrix", "матрица"], ["talc", "тальк"]];

async function pngToArray(url: string, w: number, h: number): Promise<Uint8Array> {
  const img = await createImageBitmap(await (await fetch(url)).blob());
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!; ctx.drawImage(img, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = d[i * 4]; // grayscale in R
  return out;
}

export function Corrector({ jobId, size, onVerdict }: { jobId: string; size: [number, number]; onVerdict: (v: Verdict) => void }) {
  const [w, h] = size;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseRef = useRef<ImageBitmap | null>(null);
  const spRef = useRef<Uint16Array | null>(null);
  const darkRef = useRef<Uint8Array | null>(null);
  const [state, setState] = useState<CorrectorState | null>(null);
  const [thr, setThr] = useState(60);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      baseRef.current = await createImageBitmap(await (await fetch(imageUrl(jobId))).blob());
      const phasesGray = await pngToArray(maskUrl(jobId, "phases"), w, h); // 0/1/2 already
      const talc = await pngToArray(maskUrl(jobId, "talc"), w, h);
      spRef.current = await loadSuperpixels(mapUrl(jobId, "superpixels"), w, h);
      darkRef.current = await pngToArray(mapUrl(jobId, "darkness"), w, h);
      setState(initState(Uint8Array.from(phasesGray), Uint8Array.from(talc.map((v) => (v > 127 ? 1 : 0))), w, h));
    })();
  }, [jobId, w, h]);

  useEffect(() => { if (state) draw(); });

  function draw() {
    const cv = canvasRef.current, s = state;
    if (!cv || !s || !baseRef.current) return;
    const ctx = cv.getContext("2d")!;
    ctx.drawImage(baseRef.current, 0, 0, w, h);
    const overlay = ctx.getImageData(0, 0, w, h);
    for (let i = 0; i < w * h; i++) {
      const cls = s.phaseMap[i];
      const rgb = cls ? PHASE_RGB[cls] : null;
      if (rgb) { overlay.data[i * 4] = 0.45 * overlay.data[i * 4] + 0.55 * rgb[0]; overlay.data[i * 4 + 1] = 0.45 * overlay.data[i * 4 + 1] + 0.55 * rgb[1]; overlay.data[i * 4 + 2] = 0.45 * overlay.data[i * 4 + 2] + 0.55 * rgb[2]; }
      if (s.talc[i]) { overlay.data[i * 4] = 0.4 * overlay.data[i * 4] + 0.6 * TALC_RGB[0]; overlay.data[i * 4 + 1] = 0.4 * overlay.data[i * 4 + 1] + 0.6 * TALC_RGB[1]; overlay.data[i * 4 + 2] = 0.4 * overlay.data[i * 4 + 2] + 0.6 * TALC_RGB[2]; }
    }
    ctx.putImageData(overlay, 0, 0);
  }

  function paintAt(cx: number, cy: number) {
    if (!state) return;
    const idx = cy * w + cx;
    const isTalc = state.layer === "talc";
    if (state.tool === "superpixel" && spRef.current) {
      const idxs = cellIndices(spRef.current, idx);
      setState(isTalc ? applyTalc(state, idxs, true) : applyPhase(state, idxs, state.layer));
    } else if (state.tool === "brush" || state.tool === "eraser") {
      const r = state.brush, idxs: number[] = [];
      for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x >= 0 && x < w && y >= 0 && y < h && dx * dx + dy * dy <= r * r) idxs.push(y * w + x);
      }
      const erase = state.tool === "eraser";
      setState(isTalc ? applyTalc(state, idxs, !erase) : applyPhase(state, idxs, erase ? "matrix" : state.layer));
    } else if (state.tool === "threshold" && darkRef.current) {
      const idxs: number[] = [];
      for (let i = 0; i < w * h; i++) if (darkRef.current[i] <= thr && state.phaseMap[i] === 0) idxs.push(i);
      setState(applyTalc(state, idxs, true)); // dark-area → talc within matrix
    } else if (state.tool === "autofill") {
      // seed: re-load pipeline talc as-is (already in state); no-op placeholder for future detectors
    }
  }

  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const cx = Math.floor(((e.clientX - rect.left) / rect.width) * w);
    const cy = Math.floor(((e.clientY - rect.top) / rect.height) * h);
    paintAt(cx, cy);
  }

  async function save() {
    if (!state) return;
    setSaving(true);
    try {
      const phaseBlob = await maskToPngBlob(state.phaseMap, w, h); // 0/1/2 stored as-is (grayscale)
      const talcBlob = await maskToPngBlob(state.talc, w, h);
      const v = await saveMasks(jobId, phaseBlob, talcBlob);
      onVerdict(v);
    } finally { setSaving(false); }
  }

  if (!state) return <div className="stage" style={{ padding: 40 }}>Загрузка редактора…</div>;
  return (
    <div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {TOOLS.map(([t, ru]) => <button key={t} onClick={() => setState({ ...state, tool: t })}
          style={{ fontWeight: state.tool === t ? 700 : 400 }}>{ru}</button>)}
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        {LAYERS.map(([l, ru]) => <button key={l} onClick={() => setState({ ...state, layer: l })}
          style={{ fontWeight: state.layer === l ? 700 : 400 }}>{ru}</button>)}
        <label>кисть <input type="range" min={2} max={40} value={state.brush} onChange={(e) => setState({ ...state, brush: +e.target.value })} /></label>
        {state.tool === "threshold" && <label>порог <input type="range" min={5} max={200} value={thr} onChange={(e) => setThr(+e.target.value)} /></label>}
        <button onClick={() => setState(undo(state))}>↶</button>
        <button onClick={() => setState(redo(state))}>↷</button>
      </div>
      <div className="stage"><canvas ref={canvasRef} width={w} height={h} onClick={onClick} onMouseDown={onClick} style={{ width: "100%", cursor: "crosshair" }} /></div>
      <button onClick={save} disabled={saving} style={{ marginTop: 8 }}>{saving ? "Сохранение…" : "💾 Сохранить и пересчитать вердикт"}</button>
    </div>
  );
}
```

> **Note on the phase mask encoding:** the backend stores `phases.png` as a grayscale image whose pixel values are the class ids `0/1/2` (see Task 5 `_persist_maps`), and `POST /api/masks/{id}` reads them back the same way (Task 7 `save_masks` reads `phases` as a `uint8` label map, not a `>127` threshold). `maskToPngBlob` here must write the raw `0/1/2` values, NOT `0/255` — pass the phaseMap through a variant that writes `v` directly. Add `rawMaskToPngBlob(map,w,h)` to `lib/mask/encode.ts` (same as `maskToPngBlob` but `const v = map[i]`) and use it for the phase blob; keep `maskToPngBlob` (0/255) for talc.

- [ ] **Step 3: Add `rawMaskToPngBlob` to `frontend/lib/mask/encode.ts`**

```typescript
export async function rawMaskToPngBlob(map: Uint8Array, w: number, h: number): Promise<Blob> {
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d")!; const img = ctx.createImageData(w, h);
  for (let i = 0; i < map.length; i++) { const v = map[i]; img.data[i*4]=v; img.data[i*4+1]=v; img.data[i*4+2]=v; img.data[i*4+3]=255; }
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => cv.toBlob((b) => res(b as Blob), "image/png"));
}
```
Then in `Corrector.tsx` `save()`, use `rawMaskToPngBlob(state.phaseMap, w, h)` for `phaseBlob`.

- [ ] **Step 4: Wire «Доработать» into `frontend/app/page.tsx`**

Add state `const [editing, setEditing] = useState(false)` and `const [verdictOverride, setVerdictOverride] = useState<Verdict|null>(null)`. When `result` exists and `mode==="closeup"`, render a **«Доработать»** button under the verdict panel that sets `editing=true`; when editing, replace the `.stage` image with `<Corrector jobId={jobId!} size={result.size!} onVerdict={(v)=>{setVerdictOverride(v); }} />`. If `verdictOverride` is set, pass a merged result (`{...result, verdict: verdictOverride}`) to `VerdictPanel`. Exact diff:

```tsx
// imports
import { Corrector } from "@/components/corrector/Corrector";
import type { Verdict } from "@/lib/api/types";
// state (inside Home)
const [editing, setEditing] = useState(false);
const [vOverride, setVOverride] = useState<Verdict | null>(null);
// shown result
const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
// left stage:
{editing && result?.size ? (
  <Corrector jobId={jobId!} size={result.size} onVerdict={(v) => setVOverride(v)} />
) : jobId && shown ? <img src={shown.overlay_url ?? imageUrl(jobId)} alt="шлиф" /> :
  <div style={{ padding: 40, color: "var(--muted)" }}>{busy ? "Анализ…" : "Загрузите снимок шлифа"}</div>}
// right panel, under VerdictPanel:
{shown && mode === "closeup" && !editing ? <button onClick={() => setEditing(true)} style={{ marginTop: 12 }}>✎ Доработать маски</button> : null}
```

- [ ] **Step 5: Build**

Run: `cd frontend && npm run build`
Expected: `✓ Compiled successfully`.

- [ ] **Step 6: Manual smoke (superpixel decode risk)**

Run the full stack and verify the corrector visually:
```bash
cd .. && cp ../hakaton_nornikel/out/classifier.pkl backend/models/ 2>/dev/null || true
docker compose up -d --build && sleep 8
echo "open http://localhost , upload a close-up, click Доработать, test Кисть + Суперпиксель + Тёмные области, Save"
```
Expected: superpixel clicks select a coherent region (if not, apply the RGB-packing fallback from Step 1's note). Then `docker compose down`.

- [ ] **Step 7: Commit**

```bash
git add frontend && git commit -m "feat(corrector): canvas multi-layer editor (superpixel/brush/eraser/threshold) + Save→recompute verdict"
```

---

## Task 13: Panorama result view (coarse, read-only)

**Files:**
- Modify: `frontend/app/page.tsx`, `frontend/components/verdict/VerdictPanel.tsx`

**Interfaces:**
- Consumes: `AnalyzeResult` with `overlay_url`, `n_ore`, `n_tiles`, `verdict.metrics.talc_frac`.
- Produces: panorama results render the stitched overlay + a section-verdict card («на проверку»); the «Доработать» button is hidden in panorama mode (v1 = read-only, per spec §11).

- [ ] **Step 1: Extend `VerdictPanel` to show panorama stats when `result.mode==="panorama"`**

```tsx
// inside VerdictPanel, before the closeup PhaseBars, add:
{result.mode === "panorama" ? (
  <div className="verdict" style={{ marginBottom: 14 }}>
    <div className="vh"><div className="eye">Секционный вердикт · НА ПРОВЕРКУ</div>
      <div style={{ marginTop: 8 }}><span className={`oreclass ${["ordinary","hard","talcose"].includes(result.verdict.ore_class) ? result.verdict.ore_class : "review"}`}>
        {oreRu(result.verdict.ore_class)}</span></div></div>
    <div className="vb">
      <div className="kv"><span className="k">Тальк-кандидаты</span><span className="v">{((result.verdict.metrics.talc_frac ?? 0) * 100).toFixed(1)}%</span></div>
      <div className="kv"><span className="k">Рудных тайлов</span><span className="v">{result.n_ore} / {result.n_tiles}</span></div>
    </div>
  </div>
) : <><SortCard sort={result.sort} /><PhaseBars verdict={result.verdict} /></>}
```
Add the `oreRu` helper (same mapping as PhaseBars) to this file, and only render the closeup branch when `mode !== "panorama"`.

- [ ] **Step 2: Hide «Доработать» in panorama mode** — already guarded by `mode === "closeup"` in Task 12 Step 4. Verify the guard is present.

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: `✓ Compiled successfully`.

- [ ] **Step 4: Commit**

```bash
cd .. && git add frontend && git commit -m "feat(frontend): panorama result view — overlay + section verdict (read-only v1)"
```

---

## Task 14: Model wiring, README, end-to-end verify

**Files:**
- Create: `README.md` (expand the stub), `backend/models/.gitkeep`, `backend/data/.gitkeep`
- Modify: none (copy `classifier.pkl` into `backend/models/` — gitignored)

**Interfaces:**
- Consumes: everything. Produces the run instructions + a verified end-to-end demo on the classical path.

- [ ] **Step 1: Provide the classifier so the sort card + panorama light up**

Run:
```bash
mkdir -p backend/models backend/data && touch backend/models/.gitkeep backend/data/.gitkeep
cp ../hakaton_nornikel/out/classifier.pkl backend/models/classifier.pkl && echo "classifier in place"
```
(If `out/classifier.pkl` is absent locally, note it in the README — the app still runs, the sort card shows the “недоступен” note, panorama analyze returns an error surfaced in the UI.)

- [ ] **Step 2: Backend regression — run the full suite (panorama test now un-skips)**

Run: `cd backend && .venv/bin/pytest -q`
Expected: all tests pass; `test_panorama_runs` now runs (classifier present).

- [ ] **Step 3: Frontend tests**

Run: `cd frontend && node --test --experimental-strip-types tests/`
Expected: all pass.

- [ ] **Step 4: End-to-end via the whole stack**

Run:
```bash
cd .. && docker compose up -d --build && sleep 10
JID=$(curl -fsS -F "mode=closeup" -F "image=@../hakaton_nornikel/out/panorama_result.png" http://localhost/api/analyze | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
for i in $(seq 1 60); do S=$(curl -fsS http://localhost/api/jobs/$JID | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])"); [ "$S" = done ] && break; sleep 1; done
echo "final status: $S"
curl -fsS -o /dev/null -w "phases.png %{http_code}\n" http://localhost/api/masks/$JID/phases.png
docker compose down
```
Expected: `final status: done` and `phases.png 200`.

- [ ] **Step 5: Write the full `README.md`** (replace the stub) — cover: what it is, `docker compose up`, local dev (`backend`: `uv` + `granian --reload main:app`; `frontend`: `npm run dev`), where models go (`backend/models/`), the CPU/GPU note, and a link to the spec + this plan.

- [ ] **Step 6: Commit**

```bash
git add README.md backend/models/.gitkeep backend/data/.gitkeep && git commit -m "docs: README run guide + wire classifier.pkl; end-to-end verified on classical path"
```

- [ ] **Step 7: Push**

```bash
git push origin master
```

---

## Self-Review

**Spec coverage:**
- §3 monorepo + vendoring → Tasks 1–4. ✓
- §4.1 endpoints (analyze/jobs/masks/maps/images/health) → Tasks 1,4,7. ✓
- §4.2 SQLite job store + threadpool → Task 6. ✓
- §4.3 closeup + panorama + masks + recompute → Tasks 5,8. ✓ (`verdict_from_masks` refactor keeps it DRY)
- §4.4 data layout → Tasks 1 (paths), 7 (persist). ✓
- §5 frontend page + verdict panel + design tokens → Tasks 9,10. ✓
- §5.1 corrector: phase label map + talc overlay, superpixel/brush/eraser/threshold/autofill, undo/redo, save→recompute; sort card unchanged on edit (backend `save_masks` returns only phase-composition verdict) → Tasks 11,12. ✓
- §6 data flow → Tasks 7,10,12. ✓
- §7 Traefik + compose + override + GPU gating → Task 3. ✓
- §8 error handling: job error→status, model absent→note, OOM guard (`thumbnail`, panorama `load_rgb(max_pixels)`), idempotent overwrite → Tasks 6,7,8,10. ✓
- §9 testing (backend pytest + frontend node --test + e2e verify) → every task + Task 14. ✓
- §11 milestones map 1:1 onto Tasks 3→14; panorama correction coarse/read-only → Task 13. ✓

**Placeholder scan:** the only intentional deferral is Task 7 Step 6's `panorama.py` stub, replaced with the real impl in Task 8 Step 1 — flagged explicitly, not a silent TODO. Task 12 `autofill` is a documented no-op seed (talc already pre-filled from the pipeline); not load-bearing. No `TBD`/`add error handling`/`similar to`.

**Type consistency:** `verdict_from_masks` (analyze.py) returns `{ore_class,text,metrics,normal,fine}`; `verdict_from_masks_dict` (masks.py) narrows to `{ore_class,text,metrics}` — the API returns that dict and the frontend `Verdict` type matches (`ore_class`,`text`,`metrics`). Phase ids `0/1/2` are consistent backend (`phases.MATRIX/MAGNETITE/SULFIDE`) ↔ frontend (`layerToClass`). `phases.png` is a raw-label grayscale on both write (`_persist_maps`), read-back (`save_masks`), and client encode (`rawMaskToPngBlob`) — the Task 12 Step 2/3 note fixes the one place a 0/255 blob would have corrupted it. Job `result` shape (`mode/verdict/sort/size/overlay_url/n_ore/n_tiles`) matches `AnalyzeResult`.
