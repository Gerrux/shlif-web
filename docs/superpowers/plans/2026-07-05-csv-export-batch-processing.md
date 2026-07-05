# CSV-экспорт и пакетная обработка — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CSV export (single job + batch summary) and web-UI batch image processing to the
«Шлиф-Web» service, per `docs/superpowers/specs/2026-07-05-csv-export-batch-processing-design.md`.

**Architecture:** Backend gains a `batch_id`/`filename`/`created_at` on each job row (additive
SQLite migration), a `GET /api/jobs?batch_id=` list endpoint, a small `params` snapshot on each
job's stored result for reproducibility, and a configurable job-worker pool so batch uploads
process concurrently in the background. Frontend gains multi-file upload, a batch gallery view
(reusing the existing single-job workspace unchanged for each item), and client-side CSV generation
from data the API already returns (no new backend CSV endpoint).

**Tech Stack:** FastAPI + SQLite (backend), Next.js/React + TanStack Query (frontend), pytest,
`node --test` (via `tsx`).

## Global Constraints

- Batch processing is triggered **only** from the web UI (multi-file drag/select) — no CLI/headless
  batch mode in this plan.
- CSV is generated **client-side** from the JSON already returned by the API — no new backend CSV
  endpoint.
- Batch persistence (surviving a page reload) is via a `batch=<id>` **URL query parameter** +
  `GET /api/jobs?batch_id=` (not `localStorage` — this repo already has an equivalent convention
  for the single-job case, `frontend/lib/jobUrl.ts`, discovered while reading the current code; this
  plan follows that existing idiom instead of introducing a second, inconsistent persistence
  mechanism, while still meeting the spec's "survives reload" requirement).
- Mode (`closeup`/`panorama`) is auto-detected server-side per image (`app/pipeline/detect.py`) —
  there is no manual mode toggle to preserve or extend for batches.
- Single-file upload behavior must not change at all: dropping exactly one file keeps today's exact
  flow (no batch_id, no gallery).
- CSV metric values are raw fractions (0..1), not formatted percentages, so the export is directly
  usable for spreadsheet math.
- Existing correction flow (`POST /api/masks/{jid}`) does **not** persist the recomputed verdict back
  into the job's stored `result` — the existing PDF export already reflects only the original
  (uncorrected) analysis, not any in-browser correction (`vOverride`). CSV export must match this
  existing behavior for consistency: it reads from the polled `Job` record, not from `vOverride`.
- The reproducibility `params` snapshot (Task 3) is stored for audit purposes only — it is
  deliberately **not** a CSV column (out of the confirmed CSV scope), retrievable via
  `GET /api/jobs/{jid}`.

---

## Task 1: JobStore gains batch grouping + filename + created_at

**Files:**
- Modify: `backend/app/jobs/store.py`
- Modify: `backend/app/schemas/jobs.py`
- Test: `backend/tests/test_job_batch.py` (new)

**Interfaces:**
- Consumes: nothing new (pure schema/store change).
- Produces: `JobStore.create(mode: str, batch_id: str | None = None, filename: str | None = None) -> str`;
  `JobStore.get(jid: str) -> JobRecord | None` (now includes `batch_id`, `filename`, `created_at`);
  `JobStore.list_by_batch(batch_id: str) -> list[JobRecord]`; `JobRecord` gains
  `batch_id: Optional[str]`, `filename: Optional[str]`, `created_at: Optional[str]`. Tasks 2 and 4
  depend on these exact names/signatures.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_job_batch.py`:

```python
import sqlite3
from app.jobs.store import JobStore


def test_create_stores_batch_id_and_filename(tmp_path):
    store = JobStore(tmp_path / "t.db")
    jid = store.create("closeup", batch_id="batch-1", filename="a.png")
    rec = store.get(jid)
    assert rec.batch_id == "batch-1"
    assert rec.filename == "a.png"
    assert rec.created_at is not None


def test_create_without_batch_id_leaves_it_null(tmp_path):
    store = JobStore(tmp_path / "t.db")
    jid = store.create("closeup")
    rec = store.get(jid)
    assert rec.batch_id is None
    assert rec.filename is None


def test_list_by_batch_returns_jobs_in_creation_order(tmp_path):
    store = JobStore(tmp_path / "t.db")
    j1 = store.create("closeup", batch_id="batch-1", filename="a.png")
    j2 = store.create("closeup", batch_id="batch-1", filename="b.png")
    store.create("closeup", batch_id="other-batch", filename="c.png")
    recs = store.list_by_batch("batch-1")
    assert [r.id for r in recs] == [j1, j2]
    assert [r.filename for r in recs] == ["a.png", "b.png"]


def test_list_by_batch_empty_for_unknown_batch(tmp_path):
    store = JobStore(tmp_path / "t.db")
    assert store.list_by_batch("nope") == []


def test_legacy_db_without_batch_columns_migrates_cleanly(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE jobs(
        id TEXT PRIMARY KEY, mode TEXT, status TEXT, progress REAL,
        message TEXT, result TEXT)""")
    conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                 ("old1", "closeup", "done", 1.0, None, None))
    conn.commit()
    conn.close()

    store = JobStore(db_path)
    rec = store.get("old1")
    assert rec.batch_id is None
    assert rec.filename is None

    new_jid = store.create("closeup", batch_id="b1", filename="x.jpg")
    assert store.get(new_jid).batch_id == "b1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_job_batch.py -v`
Expected: FAIL — `TypeError: create() got an unexpected keyword argument 'batch_id'` (or
`AttributeError: 'JobStore' object has no attribute 'list_by_batch'`).

- [ ] **Step 3: Update `JobRecord`**

In `backend/app/schemas/jobs.py`, replace the class body:

```python
class JobRecord(BaseModel):
    id: str
    mode: str
    status: Status = "queued"
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    batch_id: Optional[str] = None
    filename: Optional[str] = None
    created_at: Optional[str] = None
```

- [ ] **Step 4: Update `JobStore`**

In `backend/app/jobs/store.py`:

Add `time` to the top-level import (`import json, sqlite3, threading, time, uuid`).

Replace `__init__`:

```python
    def __init__(self, db_path: Path):
        self._path = str(db_path)
        self._lock = threading.Lock()
        with self._tx() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, mode TEXT, status TEXT, progress REAL,
                message TEXT, result TEXT, batch_id TEXT, filename TEXT, created_at TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS corrections(
                id TEXT PRIMARY KEY, job_id TEXT, layer TEXT, n_pixels INTEGER, ts TEXT)""")
            existing = {row[1] for row in c.execute("PRAGMA table_info(jobs)").fetchall()}
            for col in ("batch_id", "filename", "created_at"):
                if col not in existing:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
```

Replace `create`:

```python
    def create(self, mode: str, batch_id: str | None = None, filename: str | None = None) -> str:
        jid = uuid.uuid4().hex
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock, self._tx() as c:
            c.execute(
                "INSERT INTO jobs(id,mode,status,progress,message,result,batch_id,filename,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (jid, mode, "queued", 0.0, None, None, batch_id, filename, created_at))
        return jid
```

Replace `get`:

```python
    def get(self, jid: str) -> JobRecord | None:
        with self._tx() as c:
            row = c.execute(
                "SELECT id,mode,status,progress,message,result,batch_id,filename,created_at "
                "FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row: return None
        return JobRecord(id=row[0], mode=row[1], status=row[2], progress=row[3],
                         message=row[4], result=json.loads(row[5]) if row[5] else None,
                         batch_id=row[6], filename=row[7], created_at=row[8])
```

Add a new method (place after `get`):

```python
    def list_by_batch(self, batch_id: str) -> list[JobRecord]:
        with self._tx() as c:
            rows = c.execute(
                "SELECT id,mode,status,progress,message,result,batch_id,filename,created_at "
                "FROM jobs WHERE batch_id=? ORDER BY created_at, rowid", (batch_id,)).fetchall()
        return [JobRecord(id=r[0], mode=r[1], status=r[2], progress=r[3], message=r[4],
                          result=json.loads(r[5]) if r[5] else None,
                          batch_id=r[6], filename=r[7], created_at=r[8]) for r in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_job_batch.py tests/test_jobs.py -v`
Expected: PASS (all tests in both files — `test_jobs.py`'s existing lifecycle tests must still pass
unchanged since `batch_id`/`filename` are optional with defaults).

- [ ] **Step 6: Run the full backend suite to catch regressions**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS (no other test constructs `JobRecord`/`JobStore` positionally in a way this breaks —
`test_report.py` builds its own plain dicts, not `JobRecord`, so it is unaffected).

- [ ] **Step 7: Commit**

```bash
git add backend/app/jobs/store.py backend/app/schemas/jobs.py backend/tests/test_job_batch.py
git commit -m "feat(jobs): batch_id/filename/created_at on JobRecord + list_by_batch"
```

---

## Task 2: `POST /api/analyze` accepts an optional `batch_id`

**Files:**
- Modify: `backend/app/api/analyze.py`
- Test: `backend/tests/test_batch_api.py` (new)

**Interfaces:**
- Consumes: `JobStore.create(mode, batch_id=None, filename=None)` from Task 1.
- Produces: `POST /api/analyze` now accepts an optional multipart form field `batch_id`; every job
  (batch or not) gets its original upload `filename` stored. Tasks 3 and 4 build on this same file.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_batch_api.py`:

```python
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from main import app


def _png_bytes(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def _poll(c, jid):
    for _ in range(100):
        r = c.get(f"/api/jobs/{jid}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_analyze_stores_batch_id_and_filename(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze",
                data={"batch_id": "batch-xyz"},
                files={"image": ("sample.png", _png_bytes(tiny_rgb), "image/png")})
    assert up.status_code == 200
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["batch_id"] == "batch-xyz"
    assert done["filename"] == "sample.png"


def test_analyze_without_batch_id_leaves_it_null(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", files={"image": ("solo.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["batch_id"] is None
    assert done["filename"] == "solo.png"
```

(`tiny_rgb` comes from `backend/tests/conftest.py`, already shared by other API tests.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_batch_api.py -v`
Expected: FAIL — `filename`/`batch_id` come back `None` because `analyze()` never passes them to
`store.create`.

- [ ] **Step 3: Wire `batch_id` + `filename` through `analyze()`**

In `backend/app/api/analyze.py`, change the import line:

```python
from fastapi import APIRouter, UploadFile, File, Form
```

Change the route signature and the `store.create` call:

```python
@router.post("/analyze")
async def analyze(image: UploadFile = File(...), batch_id: str | None = Form(None)):
    data = await image.read()
    cfg = loader.get_config()
    iw, ih = Image.open(io.BytesIO(data)).size
    mode = detect.detect_mode(iw, ih, cfg)
    jid = get_runtime().store.create(mode, batch_id=batch_id, filename=image.filename)
```

(Everything below that line in the function is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_batch_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/analyze.py backend/tests/test_batch_api.py
git commit -m "feat(api): accept optional batch_id on /api/analyze, store upload filename"
```

---

## Task 3: Log analysis parameters for reproducibility

This closes the spec's "logging analysis parameters for reproducibility" requirement. It piggybacks
on data the pipeline already computes (`loader.model_status()`, `loader.gpu_available()`) — no new
infrastructure, and it's added from *outside* `panorama.py`/`closeup.py` (wrapping their returned
dict in `analyze.py`), so those pipeline modules stay untouched.

**Files:**
- Modify: `backend/app/api/analyze.py`
- Test: `backend/tests/test_batch_api.py`

**Interfaces:**
- Consumes: `loader.model_status() -> dict`, `loader.gpu_available() -> bool` (existing,
  `backend/app/pipeline/loader.py`).
- Produces: every job's stored `result` dict gains a `params` key:
  `{"mode": str, "models": dict, "gpu": bool}`. Not a CSV column (see Global Constraints) —
  available via `GET /api/jobs/{jid}` for audit/debugging.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_batch_api.py`:

```python
def test_analyze_result_includes_reproducibility_params(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", files={"image": ("p.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    params = done["result"]["params"]
    assert params["mode"] == "closeup"
    assert "models" in params
    assert "gpu" in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_batch_api.py -v -k reproducibility`
Expected: FAIL — `KeyError: 'params'`.

- [ ] **Step 3: Add the params snapshot**

In `backend/app/api/analyze.py`, add a helper right after the `Image.MAX_IMAGE_PIXELS = None` line:

```python
def _analysis_params(mode: str) -> dict:
    return {"mode": mode, "models": loader.model_status(), "gpu": loader.gpu_available()}
```

Then update `work()` to attach it to both the panorama and closeup return values. The full function
(with Task 2's `batch_id` change already applied) becomes:

```python
    def work(report):
        if mode == "panorama":
            result = panorama.analyze_panorama(str(up), cfg, jid, on_progress=report)
            result["params"] = _analysis_params(mode)
            return result
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
                "low_conf_zones": r["low_conf_zones"],
                "params": _analysis_params(mode)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_batch_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS (confirms no test elsewhere asserts an exact `result == {...}` dict that the new
`params` key would break — already checked: the only such exact-match assertion in the suite,
`test_jobs.py::test_job_lifecycle_success`, calls `JobRunner.submit` directly with a raw lambda and
never goes through `analyze.py`, so it's unaffected).

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/analyze.py backend/tests/test_batch_api.py
git commit -m "feat(api): record analysis params (mode/models/gpu) on each job for reproducibility"
```

---

## Task 4: `GET /api/jobs?batch_id=` list endpoint

**Files:**
- Modify: `backend/app/api/jobs.py`
- Modify: `backend/tests/test_batch_api.py`

**Interfaces:**
- Consumes: `JobStore.list_by_batch(batch_id)` from Task 1; `batch_id` form field from Task 2.
- Produces: `GET /api/jobs?batch_id=<id>` -> `list[JobRecord]` JSON (required query param, 422 if
  missing). Frontend Task 6 (`listJobsByBatch`) depends on this exact path/query-param name.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_batch_api.py`:

```python
def test_list_jobs_by_batch_returns_all_members(tiny_rgb):
    c = TestClient(app)
    jids = []
    for name in ("one.png", "two.png"):
        r = c.post("/api/analyze", data={"batch_id": "batch-list"},
                   files={"image": (name, _png_bytes(tiny_rgb), "image/png")})
        jids.append(r.json()["job_id"])
    for jid in jids:
        _poll(c, jid)
    listed = c.get("/api/jobs", params={"batch_id": "batch-list"}).json()
    assert sorted(j["id"] for j in listed) == sorted(jids)
    assert {j["filename"] for j in listed} == {"one.png", "two.png"}


def test_list_jobs_requires_batch_id_query_param():
    c = TestClient(app)
    r = c.get("/api/jobs")
    assert r.status_code == 422


def test_list_jobs_empty_for_unknown_batch():
    c = TestClient(app)
    r = c.get("/api/jobs", params={"batch_id": "does-not-exist"})
    assert r.status_code == 200
    assert r.json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_batch_api.py -v -k list_jobs`
Expected: FAIL — 404 Not Found (no `GET /api/jobs` route without a path param yet).

- [ ] **Step 3: Add the list route**

In `backend/app/api/jobs.py`, add above the existing `get_job` route:

```python
@router.get("/jobs")
def list_jobs(batch_id: str):
    return get_runtime().store.list_by_batch(batch_id)
```

Full resulting file:

```python
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.runtime import get_runtime

router = APIRouter()

@router.get("/jobs")
def list_jobs(batch_id: str):
    return get_runtime().store.list_by_batch(batch_id)

@router.get("/jobs/{jid}")
def get_job(jid: str):
    rec = get_runtime().store.get(jid)
    if rec is None:
        raise HTTPException(404, "job not found")
    return rec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_batch_api.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/jobs.py backend/tests/test_batch_api.py
git commit -m "feat(api): GET /api/jobs?batch_id= to list a batch's jobs"
```

---

## Task 5: Configurable job-worker concurrency

**Files:**
- Modify: `backend/app/runtime.py`
- Modify: `backend/tests/test_runtime.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SHLIF_JOB_WORKERS` env var controls `Runtime.runner`'s pool size (default `4`, was a
  hardcoded `2`). No API-visible change; purely an operational knob so batch uploads actually run
  concurrently instead of queueing one-at-a-time behind a long panorama analysis.

- [ ] **Step 1: Write the failing test**

Replace `backend/tests/test_runtime.py` entirely:

```python
"""Runtime wires JobRunner's worker count from SHLIF_JOB_WORKERS (default 4) so a batch
upload of many images actually runs concurrently in the background instead of queueing
strictly one at a time behind a long-running panorama analysis."""
from app.runtime import Runtime


def test_runner_uses_default_worker_count_without_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.paths.db_path", lambda: tmp_path / "t.db")
    monkeypatch.delenv("SHLIF_JOB_WORKERS", raising=False)
    rt = Runtime()
    assert rt.runner._pool._max_workers == 4


def test_runner_honors_shlif_job_workers_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.paths.db_path", lambda: tmp_path / "t.db")
    monkeypatch.setenv("SHLIF_JOB_WORKERS", "6")
    rt = Runtime()
    assert rt.runner._pool._max_workers == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && .venv/bin/pytest tests/test_runtime.py -v`
Expected: FAIL — `assert 2 == 4` (current hardcoded value).

- [ ] **Step 3: Make the worker count configurable**

Replace `backend/app/runtime.py`:

```python
from __future__ import annotations
import os
from app.core import paths

class Runtime:
    """Holds the app-scoped job store + runner (single granian worker)."""
    def __init__(self) -> None:
        from app.jobs.store import JobStore
        from app.jobs.runner import JobRunner
        self.store = JobStore(paths.db_path())
        workers = int(os.environ.get("SHLIF_JOB_WORKERS", "4"))
        self.runner = JobRunner(self.store, max_workers=workers)

_runtime: Runtime | None = None

def get_runtime() -> Runtime:
    """Return the process-wide Runtime, creating it on first use (call-time, not import-time)."""
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && .venv/bin/pytest tests/test_runtime.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS (this is the last backend task — the whole backend suite should be green here).

- [ ] **Step 6: Commit**

```bash
git add backend/app/runtime.py backend/tests/test_runtime.py
git commit -m "feat(runtime): make job-worker pool size configurable via SHLIF_JOB_WORKERS"
```

---

## Task 6: Frontend API surface for batches

**Files:**
- Modify: `frontend/lib/api/types.ts`
- Modify: `frontend/lib/api/client.ts`
- Modify: `frontend/lib/api/hooks.ts`

**Interfaces:**
- Consumes: `GET /api/jobs?batch_id=` and the `batch_id` form field from Tasks 2/4.
- Produces: `Job` type gains `batch_id: string | null; filename: string | null; created_at: string | null`;
  `analyze(file: File, batchId?: string): Promise<{ job_id: string }>`;
  `listJobsByBatch(batchId: string): Promise<Job[]>`; `useBatchJobs(batchId: string | null)` (React
  Query hook). Tasks 10/11 (BatchGallery, page.tsx) depend on all of these exact names.

No new automated test in this task: `getJob`/`analyze` (the existing fetch-calling functions this
mirrors) have never had unit tests in this repo either — `client.test.mjs` only covers pure URL
builders, and there's no `hooks.test.mjs` (React Query hooks aren't unit-tested here). This task is
verified by `npm run build` (typecheck) now, and by Task 12's end-to-end pass later.

- [ ] **Step 1: Extend the `Job` type**

In `frontend/lib/api/types.ts`, replace the `Job` interface:

```ts
export interface Job {
  id: string; mode: string; status: "queued" | "running" | "done" | "error";
  progress: number; message: string | null; result: AnalyzeResult | null;
  batch_id: string | null; filename: string | null; created_at: string | null;
}
```

- [ ] **Step 2: Extend `analyze()` and add `listJobsByBatch()`**

In `frontend/lib/api/client.ts`, replace `analyze`:

```ts
export async function analyze(file: File, batchId?: string): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("image", file);
  if (batchId) fd.append("batch_id", batchId);
  const r = await fetch(`${base}/api/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`analyze failed: ${r.status}`);
  return r.json();
}
```

Add, near `getJob`:

```ts
export async function listJobsByBatch(batchId: string): Promise<Job[]> {
  const r = await fetch(`${base}/api/jobs?batch_id=${encodeURIComponent(batchId)}`);
  if (!r.ok) throw new Error(`batch list failed: ${r.status}`);
  return r.json();
}
```

- [ ] **Step 3: Add `useBatchJobs`**

In `frontend/lib/api/hooks.ts`, add (leave `useAnalyze`/`useJob` untouched):

```ts
import { analyze, getJob, listJobsByBatch } from "./client";
```

(replaces the existing `import { analyze, getJob } from "./client";` line)

```ts
export function useBatchJobs(batchId: string | null) {
  return useQuery({
    queryKey: ["batch", batchId],
    queryFn: () => listJobsByBatch(batchId as string),
    enabled: !!batchId,
    refetchInterval: (q) => {
      const jobs = q.state.data;
      // No rows yet could mean "still uploading" — keep polling rather than giving up.
      if (!jobs || jobs.length === 0) return 800;
      const pending = jobs.some((j) => j.status === "queued" || j.status === "running");
      return pending ? 800 : false;
    },
  });
}
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npm run build`
Expected: build succeeds (no TypeScript errors). Ignore pre-existing unrelated warnings if any.

- [ ] **Step 5: Run the existing frontend test suite**

Run: `cd frontend && npm test`
Expected: PASS (nothing in this task touches tested pure functions, so this just guards against
accidental breakage).

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/api/types.ts frontend/lib/api/client.ts frontend/lib/api/hooks.ts
git commit -m "feat(frontend): batch-aware Job type, analyze(batchId), useBatchJobs"
```

---

## Task 7: CSV serialization

**Files:**
- Create: `frontend/lib/csv.ts`
- Test: `frontend/tests/csv.test.mjs` (new)

**Interfaces:**
- Consumes: `Job` type from Task 6 (`batch_id`, `filename`, `created_at`, `result.verdict.metrics`,
  `result.verdict.ore_class`).
- Produces: `jobsToCsv(jobs: Job[]): string`; `downloadCsv(filename: string, csv: string): void`.
  Tasks 10/11 (BatchGallery, page.tsx) call both by these exact names.

- [ ] **Step 1: Write the failing tests**

Create `frontend/tests/csv.test.mjs`:

```ts
import { test } from "node:test";
import assert from "node:assert";
import { jobsToCsv } from "../lib/csv.ts";

function job(overrides = {}) {
  return {
    id: "job1", mode: "closeup", status: "done", progress: 1, message: null,
    batch_id: null, filename: "sample.png", created_at: "2026-07-05T10:00:00",
    result: {
      mode: "closeup",
      verdict: {
        ore_class: "ordinary", text: "заключение",
        metrics: {
          sulfide_frac: 0.21, magnetite_frac: 0.05, matrix_frac: 0.74,
          talc_frac: 0.03, talc_share_est: 0.04, fine_share: 0.3,
          confidence: 0.71, undetermined_fraction: 0.08,
        },
      },
      sort: null,
    },
    ...overrides,
  };
}

test("jobsToCsv writes a header row and one data row per job", () => {
  const csv = jobsToCsv([job()]);
  const lines = csv.split("\r\n");
  assert.strictEqual(lines.length, 2);
  assert.strictEqual(
    lines[0],
    "job_id,filename,mode,status,ore_class,sulfide_frac,magnetite_frac,matrix_frac,talc_frac,talc_share_est,fine_share,confidence,undetermined_fraction,created_at",
  );
  assert.strictEqual(
    lines[1],
    "job1,sample.png,closeup,done,ordinary,0.21,0.05,0.74,0.03,0.04,0.3,0.71,0.08,2026-07-05T10:00:00",
  );
});

test("jobsToCsv leaves missing metrics blank instead of crashing", () => {
  const pano = job({
    id: "job2", mode: "panorama", filename: "pano.png",
    result: {
      mode: "panorama",
      verdict: { ore_class: "review", text: "", metrics: { talc_frac: 0.01, confidence: 0.4 } },
      sort: null,
    },
  });
  const row = jobsToCsv([pano]).split("\r\n")[1];
  assert.strictEqual(
    row,
    "job2,pano.png,panorama,done,review,,,,0.01,,,0.4,,2026-07-05T10:00:00",
  );
});

test("jobsToCsv escapes commas, quotes and newlines in filenames", () => {
  const weird = job({ id: "job3", filename: 'a, "tricky"\nname.png' });
  const row = jobsToCsv([weird]).split("\r\n")[1];
  assert.ok(row.startsWith('job3,"a, ""tricky""\nname.png",closeup,done,ordinary,'));
});

test("jobsToCsv handles a job with no result yet (queued/running)", () => {
  const pending = job({ id: "job4", status: "queued", filename: "later.png", result: null });
  const row = jobsToCsv([pending]).split("\r\n")[1];
  assert.strictEqual(row, "job4,later.png,closeup,queued,,,,,,,,,,2026-07-05T10:00:00");
});

test("jobsToCsv concatenates multiple jobs as separate rows", () => {
  const csv = jobsToCsv([job({ id: "a" }), job({ id: "b", filename: "second.png" })]);
  assert.strictEqual(csv.split("\r\n").length, 3);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && node --import tsx --test tests/csv.test.mjs`
Expected: FAIL — cannot find module `../lib/csv.ts`.

- [ ] **Step 3: Implement `lib/csv.ts`**

Create `frontend/lib/csv.ts`:

```ts
import type { Job } from "./api/types";

const METRIC_KEYS = [
  "sulfide_frac", "magnetite_frac", "matrix_frac", "talc_frac",
  "talc_share_est", "fine_share", "confidence", "undetermined_fraction",
] as const;

const HEADERS = ["job_id", "filename", "mode", "status", "ore_class", ...METRIC_KEYS, "created_at"];

function csvEscape(value: string): string {
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

function jobRow(job: Job): string {
  const metrics = job.result?.verdict?.metrics ?? {};
  const cells = [
    job.id,
    job.filename ?? "",
    job.mode,
    job.status,
    job.result?.verdict?.ore_class ?? "",
    ...METRIC_KEYS.map((k) => (metrics[k] != null ? String(metrics[k]) : "")),
    job.created_at ?? "",
  ];
  return cells.map(csvEscape).join(",");
}

export function jobsToCsv(jobs: Job[]): string {
  return [HEADERS.join(","), ...jobs.map(jobRow)].join("\r\n");
}

export function downloadCsv(filename: string, csv: string): void {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && node --import tsx --test tests/csv.test.mjs`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run the full frontend suite**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/csv.ts frontend/tests/csv.test.mjs
git commit -m "feat(frontend): client-side CSV export for jobs (single + batch)"
```

---

## Task 8: Batch URL persistence

**Files:**
- Create: `frontend/lib/batchUrl.ts`
- Test: `frontend/tests/batchUrl.test.mjs` (new)

**Interfaces:**
- Consumes: nothing (pure query-string logic, mirrors the existing `frontend/lib/jobUrl.ts`).
- Produces: `parseBatchParams(sp: URLSearchParams): { batchId: string | null }`;
  `buildBatchQuery(batchId: string | null, jobId: string | null): string`. Task 11 (page.tsx)
  depends on both names.

- [ ] **Step 1: Write the failing tests**

Create `frontend/tests/batchUrl.test.mjs`:

```ts
import { test } from "node:test";
import assert from "node:assert";
import { parseBatchParams, buildBatchQuery } from "../lib/batchUrl.ts";

test("parseBatchParams reads batch", () => {
  assert.deepStrictEqual(parseBatchParams(new URLSearchParams("batch=b1")), { batchId: "b1" });
});

test("parseBatchParams is empty without a batch", () => {
  assert.deepStrictEqual(parseBatchParams(new URLSearchParams("job=abc")), { batchId: null });
});

test("buildBatchQuery is empty without a batchId", () => {
  assert.strictEqual(buildBatchQuery(null, null), "");
});

test("buildBatchQuery encodes batch alone", () => {
  assert.strictEqual(buildBatchQuery("b1", null), "batch=b1");
});

test("buildBatchQuery encodes batch and job together", () => {
  assert.strictEqual(buildBatchQuery("b1", "j1"), "batch=b1&job=j1");
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && node --import tsx --test tests/batchUrl.test.mjs`
Expected: FAIL — cannot find module `../lib/batchUrl.ts`.

- [ ] **Step 3: Implement `lib/batchUrl.ts`**

Create `frontend/lib/batchUrl.ts`:

```ts
// Кодирование/декодирование batch_id в query-параметр адресной строки — по аналогии с
// jobUrl.ts, чтобы перезагрузка страницы во время партийной обработки возвращала в галерею
// партии (а не на пустой экран загрузки), а не только к одиночному job.

export interface BatchParams {
  batchId: string | null;
}

export function parseBatchParams(sp: URLSearchParams): BatchParams {
  return { batchId: sp.get("batch") };
}

export function buildBatchQuery(batchId: string | null, jobId: string | null): string {
  if (!batchId) return "";
  const params = new URLSearchParams();
  params.set("batch", batchId);
  if (jobId) params.set("job", jobId);
  return params.toString();
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && node --import tsx --test tests/batchUrl.test.mjs`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Run the full frontend suite**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/lib/batchUrl.ts frontend/tests/batchUrl.test.mjs
git commit -m "feat(frontend): batch_id URL persistence (mirrors jobUrl.ts)"
```

---

## Task 9: `Welcome` accepts multiple files

**Files:**
- Modify: `frontend/components/Welcome.tsx`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Welcome`'s prop is renamed `onFile: (f: File) => void` -> `onFiles: (files: File[]) => void`.
  Task 11 (page.tsx) must update its `<Welcome .../>` call site to match.

No dedicated automated test: this repo has no component-render test harness (no React Testing
Library), and `Welcome.tsx` has never had one either. Verified via `npm run build` here and the
manual pass in Task 12.

- [ ] **Step 1: Update the component**

Replace `frontend/components/Welcome.tsx`'s body from the `export function Welcome` line onward:

```tsx
export function Welcome({ onFiles }: { onFiles: (files: File[]) => void }) {
  const [drag, setDrag] = useState(false);

  function pickFromInput(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length) onFiles(files);
  }
  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (files.length) onFiles(files);
  }

  return (
    <section className="welcome full" aria-label="DATA FORCE — классификация руд">
      <div className="welcome-bg" />
      <div className="welcome-scrim" />
      <div className="welcome-inner">
        <h1 className="welcome-title">
          <span className="wm">DATA&nbsp;FORCE</span>
          <span className="rocket">🚀</span>
        </h1>
        <p className="welcome-tagline">Скажи мне, кто твой шлиф</p>
        <p className="welcome-desc">
          Определяет сорт руды по снимку шлифа — рядовая, труднообогатимая или оталькованная — за секунды
          вместо часов ручной работы геолога. Нейросеть отделяет руду от породы и оценивает сорт по размеру
          зёрен сульфидов, а спорные зоны и итоговую маску можно проверить и поправить прямо в интерфейсе.
        </p>
        <label
          className={`dropzone${drag ? " drag" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragEnter={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
        >
          <IconUpload className="ico-lg dz-ico" />
          <span className="dz-title">Перетащите снимок шлифа сюда</span>
          <span className="dz-sub">или нажмите, чтобы выбрать (можно сразу несколько) · JPG / PNG · OM, отражённый свет</span>
          <input type="file" accept="image/*" multiple onChange={pickFromInput} style={{ display: "none" }} />
        </label>
      </div>
      <div className="welcome-credits">
        <span className="wc-label">Команда</span>
        <div className="wc-links">
          {TEAM.map((m) => (
            <a key={m.url} className="wc-link" href={m.url} target="_blank" rel="noopener noreferrer">
              <IconTelegram className="ico-sm" />{m.name}
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
```

(The `"use client"` directive, the comment block, imports, and the `TEAM` constant above this stay
unchanged — only the exported function's signature/body changes.)

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npm run build`
Expected: FAILS at this point — `page.tsx` still passes `onFile` to `<Welcome>`. This is expected;
Task 11 fixes the call site (Task 10, BatchGallery, doesn't touch page.tsx either, so the build stays
red through both). Confirm the *only* new error is in `app/page.tsx` about the `onFile`/`onFiles`
prop mismatch (not some unrelated break).

- [ ] **Step 3: Commit**

```bash
git add frontend/components/Welcome.tsx
git commit -m "feat(frontend): Welcome accepts multiple files for batch upload"
```

(This task intentionally leaves the build red until Task 11 rewires `page.tsx` — that's the only
place `<Welcome>` is used. If executing tasks out of order, don't treat this red build as a
regression; it's resolved two tasks later.)

---

## Task 10: Status labels + `BatchGallery` component

**Files:**
- Create: `frontend/lib/statusLabels.ts`
- Create: `frontend/components/batch/BatchGallery.tsx`
- Modify: `frontend/app/globals.css`

**Interfaces:**
- Consumes: `Job` type (Task 6), `jobsToCsv`/`downloadCsv` (Task 7), `IconDownload`/`IconUpload`
  (existing `frontend/components/icons.tsx`), `imageUrl` (existing `frontend/lib/api/client.ts`).
- Produces: `STATUS_LABELS: Record<string, [string, string]>`; `BatchGallery({ batchId, jobs, onOpen,
  onNewAnalysis })` component. Task 11 (page.tsx) imports both.

No dedicated automated test (same rationale as Task 9 — no component-render harness in this repo).
Verified via `npm run build` and Task 12's manual pass.

- [ ] **Step 1: Extract shared status labels**

Create `frontend/lib/statusLabels.ts`:

```ts
export const STATUS_LABELS: Record<string, [string, string]> = {
  queued: ["queued", "в очереди"],
  running: ["running", "анализ"],
  done: ["done", "готово"],
  error: ["error", "ошибка"],
};
```

- [ ] **Step 2: Create `BatchGallery`**

Create `frontend/components/batch/BatchGallery.tsx`:

```tsx
"use client";
import type { Job } from "@/lib/api/types";
import { jobsToCsv, downloadCsv } from "@/lib/csv";
import { STATUS_LABELS } from "@/lib/statusLabels";
import { imageUrl } from "@/lib/api/client";
import { IconDownload, IconUpload } from "@/components/icons";

export function BatchGallery({
  batchId, jobs, onOpen, onNewAnalysis,
}: {
  batchId: string;
  jobs: Job[];
  onOpen: (jobId: string) => void;
  onNewAnalysis: () => void;
}) {
  const doneCount = jobs.filter((j) => j.status === "done").length;

  return (
    <div className="batch-gallery">
      <div className="batch-head">
        <div>
          <div className="side-h">Партия снимков</div>
          <div className="sub">{doneCount} / {jobs.length} готово</div>
        </div>
        <div className="grow" />
        <button
          type="button"
          className="btn ghost sm"
          disabled={doneCount === 0}
          onClick={() => downloadCsv(`shlif-batch-${batchId}.csv`, jobsToCsv(jobs))}
        >
          <IconDownload className="ico-sm" /> Скачать CSV (партия)
        </button>
        <button type="button" className="btn ghost sm" onClick={onNewAnalysis}>
          <IconUpload className="ico-sm" /> Новый анализ
        </button>
      </div>
      {jobs.length === 0 ? (
        <div className="stage-empty"><div className="hint">Загрузка файлов…</div></div>
      ) : (
        <div className="batch-grid">
          {jobs.map((job) => {
            const badge = STATUS_LABELS[job.status];
            return (
              <button key={job.id} type="button" className="batch-card" onClick={() => onOpen(job.id)}>
                {job.status === "done" ? (
                  <img className="batch-card-thumb" src={imageUrl(job.id)} alt={job.filename ?? job.id} />
                ) : (
                  <div className="batch-card-thumb placeholder" aria-hidden="true" />
                )}
                <span className="batch-card-name">{job.filename ?? job.id}</span>
                {badge ? (
                  <span className={`status-badge ${badge[0]}`}><span className="bd" />{badge[1]}</span>
                ) : null}
                {job.status === "error" ? <span className="batch-card-err">{job.message}</span> : null}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Add gallery CSS**

In `frontend/app/globals.css`, find this existing rule:

```css
.ws-view { min-width: 0; display: grid; gap: 10px; }
```

Add immediately after it:

```css

.batch-gallery { display: grid; gap: 16px; }
.batch-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.batch-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 14px; }
.batch-card { display: grid; gap: 8px; padding: 10px; border-radius: var(--radius-lg); border: 1px solid var(--border); background: var(--surface); cursor: pointer; text-align: left; font-family: inherit; }
.batch-card:hover { border-color: var(--brand); }
.batch-card-thumb { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 8px; background: var(--surface-2); display: block; }
.batch-card-name { font-size: 12.5px; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.batch-card-err { font-size: 11.5px; color: var(--danger-ink); }
```

- [ ] **Step 4: Typecheck**

Run: `cd frontend && npm run build`
Expected: still FAILS on the same `page.tsx` `onFile`/`onFiles` mismatch as Task 9 left it (nothing
new from this task — `BatchGallery`/`statusLabels.ts` aren't imported anywhere yet). Confirm no
*additional* errors beyond that one.

- [ ] **Step 5: Commit**

```bash
git add frontend/lib/statusLabels.ts frontend/components/batch/BatchGallery.tsx frontend/app/globals.css
git commit -m "feat(frontend): BatchGallery component + shared status labels"
```

---

## Task 11: Wire batch upload + CSV export into `page.tsx`

**Files:**
- Modify: `frontend/app/page.tsx`

**Interfaces:**
- Consumes: everything from Tasks 6–10 (`useBatchJobs`, `jobsToCsv`/`downloadCsv`,
  `parseBatchParams`/`buildBatchQuery`, `Welcome`'s new `onFiles` prop, `BatchGallery`,
  `STATUS_LABELS`).
- Produces: the fully wired page — single-file flow unchanged; 2+ files trigger a batch (fire all
  uploads concurrently, gallery view, click-through to the existing single-job workspace, back-to-
  gallery button, CSV buttons in both the gallery and the single-job info panel, batch survives a
  page reload via `?batch=`).

- [ ] **Step 1: Replace `frontend/app/page.tsx`**

```tsx
"use client";
import { Suspense, useEffect, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { analyze as analyzeApi } from "@/lib/api/client";
import { useAnalyze, useJob, useBatchJobs } from "@/lib/api/hooks";
import { reportUrl } from "@/lib/api/client";
import type { Verdict } from "@/lib/api/types";
import { buildJobQuery, parseJobParams } from "@/lib/jobUrl";
import { parseBatchParams, buildBatchQuery } from "@/lib/batchUrl";
import { jobsToCsv, downloadCsv } from "@/lib/csv";
import { STATUS_LABELS } from "@/lib/statusLabels";
import { VerdictPanel } from "@/components/verdict/VerdictPanel";
import { Corrector } from "@/components/corrector/Corrector";
import { Welcome } from "@/components/Welcome";
import { BatchGallery } from "@/components/batch/BatchGallery";
import { ThemeToggle } from "@/components/ThemeToggle";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { IconAlert, IconDownload, IconUpload } from "@/components/icons";
import { PanoramaZoomModal } from "@/components/PanoramaZoomModal";

function Home() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [restoredJob] = useState(() => parseJobParams(searchParams));
  const [restoredBatch] = useState(() => parseBatchParams(searchParams));

  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(() => restoredJob.jobId);
  const [startedAt, setStartedAt] = useState<number | null>(() => restoredJob.startedAt ?? (restoredJob.jobId ? Date.now() : null));
  const [vOverride, setVOverride] = useState<Verdict | null>(null);
  const [batchId, setBatchId] = useState<string | null>(() => restoredBatch.batchId);
  const [uploadFailures, setUploadFailures] = useState<{ filename: string; error: string }[]>([]);
  const analyze = useAnalyze();
  const job = useJob(jobId);
  const batchJobs = useBatchJobs(batchId);

  // Держим ссылку в актуальном состоянии: перезагрузка страницы должна вернуть к тому же
  // анализу (или к той же партии), а не к пустому экрану загрузки.
  useEffect(() => {
    const qs = batchId ? buildBatchQuery(batchId, jobId) : buildJobQuery(jobId, startedAt);
    if (qs === searchParams.toString()) return;
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }, [jobId, startedAt, batchId, pathname, router, searchParams]);

  // Job, восстановленный из ссылки, мог устареть или быть удалён на сервере — откатываемся
  // на партию (если она есть) или на экран загрузки, а не зависаем на "Анализ снимка…".
  useEffect(() => {
    if (job.isError && jobId) {
      setJobId(null);
      setStartedAt(null);
      setVOverride(null);
    }
  }, [job.isError, jobId]);

  function runAnalyze(f: File) {
    setFile(f);
    setVOverride(null);
    setJobId(null);
    setStartedAt(Date.now());
    analyze.mutate({ file: f }, { onSuccess: (r) => setJobId(r.job_id) });
  }

  function startBatch(files: File[]) {
    const id = crypto.randomUUID();
    setFile(null);
    setJobId(null);
    setStartedAt(null);
    setVOverride(null);
    setUploadFailures([]);
    setBatchId(id);
    Promise.allSettled(files.map((f) => analyzeApi(f, id))).then((results) => {
      const failed = results
        .map((res, i) => (res.status === "rejected" ? { filename: files[i].name, error: String(res.reason) } : null))
        .filter((x): x is { filename: string; error: string } => x !== null);
      if (failed.length) setUploadFailures((prev) => [...prev, ...failed]);
    });
  }

  function handleFiles(files: File[]) {
    if (files.length <= 1) {
      if (files[0]) runAnalyze(files[0]);
      return;
    }
    startBatch(files);
  }

  function openBatchItem(id: string) {
    setJobId(id);
    setStartedAt(Date.now());
    setVOverride(null);
  }

  function backToBatch() {
    setJobId(null);
    setStartedAt(null);
    setVOverride(null);
  }

  function resetToUpload() {
    analyze.reset();
    setFile(null);
    setJobId(null);
    setStartedAt(null);
    setVOverride(null);
    setBatchId(null);
    setUploadFailures([]);
  }

  const result = job.data?.status === "done" ? job.data.result : null;
  const shown = result && vOverride ? { ...result, verdict: vOverride } : result;
  const started = !!jobId || analyze.isPending;
  const badgeKey = analyze.isPending ? "running" : job.data?.status;
  const activeBatchJob = batchId && jobId ? batchJobs.data?.find((j) => j.id === jobId) ?? null : null;
  const activeFileName = file?.name ?? activeBatchJob?.filename ?? null;
  const inGallery = !!batchId && !jobId;

  const infoNode = (
    <>
      <div className="card">
        <div className="side-h">Образец<span className="ann">{shown?.mode === "panorama" ? "панорама" : "крупный план"}</span></div>
        <div className="side-b"><div className="meta-rows">
          <div className="kv"><span className="k">Файл</span><span className="v">{activeFileName ?? "—"}</span></div>
          {shown?.size ? <div className="kv"><span className="k">Размер</span><span className="v">{shown.size[0]}×{shown.size[1]}</span></div> : null}
        </div></div>
      </div>
      {shown ? <VerdictPanel result={shown} /> : null}
      {shown && jobId ? (
        <a className="btn ghost" href={reportUrl(jobId)} target="_blank" rel="noopener noreferrer">
          <IconDownload /> Скачать протокол (PDF)
        </a>
      ) : null}
      {shown && jobId ? (
        <button
          type="button"
          className="btn ghost"
          onClick={() => downloadCsv(`shlif-${jobId}.csv`, jobsToCsv([job.data!]))}
        >
          <IconDownload /> Скачать CSV
        </button>
      ) : null}
      {shown?.mode === "panorama" && jobId ? <PanoramaZoomModal jobId={jobId} /> : null}
    </>
  );

  if (!batchId && !started) {
    return (
      <>
        <Welcome onFiles={handleFiles} />
        <div className="theme-float"><ThemeToggle /></div>
      </>
    );
  }

  if (inGallery) {
    return (
      <main className="app-main">
        <header className="topbar" style={{ flexWrap: "wrap" }}>
          <div className="logo" aria-hidden="true">🚀</div>
          <div><div className="crumb">DATA FORCE · классификация руд</div><h1>Скажи мне кто твой шлиф</h1></div>
          <div className="grow" />
          <ThemeToggle />
        </header>
        <BatchGallery
          batchId={batchId!}
          jobs={batchJobs.data ?? []}
          onOpen={openBatchItem}
          onNewAnalysis={resetToUpload}
        />
        {uploadFailures.length ? (
          <div className="card" style={{ marginTop: 12 }}>
            <div className="card-b">
              {uploadFailures.map((f, i) => (
                <div key={i} className="kv"><span className="k">{f.filename}</span><span className="v">не загружен: {f.error}</span></div>
              ))}
            </div>
          </div>
        ) : null}
      </main>
    );
  }

  return (
    <main className="app-main">
      <header className="topbar" style={{ flexWrap: "wrap" }}>
        <div className="logo" aria-hidden="true">🚀</div>
        <div><div className="crumb">DATA FORCE · классификация руд</div><h1>Скажи мне кто твой шлиф</h1></div>
        <div className="grow" />
        {badgeKey && STATUS_LABELS[badgeKey] ? (
          <span className={`status-badge ${STATUS_LABELS[badgeKey][0]}`}><span className="bd" />{STATUS_LABELS[badgeKey][1]}</span>
        ) : null}
        {batchId ? (
          <button type="button" className="btn ghost sm" onClick={backToBatch}>← к партии</button>
        ) : null}
        <button type="button" className="btn ghost sm" onClick={resetToUpload}>
          <IconUpload className="ico-sm" /> Новый анализ
        </button>
        <ThemeToggle />
      </header>

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
    </main>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={null}>
      <Home />
    </Suspense>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npm run build`
Expected: PASS — this also resolves the Task 9/10 red build, since `page.tsx` now calls
`<Welcome onFiles={handleFiles} />` and imports `BatchGallery`/`STATUS_LABELS`.

- [ ] **Step 3: Run the full frontend suite**

Run: `cd frontend && npm test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat(frontend): wire batch upload, gallery, and CSV export into the main page"
```

---

## Task 12: End-to-end verification

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Full backend suite**

Run: `cd backend && .venv/bin/pytest -q`
Expected: PASS, 0 failures.

- [ ] **Step 2: Full frontend suite + build**

Run: `cd frontend && npm test && npm run build`
Expected: PASS, 0 failures; build succeeds.

- [ ] **Step 3: Drive the real app end-to-end**

Invoke the `verify` skill (this project has one — it knows how to launch the backend/frontend dev
servers per `README.md`) to confirm, by actually driving the running app rather than only trusting
tests:

- Single-file drop still behaves exactly as before (no gallery, immediate workspace, PDF button
  works, and the new "Скачать CSV" button downloads a one-row CSV).
- Dropping 2+ files opens the batch gallery; cards progress from `queued` → `running` → `done`
  without blocking navigation; clicking a `done` card opens the same single-job workspace used for
  single-file analysis (Corrector/PanoramaZoomModal all work identically); "← к партии" returns to
  the gallery while other jobs keep progressing in the background.
- "Скачать CSV (партия)" downloads a CSV with one row per image in the batch, including a row for
  any image still `queued`/`running` (blank metrics) if triggered before everything finishes.
- Reloading the browser mid-batch (URL contains `?batch=<id>`) restores the gallery with the
  batch's current state fetched from `GET /api/jobs?batch_id=`, not from local UI state.

Report back any discrepancy from the above before considering this plan done.

- [ ] **Step 4: Final commit (if verification uncovered fixups)**

If Step 3 required any code changes, commit them with a message describing what verification caught
and how it was fixed. If Step 3 passed clean, there is nothing to commit here.
