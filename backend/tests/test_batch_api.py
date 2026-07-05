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


def test_analyze_result_includes_reproducibility_params(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", files={"image": ("p.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    params = done["result"]["params"]
    assert params["mode"] == "closeup"
    assert "models" in params
    assert "gpu" in params


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
