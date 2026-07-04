"""The close-up analyze result carries the ensemble-uncertainty signal and the
confidence map is fetchable as a layer."""
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from main import app


def _png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def _poll(c, jid):
    for _ in range(100):
        r = c.get(f"/api/jobs/{jid}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_closeup_result_has_uncertainty(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze",
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["status"] == "done"
    res = done["result"]

    frac = res["verdict"]["metrics"]["undetermined_fraction"]
    assert 0.0 <= frac <= 1.0
    assert isinstance(res["low_conf_zones"], list)
    assert c.get(f"/api/maps/{jid}/confidence.png").status_code == 200
