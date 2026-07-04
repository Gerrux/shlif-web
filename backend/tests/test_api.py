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
    up = c.post("/api/analyze",
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    assert up.status_code == 200
    jid = up.json()["job_id"]
    done = _poll(c, jid)
    assert done["status"] == "done"
    assert done["progress"] == 1.0
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
