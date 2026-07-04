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
    assert done["result"]["mode"] == "closeup"  # tiny_rgb (256x256) is well under direct_max_pixels
    assert done["result"]["verdict"]["ore_class"] in {"ordinary","hard","talcose","review"}

    # layers + maps are fetchable
    assert c.get(f"/api/masks/{jid}/phases.png").status_code == 200
    assert c.get(f"/api/masks/{jid}/intergrowth.png").status_code == 200
    assert c.get(f"/api/maps/{jid}/superpixels.png").status_code == 200
    assert c.get(f"/api/maps/{jid}/darkness.png").status_code == 200

    # intergrowth.png must match phases.png's resolution exactly
    phases_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/phases.png").content)))
    ig_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/intergrowth.png").content)))
    assert ig_arr.shape == phases_arr.shape

    # edit: mark everything talc → verdict recomputes to talcose
    h, w = tiny_rgb.shape[:2]
    all_talc = np.full((h, w), 255, np.uint8)
    phases_png = c.get(f"/api/masks/{jid}/phases.png").content
    r = c.post(f"/api/masks/{jid}",
               files={"talc": ("talc.png", _png_bytes(all_talc), "image/png"),
                      "phases": ("phases.png", phases_png, "image/png")})
    assert r.status_code == 200
    assert r.json()["ore_class"] == "talcose"
    assert "intergrowth" not in r.json()  # popped server-side, must not leak into the Verdict JSON

    # resolution still matches after the recompute
    ig_arr2 = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/intergrowth.png").content)))
    assert ig_arr2.shape == phases_arr.shape


def test_edit_resizes_intergrowth_back_to_editor_resolution_when_native_differs(tiny_rgb):
    """save_masks computes intergrowth at whatever resolution the uploaded
    phases/talc arrive at, upscaled to native_size for accurate verdict metrics
    (mirroring how panorama jobs are edited) — but must persist intergrowth.png
    back down at the SAME resolution as phases.png/talc.png (the editor
    resolution), not native. Forges a panorama-shaped native_size on an
    ordinary closeup job so the mismatch path is exercised without needing an
    actual 50+ megapixel image."""
    c = TestClient(app)
    up = c.post("/api/analyze", files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    _poll(c, jid)

    from app.runtime import get_runtime
    job = get_runtime().store.get(jid)
    result = dict(job.result)
    result["native_size"] = [tiny_rgb.shape[1] * 3, tiny_rgb.shape[0] * 3]  # pretend native >> editor
    get_runtime().store.set_result(jid, result)

    h, w = tiny_rgb.shape[:2]
    all_talc = np.full((h, w), 255, np.uint8)
    phases_png = c.get(f"/api/masks/{jid}/phases.png").content
    r = c.post(f"/api/masks/{jid}",
               files={"talc": ("talc.png", _png_bytes(all_talc), "image/png"),
                      "phases": ("phases.png", phases_png, "image/png")})
    assert r.status_code == 200

    phases_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/phases.png").content)))
    ig_arr = np.asarray(Image.open(io.BytesIO(c.get(f"/api/masks/{jid}/intergrowth.png").content)))
    assert ig_arr.shape == phases_arr.shape == (h, w)
