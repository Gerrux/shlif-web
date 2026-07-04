"""When a job's stored result carries a native_size larger than the edited
PNGs (the panorama case — edited at EDIT_MAX_SIDE, analyzed at native tiled
resolution), POST /masks must upscale (nearest-neighbor) before recomputing
the verdict, so the corrected verdict is computed at native resolution just
like the original one was.

fine_share is the property that actually exposes a missing upscale: it
classifies sulfide as "fine" within an absolute dist_px=12 of magnetite. At
the small edited resolution below, the whole 10px-wide sulfide band sits
within 12px of magnetite (fine_share ~= 1.0). Upscaled 10x to native
resolution, the same band is 100px wide and only its first ~12 native
pixels are within dist_px=12 of magnetite (fine_share should drop sharply).
If save_masks recomputes at the small (un-upscaled) resolution, fine_share
stays ~1.0 and the test fails.
"""
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from app.runtime import get_runtime


def _png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def test_save_masks_upscales_to_native_size_before_recompute():
    c = TestClient(app)
    jid = get_runtime().store.create("panorama")
    get_runtime().store.set_result(jid, {"native_size": [200, 200]})  # 10x the edited size below

    w = h = 20
    pm = np.zeros((h, w), np.uint8)
    pm[:, 8:10] = 1   # magnetite band, 2px wide
    pm[:, 10:20] = 2  # sulfide band, 10px wide, right next to magnetite
    talc = np.zeros((h, w), np.uint8)

    r = c.post(f"/api/masks/{jid}",
               files={"phases": ("phases.png", _png_bytes(pm), "image/png"),
                      "talc": ("talc.png", _png_bytes(talc), "image/png")})
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    # at native (upscaled) resolution only ~12 of the 100 native sulfide
    # columns are within dist_px=12 of magnetite -> fine_share well under 1.0
    assert metrics["fine_share"] < 0.3

    # the saved edit on disk stays at editing resolution — only the
    # in-memory recompute upscales
    from app.core import paths
    saved = np.asarray(Image.open(paths.masks_dir(jid) / "phases.png"))
    assert saved.shape == (h, w)
