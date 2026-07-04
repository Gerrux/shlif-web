"""When a job's stored result carries a native_size larger than the edited
PNGs (the panorama case — edited at EDIT_MAX_SIDE, analyzed at native tiled
resolution), POST /masks must upscale (nearest-neighbor) before recomputing
the verdict, so the corrected verdict is computed at native resolution just
like the original one was.

This checks the resize MECHANISM directly (the shape of the array actually
fed into split_phase_map/verdict_from_masks_dict) rather than an emergent
numeric side effect of whatever the current phase-composition rule happens
to be: master's grain-size rewrite of _liberation_split made fine_share
(the metric an earlier version of this test relied on) approximately
scale-invariant by design — its lib_px threshold is itself a fraction of
image size, so a small synthetic fixture can give the same fine_share
whether or not the upscale actually ran, silently defeating a metric-based
check. Verifying the mechanism directly makes this test immune to whatever
verdict_from_masks does internally, now or in the future.
"""
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from app.runtime import get_runtime
from app.pipeline import masks as M
from app.api import masks as masks_api
from app.core import paths


def _png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def test_save_masks_upscales_to_native_size_before_recompute(monkeypatch):
    c = TestClient(app)
    jid = get_runtime().store.create("panorama")
    get_runtime().store.set_result(jid, {"native_size": [200, 200]})  # 10x the edited size below

    w = h = 20
    pm = np.zeros((h, w), np.uint8)
    pm[:, 8:10] = 1   # magnetite band
    pm[:, 10:20] = 2  # sulfide band
    talc = np.zeros((h, w), np.uint8)

    captured = {}
    real_split = M.split_phase_map
    def spy_split(pm_arr):
        captured["shape"] = pm_arr.shape
        return real_split(pm_arr)
    monkeypatch.setattr(masks_api.M, "split_phase_map", spy_split)

    r = c.post(f"/api/masks/{jid}",
               files={"phases": ("phases.png", _png_bytes(pm), "image/png"),
                      "talc": ("talc.png", _png_bytes(talc), "image/png")})
    assert r.status_code == 200

    # the array actually fed into verdict computation was upscaled to
    # native_size (200x200), not left at the uploaded editing size (20x20)
    assert captured["shape"] == (200, 200)

    # the saved edit on disk stays at editing resolution — only the
    # in-memory recompute upscales
    saved = np.asarray(Image.open(paths.masks_dir(jid) / "phases.png"))
    assert saved.shape == (h, w)


def test_save_masks_is_a_no_op_resize_when_native_size_absent(monkeypatch):
    """Close-up jobs never set native_size — split_phase_map must receive the
    uploaded array's own shape unchanged, proving the upscale branch is a
    true no-op there, not just skipped by accident."""
    c = TestClient(app)
    jid = get_runtime().store.create("closeup")
    get_runtime().store.set_result(jid, {})  # no native_size key at all

    w = h = 20
    pm = np.zeros((h, w), np.uint8)
    talc = np.zeros((h, w), np.uint8)

    captured = {}
    real_split = M.split_phase_map
    def spy_split(pm_arr):
        captured["shape"] = pm_arr.shape
        return real_split(pm_arr)
    monkeypatch.setattr(masks_api.M, "split_phase_map", spy_split)

    r = c.post(f"/api/masks/{jid}",
               files={"phases": ("phases.png", _png_bytes(pm), "image/png"),
                      "talc": ("talc.png", _png_bytes(talc), "image/png")})
    assert r.status_code == 200
    assert captured["shape"] == (h, w)
