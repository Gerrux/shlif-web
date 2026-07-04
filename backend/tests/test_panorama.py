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
    # same metrics keys close-up produces, computed over the whole image
    for key in ("sulfide_frac", "magnetite_frac", "matrix_frac", "talc_frac",
                "normal_share", "fine_share", "confidence", "talc_share_est"):
        assert key in r["verdict"]["metrics"]
    # same top-level sort-card shape as closeup, not buried in metrics
    assert set(r["sort"]["classes"]) <= {"ordinary", "hard", "talcose"}
    assert r["sort"]["top"] in r["sort"]["classes"]
    assert r["size"][0] > 0 and r["size"][1] > 0
    assert r["native_size"][0] >= r["size"][0] and r["native_size"][1] >= r["size"][1]
    assert isinstance(r["low_conf_zones"], list)


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_does_not_mutate_shared_config(tmp_path):
    before = loader.get_config().talc.detect_dark_frac
    img = (np.random.default_rng(2).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    panorama.analyze_panorama(str(p), loader.get_config(), "cfgtest")
    assert loader.get_config().talc.detect_dark_frac == before


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_uses_talc_unet_when_available(tmp_path, monkeypatch):
    """When loader.load_talc_unet() has weights, _run_panorama's per-tile talc
    decision must come from the U-Net (mask & matrix), not the classical
    detect_talc. Checks the gate directly (was the U-Net actually called, and
    did the classical detector never run) rather than a downstream metric:
    the reported verdict's talc_frac comes from _assemble_masks, which is
    classical-only regardless of the talc U-Net's availability (see this
    module's docstring) — so a numeric assertion on r["talc_frac"] wouldn't
    actually exercise this gate."""
    img = (np.random.default_rng(3).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()

    calls = []
    def fake_talc_unet(rgb, model, device, thr=None):
        calls.append(1)
        return np.ones(rgb.shape[:2], bool)
    monkeypatch.setattr(panorama.loader, "load_talc_unet", lambda: ("fake-model", "cpu"))
    monkeypatch.setattr(panorama, "talc_unet_mask", fake_talc_unet)

    def boom(*a, **k):
        raise AssertionError("classical detect_talc must not run when the U-Net is available")
    monkeypatch.setattr(panorama, "detect_talc", boom)

    r = panorama.analyze_panorama(str(p), cfg, "unettest")
    assert r["mode"] == "panorama"
    assert len(calls) > 0  # the U-Net talc path was actually exercised, not skipped
