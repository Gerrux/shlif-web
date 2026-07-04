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
    decision (which drives the display overlay + ore-density weighting) must
    come from the U-Net, not the classical detect_talc.

    This checks only that path, not a ban on detect_talc anywhere in
    analyze_panorama: _assemble_masks (which produces the *reported* verdict)
    is classical-only regardless of U-Net availability (see this module's
    docstring — wiring U-Net into _assemble_masks too, mirroring
    shlif.analyze.analyze_image's ore_mask pattern, is a reasonable follow-up
    but is new, unreviewed work, not something to fold into this merge) — so
    it legitimately still calls detect_talc, and a blanket ban would fail for
    a reason unrelated to what this test is actually checking."""
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

    r = panorama.analyze_panorama(str(p), cfg, "unettest")
    assert r["mode"] == "panorama"
    assert len(calls) > 0  # the U-Net talc path was actually exercised, not skipped


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_survives_tile_pyramid_failure(tmp_path, monkeypatch):
    """A broken tile pyramid must never take down the whole analysis — it's
    a display enhancement, not part of the verdict."""
    img = (np.random.default_rng(4).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()

    def boom(arr, jid):
        raise RuntimeError("disk full")
    monkeypatch.setattr(panorama.tiles, "build_pyramid", boom)

    r = panorama.analyze_panorama(str(p), cfg, "pyramidfailtest")
    assert r["mode"] == "panorama"
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
