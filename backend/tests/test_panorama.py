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
    """When loader.load_talc_unet() has weights, every tile's talc mask must come
    from the U-Net (mask & matrix), not the classical detect_talc."""
    img = (np.random.default_rng(3).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"; Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()

    monkeypatch.setattr(panorama.loader, "load_talc_unet", lambda: ("fake-model", "cpu"))
    monkeypatch.setattr(panorama, "talc_unet_mask",
                        lambda rgb, model, device, thr=None: np.ones(rgb.shape[:2], bool))

    def boom(*a, **k):
        raise AssertionError("classical detect_talc must not run when the U-Net is available")
    monkeypatch.setattr(panorama, "detect_talc", boom)

    r = panorama.analyze_panorama(str(p), cfg, "unettest")
    # mask is all-True, so per tile talc == matrix exactly -> talc_frac == 0.5
    assert r["talc_frac"] == pytest.approx(0.5)
