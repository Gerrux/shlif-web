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
