"""_run_panorama surfaces the ensemble uncertainty (undetermined_fraction +
low_conf_zones) the same way closeup.py already does, catching the classical
segmenter's exposure-driven magnetite<->sulfide flip instead of silently
mislabeling it."""
import numpy as np
import pytest
from PIL import Image

from app.pipeline import panorama, loader

CFG = loader.get_config()


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_reports_uncertainty(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)
    img[100:500, 100:500] = 220     # confident sulfide block
    img[700:1100, 700:1100] = 170   # borderline magnetite block -> disputed
                                     # under the ensemble's mild gamma/gain jitter
    p = tmp_path / "pano.png"
    Image.fromarray(img).save(p, "PNG")   # PNG, not JPEG: JPEG's block compression
                                            # smooths the flat blocks enough to hide
                                            # the dispute at these exact values

    r = panorama.analyze_panorama(str(p), CFG, "unctest")

    metrics = r["verdict"]["metrics"]
    assert "undetermined_fraction" in metrics
    assert metrics["undetermined_fraction"] > 0.0

    zones = r["low_conf_zones"]
    assert isinstance(zones, list)
    assert len(zones) >= 1
    assert any({"магнетит", "сульфид"} <= {z["phase_a"], z["phase_b"]} for z in zones)
    for z in zones:
        assert set(z) == {"bbox", "area", "phase_a", "phase_b"}
        assert len(z["bbox"]) == 4


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_uncertainty_does_not_mutate_shared_config(tmp_path):
    before = loader.get_config().talc.detect_dark_frac
    rng = np.random.default_rng(2)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)
    img[100:500, 100:500] = 220
    p = tmp_path / "pano.png"
    Image.fromarray(img).save(p, "PNG")
    panorama.analyze_panorama(str(p), loader.get_config(), "unccfgtest")
    assert loader.get_config().talc.detect_dark_frac == before
