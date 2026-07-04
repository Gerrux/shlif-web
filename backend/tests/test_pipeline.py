import numpy as np
from app.pipeline import closeup, loader

def test_analyze_closeup_structure(tiny_rgb):
    cfg = loader.get_config()
    r = closeup.analyze_closeup(tiny_rgb, cfg)
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
    assert r["phase_map"].shape == tiny_rgb.shape[:2]
    assert set(np.unique(r["phase_map"])) <= {0, 1, 2}
    assert r["talc"].shape == tiny_rgb.shape[:2]
    assert r["superpixels"].shape == tiny_rgb.shape[:2]
    assert r["darkness"].shape == tiny_rgb.shape[:2]
    assert r["sort"] is None or set(r["sort"]["classes"]) <= {"ordinary", "hard", "talcose"}
