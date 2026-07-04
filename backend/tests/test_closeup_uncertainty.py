"""analyze_closeup surfaces the ensemble uncertainty (fraction + map + zones)."""
import numpy as np

from app.pipeline import closeup, loader

CFG = loader.get_config()


def test_analyze_closeup_reports_uncertainty():
    rgb = np.full((256, 256, 3), 10, np.uint8)
    rgb[80:176, 80:176] = 245
    r = closeup.analyze_closeup(rgb, CFG)

    metrics = r["verdict"]["metrics"]
    assert "undetermined_fraction" in metrics
    assert 0.0 <= metrics["undetermined_fraction"] <= 1.0

    assert "confidence" in r
    assert r["confidence"].shape[:2] == (256, 256)   # full-res confidence layer

    assert isinstance(r["low_conf_zones"], list)
