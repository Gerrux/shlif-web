"""analyze_image surfaces the dispersed-dark-gray talc-share proxy in its metrics."""
import numpy as np

from app.shlif.analyze import analyze_image
from app.pipeline import loader

CFG = loader.get_config()


def test_metrics_include_talc_share_est():
    rng = np.random.default_rng(7)
    rgb = rng.integers(90, 150, (256, 256, 3), dtype=np.uint8)  # grey matrix-ish
    res = analyze_image(rgb, CFG, detect_talc_flag=False)
    assert "talc_share_est" in res.metrics
    v = res.metrics["talc_share_est"]
    assert isinstance(v, float) and 0.0 <= v <= 1.0
