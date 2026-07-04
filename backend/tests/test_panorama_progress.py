"""analyze_panorama reports progress through on_progress across both tile loops
(_assemble_masks, _run_panorama) and the tail stages."""
import numpy as np
import pytest
from PIL import Image

from app.pipeline import panorama, loader


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_reports_progress(tmp_path):
    img = (np.random.default_rng(5).integers(8, 30, (1200, 2400, 3))).astype(np.uint8)
    img[100:400, 100:400] = 210
    p = tmp_path / "pano.jpg"
    Image.fromarray(img).save(p, "JPEG")
    cfg = loader.get_config()
    calls = []

    r = panorama.analyze_panorama(str(p), cfg, "progresstest",
                                   on_progress=lambda pr, msg: calls.append((pr, msg)))

    assert r["mode"] == "panorama"
    assert len(calls) >= 5
    progresses = [pr for pr, _ in calls]
    assert progresses == sorted(progresses)
    assert all(0.0 <= pr <= 1.0 for pr in progresses)
    messages = " ".join(msg for _, msg in calls if msg)
    assert "сборка масок" in messages
    assert "сегментация тайлов" in messages
