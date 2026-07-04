"""When loader.load_ore_unet() returns a model bundle, _run_panorama must route
the ore/matrix decision through ore_unet_mask instead of the classical
segment_phases split. When it returns None (as in this dev sandbox, where
torch/segmentation-models-pytorch aren't installed), the classical path must
keep working exactly as before -- covered by the existing test_panorama.py."""
import numpy as np
import pytest
from PIL import Image

from app.pipeline import panorama, loader

CFG = loader.get_config()


@pytest.mark.skipif(loader.load_classifier() is None, reason="needs models/classifier.pkl")
def test_panorama_routes_ore_gate_through_unet_when_available(tmp_path, monkeypatch):
    calls = []

    def fake_mask(rgb, model, device, tile=512):
        calls.append(rgb.shape[:2])
        return np.zeros(rgb.shape[:2], bool)   # deterministic: "nothing is ore"

    monkeypatch.setattr(panorama, "ore_unet_mask", fake_mask)
    monkeypatch.setattr(panorama.loader, "load_ore_unet", lambda: (object(), "cpu"))

    rng = np.random.default_rng(4)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)
    img[100:500, 100:500] = 220   # with the REAL classical segmenter this tile
                                   # is ore-gated (n_ore == 1, verified during
                                   # planning) -- proving the fake mask's
                                   # "nothing is ore" answer actually won means
                                   # the wiring, not the classical path, decided.
    p = tmp_path / "pano.png"
    Image.fromarray(img).save(p, "PNG")

    r = panorama.analyze_panorama(str(p), CFG, "unetwiring")

    assert calls, "ore_unet_mask must be invoked when load_ore_unet() returns a model"
    assert r["n_ore"] == 0
    assert r["ore_source"] == "unet"
