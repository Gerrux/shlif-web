import numpy as np
from app.shlif import analyze_image
from app.pipeline import closeup, loader

def test_analyze_closeup_structure(tiny_rgb):
    cfg = loader.get_config()
    r = closeup.analyze_closeup(tiny_rgb, cfg)
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
    assert r["phase_map"].shape == tiny_rgb.shape[:2]
    assert set(np.unique(r["phase_map"])) <= {0, 1, 2}
    assert r["talc"].shape == tiny_rgb.shape[:2]
    assert r["intergrowth"].shape == tiny_rgb.shape[:2]
    assert set(np.unique(r["intergrowth"])) <= {0, 1, 2}
    assert r["superpixels"].shape == tiny_rgb.shape[:2]
    assert r["darkness"].shape == tiny_rgb.shape[:2]
    assert r["sort"] is None or set(r["sort"]["classes"]) <= {"ordinary", "hard", "talcose"}

def test_analyze_closeup_uses_talc_unet_when_available(tiny_rgb, monkeypatch):
    """When loader.load_talc_unet() has weights, analyze_closeup must use the
    U-Net mask (via talc_mask=) instead of the classical detect_talc seed."""
    cfg = loader.get_config()
    unet_mask = np.zeros(tiny_rgb.shape[:2], dtype=bool)
    unet_mask[:60, :60] = True

    monkeypatch.setattr(closeup.loader, "load_talc_unet", lambda: ("fake-model", "cpu"))
    monkeypatch.setattr(closeup, "talc_unet_mask",
                        lambda rgb, model, device, thr=None: unet_mask)

    def boom(*a, **k):
        raise AssertionError("classical detect_talc must not run when the U-Net is available")
    monkeypatch.setattr("app.shlif.analyze.detect_talc", boom)

    r = closeup.analyze_closeup(tiny_rgb, cfg)
    expected = analyze_image(tiny_rgb, cfg, talc_mask=unet_mask).masks["talc"]
    assert np.array_equal(r["talc"], expected)
