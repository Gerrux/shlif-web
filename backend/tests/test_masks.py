import io
import cv2
import numpy as np
from PIL import Image
from app.pipeline import masks
from app.pipeline import loader

def test_phase_map_roundtrip():
    s = np.zeros((10, 10), bool); s[0:3, 0:3] = True
    m = np.zeros((10, 10), bool); m[5:8, 5:8] = True
    pm = masks.phase_label_map(s, m)
    assert pm.dtype == np.uint8 and set(np.unique(pm)) <= {0, 1, 2}
    su, mg, mx = masks.split_phase_map(pm)
    assert (su == s).all() and (mg == m).all() and (mx == ~(s | m)).all()

def test_verdict_from_masks_reacts_to_talc():
    cfg = loader.get_config()
    s = np.zeros((100, 100), bool); s[:10] = True
    m = np.zeros((100, 100), bool)
    mx = ~(s | m)
    no_talc = masks.verdict_from_masks_dict(s, m, mx, np.zeros((100,100), bool), cfg)
    lots = np.zeros((100, 100), bool); lots[50:] = True
    talcy = masks.verdict_from_masks_dict(s, m, mx & lots | mx, lots & mx, cfg)
    assert talcy["metrics"]["talc_frac"] > no_talc["metrics"]["talc_frac"]

def test_superpixel_and_darkness_maps(tiny_rgb):
    sp = masks.build_superpixel_map(tiny_rgb, n_segments=120)
    assert sp.dtype == np.uint16 and sp.shape == tiny_rgb.shape[:2] and sp.max() >= 50
    dk = masks.build_darkness_map(tiny_rgb)
    assert dk.dtype == np.uint8 and dk.shape == tiny_rgb.shape[:2]

def test_png_gray_roundtrip():
    a = (np.arange(256, dtype=np.uint8).reshape(16, 16))
    assert (masks.decode_png_gray(masks.encode_png_gray(a)) == a).all()

def test_encode_png_label_rgb_roundtrip_survives_8bit_canvas():
    """The browser decodes this PNG via HTML canvas getImageData, which is 8-bit/channel.
    Prove the R/G byte-pack survives an independent PIL decode (== what canvas would see)
    for label ids that exercise both the high and low byte."""
    labels = np.array([
        [0, 1, 255, 256],
        [257, 600, 65535, 42],
    ], dtype=np.uint16)
    png = masks.encode_png_label_rgb(labels)
    rgb = np.asarray(Image.open(io.BytesIO(png)))
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 4, 3)
    decoded = (rgb[..., 0].astype(np.uint32) << 8) | rgb[..., 1].astype(np.uint32)
    assert (decoded == labels).all()

def test_fit_max_side_no_op_when_within_budget():
    arr = np.zeros((100, 200, 3), np.uint8)
    out = masks.fit_max_side(arr, 2400, cv2.INTER_AREA)
    assert out.shape == arr.shape

def test_fit_max_side_downscales_preserving_aspect():
    arr = np.zeros((4000, 2000, 3), np.uint8)
    out = masks.fit_max_side(arr, 2000, cv2.INTER_AREA)
    assert max(out.shape[:2]) == 2000
    assert out.shape[0] == 2 * out.shape[1]  # aspect ratio kept (4000:2000 == 2:1)

def test_persist_editor_artifacts_writes_all_files(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    r = {
        "phase_map": np.zeros((8, 8), np.uint8),
        "talc": np.zeros((8, 8), bool),
        "superpixels": np.zeros((8, 8), np.uint16),
        "darkness": np.zeros((8, 8), np.uint8),
        "confidence": np.ones((8, 8), np.float32),
    }
    masks.persist_editor_artifacts("jobx", r)
    assert (tmp_path / "masks" / "jobx" / "phases.png").exists()
    assert (tmp_path / "masks" / "jobx" / "talc.png").exists()
    assert (tmp_path / "maps" / "jobx" / "superpixels.png").exists()
    assert (tmp_path / "maps" / "jobx" / "darkness.png").exists()
    assert (tmp_path / "maps" / "jobx" / "confidence.png").exists()
