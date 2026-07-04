import io
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
