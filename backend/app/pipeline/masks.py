from __future__ import annotations
import io, cv2, numpy as np
from PIL import Image
from skimage.segmentation import slic
from app.core import paths
from app.shlif import phases
from app.shlif.analyze import verdict_from_masks

def phase_label_map(sulfide: np.ndarray, magnetite: np.ndarray) -> np.ndarray:
    pm = np.zeros(sulfide.shape, np.uint8)          # 0 = matrix
    pm[magnetite.astype(bool)] = phases.MAGNETITE   # 1
    pm[sulfide.astype(bool)] = phases.SULFIDE       # 2 (sulfide wins overlap)
    return pm

def split_phase_map(pm: np.ndarray):
    return pm == phases.SULFIDE, pm == phases.MAGNETITE, pm == phases.MATRIX

def verdict_from_masks_dict(sulfide, magnetite, matrix, talc, cfg, dist_px: int = 12) -> dict:
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px)
    return {"ore_class": v["ore_class"], "text": v["text"], "metrics": v["metrics"]}

def build_superpixel_map(rgb: np.ndarray, n_segments: int = 600) -> np.ndarray:
    seg = slic(rgb, n_segments=n_segments, compactness=12, start_label=0)
    return seg.astype(np.uint16)

def build_darkness_map(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def encode_png_gray(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr.astype(np.uint8))
    if not ok: raise RuntimeError("png encode failed")
    return buf.tobytes()

def decode_png_gray(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None: raise ValueError("png decode failed")
    return arr

def encode_png_label_rgb(labels: np.ndarray) -> bytes:
    """Pack a uint16 label map into an 8-bit RGB PNG (R=high byte, G=low byte) so it
    survives HTML-canvas getImageData (8-bit/channel) losslessly. Decode: (R<<8)|G."""
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (labels.astype(np.uint16) >> 8) & 0xFF
    rgb[..., 1] = labels.astype(np.uint16) & 0xFF
    buf = io.BytesIO(); Image.fromarray(rgb, "RGB").save(buf, "PNG"); return buf.getvalue()


EDIT_MAX_SIDE = 2400  # editing/display working resolution: the already-proven
# close-up budget (previously inlined in api/analyze.py as
# `im.thumbnail((2400, 2400))`); applied uniformly so panorama editing is
# exactly as responsive as close-up editing is today.


def fit_max_side(arr: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    """Resize `arr` (image or integer label map) so its longer side is
    `max_side`, preserving aspect ratio. No-op if already within budget."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_side:
        return arr
    s = max_side / float(max(h, w))
    return cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interpolation)


def persist_editor_artifacts(jid: str, r: dict) -> None:
    """Write the phase/talc masks + superpixel/darkness/confidence maps a
    finished job needs for the Corrector editor. Shared by the closeup and
    panorama result assembly so both produce identically-shaped, equally
    editable artifacts."""
    md = paths.masks_dir(jid)
    mp = paths.maps_dir(jid)
    (md / "phases.png").write_bytes(encode_png_gray(r["phase_map"]))
    (md / "talc.png").write_bytes(encode_png_gray((r["talc"].astype(np.uint8) * 255)))
    (mp / "superpixels.png").write_bytes(encode_png_label_rgb(r["superpixels"]))
    (mp / "darkness.png").write_bytes(encode_png_gray(r["darkness"]))
    (mp / "confidence.png").write_bytes(
        encode_png_gray(np.clip(r["confidence"] * 255.0, 0, 255).astype(np.uint8)))
