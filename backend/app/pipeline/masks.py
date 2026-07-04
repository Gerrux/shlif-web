from __future__ import annotations
import cv2, numpy as np
from skimage.segmentation import slic
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

def encode_png_u16(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr.astype(np.uint16))
    if not ok: raise RuntimeError("png u16 encode failed")
    return buf.tobytes()
