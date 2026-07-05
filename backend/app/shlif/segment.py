"""Three-phase reflectance segmentation: sulfide / magnetite / matrix.

Method (all thresholds adaptive per image, tunable via config):
  * Work in CIE-Lab. L = reflectance, chroma = sqrt(a^2+b^2), b = warmth.
  * Sulfide is the *brightest* phase in both bright close-ups and dark panoramas
    -> take the top band of a 3-level Otsu on L.
  * Magnetite is mid-reflectance and *neutral* (low chroma) -> mid L band with
    small chroma. No hue/warmth gate (see segment_phases for why).
  * Everything else is matrix.

Returns an integer label map using the constants in :mod:`shlif.phases`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import threshold_multiotsu, threshold_otsu

from . import phases


@dataclass
class SegResult:
    labels: np.ndarray            # HxW int: MATRIX / MAGNETITE / SULFIDE
    sulfide: np.ndarray           # bool mask
    magnetite: np.ndarray         # bool mask
    fractions: dict               # area fractions of each phase (of total image)


def _levels(L: np.ndarray, bright_pct: float) -> tuple[float, float]:
    """Two thresholds (dark|mid, mid|bright) from a 3-class Otsu on L, with a
    robust fallback for low-variance (near-empty) tiles."""
    finite = L[np.isfinite(L)]
    if finite.size == 0 or np.ptp(finite) < 1.0:
        hi = np.percentile(finite, bright_pct) if finite.size else 1e9
        return hi, hi
    try:
        t = threshold_multiotsu(finite, classes=3)
        return float(t[0]), float(t[1])
    except ValueError:
        try:
            t = threshold_otsu(finite)
        except ValueError:
            t = float(np.percentile(finite, bright_pct))
        return float(t), float(t)


def _clean(mask: np.ndarray, min_area: int, close_r: int) -> np.ndarray:
    """Morphological closing + small-object/hole removal (cv2, version-stable)."""
    m = mask.astype(np.uint8)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_area > 0:
        m = _drop_small(m, min_area)                       # remove specks
        m = 1 - _drop_small(1 - m, min_area)               # fill pinholes
    return m.astype(bool)


def _drop_small(m: np.ndarray, min_area: int) -> np.ndarray:
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[0] = False
    for i in range(1, n):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[lbl].astype(np.uint8)


def segment_phases(rgb: np.ndarray, cfg) -> SegResult:
    """Segment sulfide / magnetite / matrix from a preprocessed RGB image."""
    lab = rgb2lab(rgb)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a * a + b * b)

    dark_t, bright_t = _levels(L, float(cfg.bright_percentile))

    # sulfide: brightest band, above an absolute floor
    sulfide = (L >= bright_t) & (L >= float(cfg.sulfide_min_L))

    # magnetite: mid reflectance, neutral (low chroma). No hue/warmth gate --
    # a `not_olive` filter here used to assume olive hue = matrix, but that's
    # backwards for this material (olive = sulfide; real magnetite skews cool,
    # negative Lab b) and an absolute b-channel floor doesn't transfer across
    # images with different lighting/white-balance anyway. Genuinely ambiguous
    # magnetite/sulfide pixels are caught by uncertainty.py's perturbation
    # ensemble instead of forced here.
    mid = (L >= dark_t) & ~sulfide
    neutral = chroma <= float(cfg.chroma_max)
    magnetite = mid & neutral

    min_area = int(cfg.min_ore_area_px)
    close_r = int(cfg.close_radius)
    sulfide = _clean(sulfide, min_area, close_r)
    magnetite = _clean(magnetite, min_area, close_r) & ~sulfide

    labels = np.full(L.shape, phases.MATRIX, dtype=np.uint8)
    labels[magnetite] = phases.MAGNETITE
    labels[sulfide] = phases.SULFIDE

    total = L.size
    fractions = {
        "sulfide": float(sulfide.sum()) / total,
        "magnetite": float(magnetite.sum()) / total,
        "matrix": float((labels == phases.MATRIX).sum()) / total,
    }
    return SegResult(labels=labels, sulfide=sulfide, magnetite=magnetite, fractions=fractions)
