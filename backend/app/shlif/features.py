"""Texture / morphology features for intergrowth-type classification.

Weak supervision: we have only image-level labels (no pixel masks for intergrowth
type), so we describe each image by the *structure of its sulfide phase* and its
overall texture, then let a classifier learn ordinary (solid) vs hard (fine net).

Every image is resized to a fixed working long-side before extraction so that
the pixel scale is comparable across the dataset's mixed resolutions
(2272..6240 px). Physical scale (5x/10x magnification) still varies where it is
not encoded in the filename — see ``magnification_from_name``.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import skeletonize

from .preprocess import preprocess
from .segment import segment_phases

WORK_LONG_SIDE = 1600
REF_MAG = 10.0
_GLCM_DISTANCES = (1, 5)
_GRANULO_RADII = (1, 2, 4, 8, 16)
_LBP_P, _LBP_R = 8, 1.0

# magnification token: a small number immediately followed by x/х, not part of a
# longer id. Only plausible objective magnifications are accepted.
_MAG_RE = re.compile(r"(?<!\d)(\d{1,3})\s*[xх](?![\w])", re.IGNORECASE)
_PLAUSIBLE_MAG = {4, 5, 10, 16, 20, 25, 40, 50, 63, 100}


def magnification_from_name(name: str) -> float | None:
    """Parse a plausible '5x' / '10х' / '20x' objective magnification, else None.

    Rejects garbage matches (sample ids, resolutions) by requiring the value to be
    a known objective magnification. Coverage is low (~50 of 1178 files).
    """
    for m in _MAG_RE.finditer(name):
        val = int(m.group(1))
        if val in _PLAUSIBLE_MAG:
            return float(val)
    return None


def scale_normalize(rgb: np.ndarray, magnification: float | None) -> np.ndarray:
    """Bring every image to a comparable scale before texture features.

    1. Where magnification is known, resample so µm/px matches a 10x reference
       (5x -> upsample, 20x -> downsample), so grains occupy comparable pixels.
    2. Cap the working long side for all images (uniform pixel grid) — this is the
       load-bearing step, since it removes the 2272/4160 resolution variance that
       otherwise leaks class information.
    """
    h, w = rgb.shape[:2]
    if magnification:
        s = REF_MAG / magnification
        s = min(s, 1.5)  # don't over-upsample low-mag images
        if abs(s - 1.0) > 0.05:
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            w, h = int(w * s), int(h * s)
            rgb = cv2.resize(rgb, (w, h), interpolation=interp)
    long_side = max(h, w)
    if long_side > WORK_LONG_SIDE:
        s2 = WORK_LONG_SIDE / long_side
        rgb = cv2.resize(rgb, (int(w * s2), int(h * s2)), interpolation=cv2.INTER_AREA)
    return rgb


def _cc_features(mask: np.ndarray) -> dict:
    area = float(mask.sum())
    total = float(mask.size)
    if area < 1:
        return dict(sulf_frac=0.0, cc_per_mpx=0.0, area_med=0.0, area_p90=0.0,
                    solidity=0.0, extent=0.0, compactness=0.0, skel_ratio=0.0)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas, solid, ext, comp = [], [], [], []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 4:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        comp_i = (lbl == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        peri = sum(cv2.arcLength(c, True) for c in cnts) + 1e-6
        hull = cv2.convexHull(np.vstack(cnts)) if cnts else None
        hull_a = cv2.contourArea(hull) if hull is not None else a
        areas.append(a)
        solid.append(a / (hull_a + 1e-6))
        ext.append(a / (w * h + 1e-6))
        comp.append(peri * peri / (a + 1e-6))  # perimeter^2/area — lacy = high
    if not areas:
        areas = [0]
    skel = skeletonize(mask)
    return dict(
        sulf_frac=area / total,
        cc_per_mpx=(len(areas) / total) * 1e6,
        area_med=float(np.median(areas)) / total,
        area_p90=float(np.percentile(areas, 90)) / total,
        solidity=float(np.mean(solid)) if solid else 0.0,
        extent=float(np.mean(ext)) if ext else 0.0,
        compactness=float(np.mean(comp)) if comp else 0.0,
        skel_ratio=float(skel.sum()) / (area + 1e-6),  # network -> long skeleton per area
    )


def _granulometry(mask: np.ndarray) -> dict:
    """Pattern spectrum: fraction of sulfide surviving an opening at each radius.
    A fine network collapses at small radii; solid grains survive to large radii."""
    base = float(mask.sum()) + 1e-6
    out = {}
    m = mask.astype(np.uint8)
    for r in _GRANULO_RADII:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        out[f"granulo_r{r}"] = float(opened.sum()) / base
    return out


def _glcm_lbp(gray: np.ndarray, mask: np.ndarray) -> dict:
    q = (gray / 32).astype(np.uint8)  # 8 levels
    glcm = graycomatrix(q, distances=list(_GLCM_DISTANCES), angles=[0, np.pi / 2],
                        levels=8, symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop).mean(axis=1)  # avg over angles
        for d, v in zip(_GLCM_DISTANCES, vals):
            out[f"glcm_{prop}_d{d}"] = float(v)
    lbp = local_binary_pattern(gray, _LBP_P, _LBP_R, method="uniform")
    hist, _ = np.histogram(lbp[mask] if mask.any() else lbp, bins=_LBP_P + 2,
                           range=(0, _LBP_P + 2), density=True)
    for i, v in enumerate(hist):
        out[f"lbp_{i}"] = float(v)
    return out


def extract_features(rgb: np.ndarray, cfg, name: str = "") -> dict:
    """Full feature dict for one image (label-agnostic).

    Magnification is used only to normalise scale, never as a feature (it is
    sparse, unreliable, and class-correlated → would leak).
    """
    mag = magnification_from_name(name)
    pre = preprocess(scale_normalize(rgb, mag), cfg.preprocess)
    seg = segment_phases(pre, cfg.segment)
    gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY)

    feats: dict[str, float] = {}
    feats.update(_cc_features(seg.sulfide))
    feats["magn_frac"] = seg.fractions["magnetite"]
    feats.update(_granulometry(seg.sulfide))
    feats.update(_glcm_lbp(gray, seg.sulfide))
    return feats
