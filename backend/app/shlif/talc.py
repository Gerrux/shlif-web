"""Talc handling.

Two jobs:
  1. Extract a *ground-truth* talc mask from the blue hand-drawn contours in the
     annotation images — used to calibrate and validate the detector. The marks
     are mixed: some are closed loops around talc, some are open phase-boundary
     strokes. We fill only regions enclosed by loops, detected as background
     connected-components that do not touch the image border. (Naive flood-fill
     from a corner mislabels whole halves — do not use it.)
  2. Detect talc on an unlabelled image as the darkest sub-population of the
     non-ore matrix. Reliable on bright close-ups; low-confidence on dark
     panoramas (flagged by the caller).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def blue_line_mask(rgb: np.ndarray, cfg) -> np.ndarray:
    """Boolean mask of the drawn blue/cyan annotation strokes.

    Blue strokes read B high with B≫R and B≫G. Some annotators draw in cyan, which
    reads B≫R but B≈G, so the pure-blue rule misses it — we add a cyan branch
    (B high, G high, both ≫ R) and union the two. The ``cyan_*`` thresholds fall
    back to the blue ones when absent, so the vendored/origin config stays valid.
    """
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    blue = (
        (b > int(cfg.blue_b_min))
        & (b - r > int(cfg.blue_minus_r))
        & (b - g > int(cfg.blue_minus_g))
    )
    cyan_b_min = int(getattr(cfg, "cyan_b_min", cfg.blue_b_min))
    cyan_g_min = int(getattr(cfg, "cyan_g_min", cfg.blue_b_min))
    cyan_minus_r = int(getattr(cfg, "cyan_minus_r", cfg.blue_minus_r))
    cyan = (
        (b > cyan_b_min)
        & (g > cyan_g_min)
        & (b - r > cyan_minus_r)
        & (g - r > cyan_minus_r)
    )
    return blue | cyan


def strip_annotation(rgb: np.ndarray, cfg, dilate: int = 3, radius: int = 4) -> np.ndarray:
    """Inpaint hand-drawn blue/cyan annotation out of ``rgb`` (cv2 TELEA).

    Keeps the drawn strokes from leaking into GLCM/LBP/granulometry features. The
    stroke mask is dilated a little first so the anti-aliased fringe is covered.
    Returns ``rgb`` unchanged (no copy) when there is no annotation, so a clean
    upload pays nothing.
    """
    mask = blue_line_mask(rgb, cfg)
    if not mask.any():
        return rgb
    m = mask.astype(np.uint8)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        m = cv2.dilate(m, k)
    return cv2.inpaint(np.ascontiguousarray(rgb), m, radius, cv2.INPAINT_TELEA)


def talc_mask_from_contours(rgb: np.ndarray, cfg) -> np.ndarray:
    """Ground-truth talc mask = interiors enclosed by the blue loops + the lines."""
    line = blue_line_mask(rgb, cfg)
    if not line.any():
        return np.zeros(rgb.shape[:2], dtype=bool)

    r = int(cfg.close_radius)
    if r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_CLOSE, k, iterations=2).astype(bool)

    bg = ~line
    lbl, n = ndi.label(bg)
    border = set(np.unique(lbl[0, :])) | set(np.unique(lbl[-1, :]))
    border |= set(np.unique(lbl[:, 0])) | set(np.unique(lbl[:, -1]))
    interior_ids = [i for i in range(1, n + 1) if i not in border]
    interior = np.isin(lbl, interior_ids)
    return interior | line


def detect_talc(rgb: np.ndarray, matrix_mask: np.ndarray, cfg) -> np.ndarray:
    """Detect talc as the darkest fraction of the (non-ore) matrix."""
    if not matrix_mask.any():
        return np.zeros(rgb.shape[:2], dtype=bool)
    v = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    vals = v[matrix_mask]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-3:
        return np.zeros(rgb.shape[:2], dtype=bool)
    thr = lo + float(cfg.detect_dark_frac) * (hi - lo)
    return matrix_mask & (v <= thr)


def talc_fraction(talc: np.ndarray) -> float:
    """Talc area as a fraction of the whole image."""
    return float(talc.sum()) / float(talc.size)


def dark_gray_phase(rgb: np.ndarray, cfg) -> tuple[np.ndarray, float]:
    """Robust area proxy for the dispersed medium-dark (talc-like) phase.

    Not a segmentation — a graceful fallback for the talc SHARE where the darkness
    segmenter over/under-fires. Selects pixels that are:
      * grey — low HSV saturation and low Lab chroma (not coloured, not annotation);
      * medium-dark — in the lower-middle of the dynamic range, i.e. above the black
        bottom (``dg_black_pct``) and below ``dg_dark_span`` of the way up to the
        bright-sulfide reference (``dg_bright_pct``);
      * dispersed — any single connected component larger than ``dg_cap_frac`` of
        the image is dropped, since a big solid region is matrix/hole, not talc.
    Returns ``(mask, area_fraction)``. Thresholds fall back to sane defaults so the
    vendored/origin config stays valid.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    a = lab[..., 1] - 128.0
    bb = lab[..., 2] - 128.0
    chroma = np.sqrt(a * a + bb * bb)
    sat = hsv[..., 1].astype(np.float32)

    black_pct = float(getattr(cfg, "dg_black_pct", 8.0))
    bright_pct = float(getattr(cfg, "dg_bright_pct", 88.0))
    span = float(getattr(cfg, "dg_dark_span", 0.55))
    sat_max = float(getattr(cfg, "dg_sat_max", 60.0))
    chroma_max = float(getattr(cfg, "dg_chroma_max", 22.0))
    cap = float(getattr(cfg, "dg_cap_frac", 0.12))

    floor = float(np.percentile(gray, black_pct))
    bright_ref = float(np.percentile(gray, bright_pct))
    mid = floor + span * (bright_ref - floor)

    cand = (gray > floor) & (gray <= mid) & (sat <= sat_max) & (chroma <= chroma_max)
    cand &= ~blue_line_mask(rgb, cfg)
    if not cand.any():
        return np.zeros((h, w), bool), 0.0

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    cap_px = cap * h * w
    keep = np.zeros((h, w), bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= cap_px:
            keep |= lbl == i
    return keep, float(keep.mean())
