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
    """Boolean mask of the drawn blue annotation strokes."""
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    return (
        (b > int(cfg.blue_b_min))
        & (b - r > int(cfg.blue_minus_r))
        & (b - g > int(cfg.blue_minus_g))
    )


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
