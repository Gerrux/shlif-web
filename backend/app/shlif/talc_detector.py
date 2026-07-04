"""Learned talc detector: SLIC superpixels + Random Forest, trained on the blue
polygon zones. Tuned/evaluated for IoU + Hausdorff (the scored segmentation
metrics), grouped by image so no image leaks between train and test.

The polygons are coarse *seed zones*, not exact masks — superpixels snap the
decision boundary to real image edges, which lands better on IoU than per-pixel
thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic

from .preprocess import preprocess
from .segment import segment_phases
from .talc import talc_mask_from_contours

FEATURE_NAMES = ["L", "a", "b", "std_L", "grad", "gray", "ore_frac"]


@dataclass
class ImageSP:
    sp: np.ndarray            # HxW superpixel labels
    feats: np.ndarray         # (n_sp, n_feat)
    ore_frac: np.ndarray      # (n_sp,) fraction of superpixel that is ore
    talc_frac: np.ndarray     # (n_sp,) fraction inside GT talc zone
    shape: tuple[int, int]


def superpixels(pre: np.ndarray, seg, n_segments: int = 700, compactness: float = 12.0):
    """SLIC superpixels + per-superpixel colour/texture/ore features."""
    lab = cv2.cvtColor(pre, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)

    sp = slic(pre, n_segments=n_segments, compactness=compactness, start_label=0)
    ids = np.arange(int(sp.max()) + 1)
    ore = (seg.labels > 0).astype(np.float32)

    def mean(x):
        return ndi.mean(x, sp, ids)

    feats = np.stack(
        [mean(lab[..., 0]), mean(lab[..., 1]), mean(lab[..., 2]),
         np.sqrt(np.maximum(ndi.variance(lab[..., 0], sp, ids), 0)),
         mean(grad), mean(gray)],
        axis=1,
    )
    ore_frac = ndi.mean(ore, sp, ids)
    feats = np.column_stack([feats, ore_frac])
    return sp, ids, feats, ore_frac


def prepare_image(raw: np.ndarray, annotated: np.ndarray, cfg,
                  n_segments: int = 700) -> ImageSP:
    """Build superpixel features for one raw image + its GT talc zones."""
    pre = preprocess(raw, cfg.preprocess)
    seg = segment_phases(pre, cfg.segment)
    gt = talc_mask_from_contours(annotated, cfg.talc).astype(np.float32)
    sp, ids, feats, ore_frac = superpixels(pre, seg, n_segments)
    talc_frac = ndi.mean(gt, sp, ids)
    return ImageSP(sp=sp, feats=feats, ore_frac=ore_frac, talc_frac=talc_frac,
                   shape=raw.shape[:2])


def training_rows(img: ImageSP, pos_thr: float = 0.5, neg_thr: float = 0.15,
                  ore_thr: float = 0.5):
    """(X, y) superpixel rows for one image: talc-zone vs matrix, ore excluded."""
    keep_ore = img.ore_frac < ore_thr
    pos = keep_ore & (img.talc_frac >= pos_thr)
    neg = keep_ore & (img.talc_frac <= neg_thr)
    X = np.vstack([img.feats[pos], img.feats[neg]])
    y = np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
    return X, y


def predict_mask(img: ImageSP, clf, ore_thr: float = 0.5, prob_thr: float = 0.5):
    """Per-superpixel talc probability -> full-resolution talc mask."""
    proba = clf.predict_proba(img.feats)[:, list(clf.classes_).index(1)]
    talc_sp = (proba >= prob_thr) & (img.ore_frac < ore_thr)
    return talc_sp[img.sp]
