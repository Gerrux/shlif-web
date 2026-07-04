"""End-to-end analysis of a single (close-up) image → phases, metrics, verdict.

The intergrowth split here is a *first-cut proxy*: sulfide pixels close to
magnetite are "fine" (densely laced), solid sulfide away from magnetite is
"normal". A texture/GLCM classifier per sulfide grain replaces this proxy next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import phases
from .preprocess import preprocess
from .segment import segment_phases
from .talc import detect_talc, talc_fraction


@dataclass
class Analysis:
    ore_class: str
    text: str
    metrics: dict
    masks: dict = field(default_factory=dict)


def _intergrowth_split(sulfide: np.ndarray, magnetite: np.ndarray, dist_px: int):
    """Split sulfide into (normal, fine) by proximity to magnetite."""
    if not sulfide.any():
        z = np.zeros_like(sulfide)
        return z, z
    non_mag = (~magnetite).astype(np.uint8)
    dist = cv2.distanceTransform(non_mag, cv2.DIST_L2, 3)
    fine = sulfide & (dist <= float(dist_px))
    normal = sulfide & ~fine
    return normal, fine


def verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px: int = 12) -> dict:
    """Phase-composition verdict from (already-decided) phase masks + talc overlay.
    Returns {ore_class, text, metrics}. Shared by analyze_image and the web recompute."""
    talc_frac = talc_fraction(talc)
    normal, fine = _intergrowth_split(sulfide, magnetite, dist_px)
    sulf_area = float(sulfide.sum())
    fine_share = float(fine.sum()) / sulf_area if sulf_area > 0 else 0.0
    normal_share = 1.0 - fine_share

    rule = cfg.rule
    talc_thr = float(rule.talc_threshold)
    dom_thr = float(rule.dominance_threshold)
    if talc_frac > talc_thr:
        ore = phases.ORE_TALCOSE
        confidence = min(1.0, (talc_frac - talc_thr) / max(talc_thr, 1e-6) + 0.5)
    else:
        margin = abs(fine_share - dom_thr) / max(dom_thr, 1e-6)
        confidence = min(1.0, 0.5 + margin)
        ore = phases.ORE_HARD if fine_share > dom_thr else phases.ORE_ORDINARY
        if confidence < float(rule.fine_min_confidence):
            ore = phases.ORE_REVIEW

    total = matrix.size
    metrics = {
        "sulfide_frac": float(sulfide.sum()) / total,
        "magnetite_frac": float(magnetite.sum()) / total,
        "matrix_frac": float(matrix.sum()) / total,
        "talc_frac": talc_frac,
        "normal_share": normal_share,
        "fine_share": fine_share,
        "confidence": confidence,
    }
    return {"ore_class": ore, "text": _verdict_text(ore, metrics),
            "metrics": metrics, "normal": normal, "fine": fine}


def analyze_image(rgb: np.ndarray, cfg, dist_px: int = 12, detect_talc_flag: bool = False,
                  ore_mask: np.ndarray | None = None,
                  talc_mask: np.ndarray | None = None) -> Analysis:
    """Run the full close-up pipeline and apply the expert rule.

    ``talc_mask`` (bool HxW), when supplied, is a precomputed talc segmentation
    (e.g. the trained talc U-Net from ``shlif.talc_unet``) and it OWNS the talc
    decision: it is intersected with the matrix (talc is by definition dispersed
    dark phase in the *non-ore* matrix) and drives ``talc_frac`` and the talcose
    verdict, overriding both ``detect_talc_flag`` and the empty default. It is
    resized to the image if its shape differs.

    ``detect_talc_flag`` gates the *naive* darkness-based talc detector, which
    over-detects (dark != talc). Used as the classical fallback when no
    ``talc_mask`` is given. Off by default (empty talc) otherwise.

    ``ore_mask`` (bool HxW), when supplied, is the illumination-robust binary
    U-Net ore/matrix decision and it OWNS the ore/matrix boundary: magnetite and
    sulfide are constrained to inside ore, so neutral silicate matrix can never be
    mislabelled magnetite under odd lighting (the pervasive bug on differently-lit
    talcose close-ups). Without it we fall back to the classical 3-phase split.
    """
    pre = preprocess(rgb, cfg.preprocess)
    seg = segment_phases(pre, cfg.segment)

    if ore_mask is not None:
        ore_mask = np.asarray(ore_mask, dtype=bool)
        if ore_mask.shape != seg.labels.shape:
            ore_mask = cv2.resize(ore_mask.astype(np.uint8),
                                  (seg.labels.shape[1], seg.labels.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        magnetite = seg.magnetite & ore_mask       # magnetite only where U-Net says ore
        sulfide = ore_mask & ~magnetite            # remaining ore = bright opaque (sulfide)
        matrix = ~ore_mask
    else:
        magnetite = seg.magnetite
        sulfide = seg.sulfide
        matrix = seg.labels == phases.MATRIX

    if talc_mask is not None:
        talc = np.asarray(talc_mask, dtype=bool)
        if talc.shape != matrix.shape:
            talc = cv2.resize(talc.astype(np.uint8), (matrix.shape[1], matrix.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        talc = talc & matrix              # talc lives in the non-ore matrix (operational def)
    elif detect_talc_flag:
        talc = detect_talc(pre, matrix, cfg.talc)
    else:
        talc = np.zeros(rgb.shape[:2], dtype=bool)
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg, dist_px)
    ore, text, metrics, normal, fine = v["ore_class"], v["text"], v["metrics"], v["normal"], v["fine"]
    masks = {
        "sulfide": sulfide,
        "magnetite": magnetite,
        "matrix": matrix,
        "talc": talc,
        "normal": normal,
        "fine": fine,
        "preprocessed": pre,
        "seg": seg,
    }
    return Analysis(ore_class=ore, text=text, metrics=metrics, masks=masks)


def _verdict_text(ore: str, m: dict) -> str:
    name = phases.ORE_CLASS_RU[ore]
    talc = 100 * m["talc_frac"]
    fine = 100 * m["fine_share"]
    if ore == phases.ORE_TALCOSE:
        return f"Классифицирована как {name}: тальк {talc:.1f}%, преобладание тонких срастаний {fine:.0f}%."
    if ore == phases.ORE_HARD:
        return f"Классифицирована как {name}: тальк {talc:.1f}%, преобладание тонких срастаний {fine:.0f}%."
    if ore == phases.ORE_ORDINARY:
        return f"Классифицирована как {name}: тальк {talc:.1f}%, преобладание обычных срастаний {100 - fine:.0f}%."
    return f"Требует проверки: тальк {talc:.1f}%, тонкие срастания {fine:.0f}% (низкая уверенность)."
