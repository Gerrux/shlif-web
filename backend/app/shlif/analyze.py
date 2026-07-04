"""End-to-end analysis of a single (close-up) image → phases, metrics, verdict.

The intergrowth split is by sulfide **grain size** (organiser expert criterion,
2026-07-04): ore is ground before flotation, and it is the *size* of the sulfide
grains — not the % replacement by gangue — that sets how cleanly they liberate.
Coarse, blocky sulfides survive comminution as free particles → рядовая; fine,
laced sulfide networks stay locked in the non-ore matrix → труднообогатимая. The
texture/granulometry RF classifier is the primary sort; this rule card is the
interpretable estimate shown alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import phases
from .preprocess import preprocess
from .segment import segment_phases
from .talc import dark_gray_phase, detect_talc, talc_fraction


@dataclass
class Analysis:
    ore_class: str
    text: str
    metrics: dict
    masks: dict = field(default_factory=dict)


def _liberation_split(sulfide: np.ndarray, lib_px: int):
    """Split sulfide into (coarse, fine) by GRAIN SIZE — the liberation criterion.

    A morphological opening with a disk of radius ``lib_px`` (the grind/liberation
    grain size) keeps only sulfide features at least ~2*lib_px thick: coarse,
    blocky grains that liberate as free particles on grinding (рядовая marker).
    Thin, laced sulfide collapses under the opening → "fine", i.e. grains that stay
    locked in the non-ore matrix after comminution (труднообогатимая marker).

    Empirically (1109 imgs) this is what separates the classes: coarse-share
    (granulometry) drives ordinary-vs-hard, while % magnetite replacement is
    near-useless (AUC 0.51). Replaces the old magnetite-proximity proxy.
    """
    if not sulfide.any():
        z = np.zeros_like(sulfide)
        return z, z
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * lib_px + 1, 2 * lib_px + 1))
    coarse = cv2.morphologyEx(sulfide.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) & sulfide
    fine = sulfide & ~coarse
    return coarse, fine


def verdict_from_masks(sulfide, magnetite, matrix, talc, cfg) -> dict:
    """Phase-composition verdict from (already-decided) phase masks + talc overlay.
    Returns {ore_class, text, metrics}. Shared by analyze_image and the web recompute."""
    rule = cfg.rule
    talc_frac = talc_fraction(talc)
    long_side = max(sulfide.shape)
    lib_px = max(2, int(round(float(getattr(rule, "liberation_radius_frac", 0.01)) * long_side)))
    normal, fine = _liberation_split(sulfide, lib_px)  # normal = coarse/liberated grains
    sulf_area = float(sulfide.sum())
    fine_share = float(fine.sum()) / sulf_area if sulf_area > 0 else 0.0
    normal_share = 1.0 - fine_share

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


def analyze_image(rgb: np.ndarray, cfg, detect_talc_flag: bool = False,
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
    v = verdict_from_masks(sulfide, magnetite, matrix, talc, cfg)
    ore, text, metrics, normal, fine = v["ore_class"], v["text"], v["metrics"], v["normal"], v["fine"]

    # Independent talc-share proxy: dispersed medium-dark grey phase inside the
    # matrix. A robust second opinion on the talc share, reported alongside the
    # segmenter-driven talc_frac (borrowed heuristic; see talc.dark_gray_phase).
    dg, _ = dark_gray_phase(rgb, cfg.talc)
    dg = dg & matrix
    metrics["talc_share_est"] = float(dg.mean())

    masks = {
        "sulfide": sulfide,
        "magnetite": magnetite,
        "matrix": matrix,
        "talc": talc,
        "talc_dispersed": dg,
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
    coarse = 100 * m["normal_share"]
    if ore == phases.ORE_TALCOSE:
        return f"Классифицирована как {name}: тальк {talc:.1f}%, мелких (запертых) сульфидов {fine:.0f}%."
    if ore == phases.ORE_HARD:
        return (f"Классифицирована как {name}: тальк {talc:.1f}%, преобладают мелкие сульфиды "
                f"{fine:.0f}% — заперты в нерудной матрице, плохо раскрываются при измельчении.")
    if ore == phases.ORE_ORDINARY:
        return (f"Классифицирована как {name}: тальк {talc:.1f}%, преобладают крупные сульфиды "
                f"{coarse:.0f}% — свободны после измельчения.")
    return f"Требует проверки: тальк {talc:.1f}%, мелкие сульфиды {fine:.0f}% (низкая уверенность)."
