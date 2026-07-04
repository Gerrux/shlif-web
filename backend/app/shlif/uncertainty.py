"""Ensemble-perturbation uncertainty for the classical phase segmentation.

We have no probabilistic segmenter on the CPU path, so we synthesise one: run the
multi-Otsu + Lab-colour ``segment_phases`` under a handful of soft photometric
perturbations (gamma / gain jitter) and look at how stable each pixel's phase
label is across the ensemble. Pixels whose label never flips are confident; pixels
that flip between phases under mild re-lighting are disputed. This gives a
per-pixel confidence map, a scalar ``undetermined_fraction`` and the disputed-zone
mask — an honesty signal for the human-in-the-loop, exactly where the automatic
verdict should be double-checked. (Borrowed idea; adapted to the classical path.)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import phases
from .preprocess import preprocess
from .segment import segment_phases

# (gamma, gain) photometric perturbations — identity plus mild brighten/darken.
_PERTURBATIONS = (
    (1.0, 1.0),
    (0.82, 1.0),
    (1.22, 1.0),
    (1.0, 0.88),
    (1.0, 1.12),
)
_N_PHASES = 3  # matrix / magnetite / sulfide
_PHASE_RU = {phases.MATRIX: "матрица", phases.MAGNETITE: "магнетит", phases.SULFIDE: "сульфид"}

_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    """Lazily-created, process-wide thread pool for the perturbation ensemble.
    Persistent (not re-created per call) — this runs on every non-empty tile,
    potentially thousands of times per gigapixel panorama. segment_phases and
    its preprocessing are cv2/numpy/skimage calls on large arrays, which
    release the GIL, so threads (not processes) give real parallelism here
    without pickling/IPC overhead per tile. Uses double-checked locking to
    ensure thread-safe initialization."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=min(len(_PERTURBATIONS), os.cpu_count() or 1))
    return _POOL


def _perturb(rgb: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    x = np.clip((rgb.astype(np.float32) / 255.0) ** gamma * gain, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def ensemble_phase_labels(rgb: np.ndarray, cfg, perturbations=_PERTURBATIONS, on_step=None) -> np.ndarray:
    """Stack of phase-label maps (K, H, W) — one classical segmentation per
    photometric perturbation, run concurrently across a thread pool (they are
    independent of each other). `on_step(i, total)`, if given, is called once
    per perturbation in the same fixed 1..total order as before — every
    perturbation is submitted to the pool up front (so they run in parallel),
    but progress is still reported in original order, not completion order."""
    def _one(pert):
        gamma, gain = pert
        pre = preprocess(_perturb(rgb, gamma, gain), cfg.preprocess)
        return segment_phases(pre, cfg.segment).labels.astype(np.uint8)

    total = len(perturbations)
    futures = [_pool().submit(_one, pert) for pert in perturbations]
    maps = []
    for i, f in enumerate(futures, 1):
        maps.append(f.result())
        if on_step:
            on_step(i, total)
    return np.stack(maps)


def _vote_fractions(label_stack: np.ndarray, n_phases: int = _N_PHASES) -> np.ndarray:
    """Per-pixel vote fraction for each phase → (C, H, W), sums to 1 over C."""
    k = label_stack.shape[0]
    return np.stack([(label_stack == c).sum(0) / k for c in range(n_phases)])


def confidence_map(label_stack: np.ndarray, n_phases: int = _N_PHASES) -> np.ndarray:
    """Per-pixel agreement of the modal phase (1 = unanimous, 1/C = fully split)."""
    return _vote_fractions(label_stack, n_phases).max(0).astype(np.float32)


def entropy_map(label_stack: np.ndarray, n_phases: int = _N_PHASES) -> np.ndarray:
    """Normalised (0..1) per-pixel label entropy — 0 confident, 1 maximally split."""
    p = _vote_fractions(label_stack, n_phases)
    ent = -(p * np.log(p + 1e-12)).sum(0)
    return (ent / np.log(n_phases)).astype(np.float32)


def ensemble_uncertainty(rgb: np.ndarray, cfg, conf_thr: float = 0.7, on_step=None) -> dict:
    """Run the perturbation ensemble and summarise its disagreement.

    Returns ``confidence`` (HxW float 0..1), ``entropy`` (HxW float 0..1),
    ``low_conf`` (HxW bool — pixels whose modal phase held in fewer than
    ``conf_thr`` of the runs), ``undetermined_fraction`` (scalar) and the
    ensemble ``labels`` stack. ``on_step``, if given, is forwarded to
    ``ensemble_phase_labels`` for progress reporting.
    """
    stack = ensemble_phase_labels(rgb, cfg, on_step=on_step)
    conf = confidence_map(stack)
    low_conf = conf < float(conf_thr)
    return {
        "confidence": conf,
        "entropy": entropy_map(stack),
        "low_conf": low_conf,
        "undetermined_fraction": float(low_conf.mean()),
        "labels": stack,
    }


def find_low_conf_zones(result: dict, min_area: int = 64) -> list[dict]:
    """Label the disputed regions and name the two phases they argue between.

    Each zone: ``{bbox, area, phase_a, phase_b}`` where a/b are the two most-voted
    phases inside the region. Small specks below ``min_area`` are dropped.
    """
    import cv2

    low = result["low_conf"].astype(np.uint8)
    if not low.any():
        return []
    stack = result["labels"]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(low, 8)
    zones = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        region = lbl == i
        votes = [int((stack[:, region] == c).sum()) for c in range(_N_PHASES)]
        order = np.argsort(votes)[::-1]
        zones.append({
            "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                     int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
            "area": area,
            "phase_a": _PHASE_RU[int(order[0])],
            "phase_b": _PHASE_RU[int(order[1])],
        })
    zones.sort(key=lambda z: z["area"], reverse=True)
    return zones
