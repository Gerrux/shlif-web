"""Panorama product flow — tile a whole-section scan, classify ore-rich tiles,
aggregate an ore-area-weighted section verdict, and stitch a display overlay.

Ported from ``hakaton_nornikel/scripts/analyze_panorama.py::run_panorama``. The
ore/matrix gate routes through the trained U-Net (``ore_unet_mask``, see
``backend/app/shlif/ore_unet.py``) when the checkpoint and torch are available,
falling back to the classical segmenter otherwise. Torch is never imported here,
at module top level or otherwise, so `import app.pipeline.panorama` works
without torch installed.
"""

from __future__ import annotations

import copy
import time

import cv2
import numpy as np
from PIL import Image

from app.shlif import load_config  # noqa: F401 (kept for parity)
from app.shlif.features import extract_features
from app.shlif.imageio import load_rgb
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import detect_talc
from app.shlif.ore_unet import ore_unet_mask
from app.shlif.tiling import iter_tiles, tile_blend_weight, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader
from app.core import paths

SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
DISPLAY_MP = 4_000_000
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate
_UNC_MAX_SIDE = 1024  # cap the ensemble-uncertainty resolution per tile (mirrors closeup.py)


def ore_density(gray: np.ndarray, bright_thr: float) -> float:
    """Fraction of a tile brighter than the panorama's global bright threshold — an
    ore-density prior (bright sulfide flecks vs faint silicate field)."""
    return float((np.asarray(gray, np.float32) > float(bright_thr)).mean())


def aggregate_section(records, classes) -> np.ndarray:
    """Ore-density-weighted mean of the per-tile class probabilities.

    ``records`` is a list of ``(proba_dict, weight)``. Faint silicate tiles carry a
    near-zero weight so they cannot dilute the verdict; if every weight is zero the
    average falls back to unweighted (never divide-by-zero); empty → zero vector."""
    if not records:
        return np.zeros(len(classes), np.float32)
    W = np.array([max(float(w), 0.0) for _, w in records], np.float32)
    if W.sum() <= 0:
        W = np.ones(len(records), np.float32)
    P = np.array([[float(pd[c]) for c in classes] for pd, _ in records], np.float32)
    return (P * W[:, None]).sum(0) / W.sum()


def _run_panorama(path, clf, feat_names, classes, cfg, min_ore: float = 0.04,
                  display_mp: int = DISPLAY_MP) -> dict:
    """Tile a panorama, classify ore-rich tiles, aggregate a section verdict, and
    stitch a display overlay. Returns a dict with `overlay` (RGB uint8, no banner)
    plus verdict fields. `cfg.tiling.tile` and `cfg.talc.detect_dark_frac` should
    already be set by the caller. Matrix segmentation uses the trained U-Net when
    available, falling back to classical segmentation otherwise; talc detection
    stays classical-only (no GPU U-Net branch)."""
    Wt, Ht, factor = tile_grid(path, cfg.tiling)
    disp = load_rgb(path, max_pixels=display_mp)
    dh, dw = disp.shape[:2]
    rx, ry = dw / Wt, dh / Ht
    # global brightness threshold → per-tile ore-density weight (item: disseminated
    # panoramas where segmentation reads every tile as ore-bearing)
    ore_pct = float(getattr(cfg.tiling, "ore_density_pct", ORE_DENSITY_PCT))
    bright_thr = float(np.percentile(cv2.cvtColor(disp, cv2.COLOR_RGB2GRAY), ore_pct))
    # trained ore/matrix U-Net when available (IoU 0.975 vs classical 0.81);
    # None (missing checkpoint or torch/smp) -> classical segment_phases fallback
    ore_bundle = loader.load_ore_unet()
    ore_source = "unet" if ore_bundle is not None else "classical"

    base = disp.astype(np.float32)
    # Feathered stitch: accumulate weight*colour per tile and normalise, so
    # overlapping tiles blend seamlessly (no double-darkened overlap band, no hard
    # seam between differently-classified neighbours) — borrowed feather pattern.
    color_num = np.zeros((dh, dw, 3), np.float32)
    weight_den = np.zeros((dh, dw), np.float32)
    talc_disp = np.zeros((dh, dw), bool)
    records = []
    low_conf_zones = []
    talc_px = matrix_px = 0
    undet_weighted_sum = 0.0
    undet_px_total = 0
    n_tiles = n_ore = n_matrix = 0
    t0 = time.time()
    sort_alpha = 0.32

    for tile in iter_tiles(path, cfg.tiling):
        n_tiles += 1
        if tile.empty:
            continue
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        if ore_bundle is not None:
            ore_model, ore_device = ore_bundle
            matrix = ~ore_unet_mask(rgb, ore_model, ore_device)
        else:
            matrix = segment_phases(pre, cfg.segment).labels == 0
        talc = detect_talc(pre, matrix, cfg.talc)
        ore_px = int((~matrix).sum())
        ore_frac = ore_px / max(matrix.size, 1)
        talc_px += int(talc.sum()); matrix_px += int(matrix.sum())

        dx0, dy0 = int(tile.x * rx), int(tile.y * ry)
        dx1, dy1 = min(int((tile.x + rgb.shape[1]) * rx), dw), min(int((tile.y + rgb.shape[0]) * ry), dh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue

        th, tw = rgb.shape[:2]
        unc_scale = min(1.0, _UNC_MAX_SIDE / max(th, tw))
        unc_rgb = (cv2.resize(rgb, (int(tw * unc_scale), int(th * unc_scale)),
                              interpolation=cv2.INTER_AREA) if unc_scale < 1 else rgb)
        unc = ensemble_uncertainty(unc_rgb, cfg)
        undet_weighted_sum += unc["undetermined_fraction"] * (th * tw)
        undet_px_total += th * tw
        bx, by = rx / unc_scale, ry / unc_scale
        for z in find_low_conf_zones(unc):
            zx, zy, zw, zh = z["bbox"]
            low_conf_zones.append({
                "bbox": [int(dx0 + zx * bx), int(dy0 + zy * by), int(zw * bx), int(zh * by)],
                "area": z["area"], "phase_a": z["phase_a"], "phase_b": z["phase_b"],
            })

        if ore_frac >= min_ore:
            n_ore += 1
            feats = extract_features(rgb, cfg)
            proba = clf.predict_proba(np.array([[feats[k] for k in feat_names]], float))[0]
            pd = {classes[i]: float(proba[i]) for i in range(len(classes))}
            dens = ore_density(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), bright_thr)
            records.append((pd, dens))
            col = np.array(SORT_RGB[max(pd, key=lambda k: pd[k])], np.float32)
            wgt = tile_blend_weight(dy1 - dy0, dx1 - dx0)
            color_num[dy0:dy1, dx0:dx1] += wgt[..., None] * col
            weight_den[dy0:dy1, dx0:dx1] += wgt
        else:
            n_matrix += 1
        if talc.any():
            td = cv2.resize(talc.astype(np.uint8), (dx1 - dx0, dy1 - dy0),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            talc_disp[dy0:dy1, dx0:dx1] |= td

    sec = aggregate_section(records, classes)
    verdict = classes[int(sec.argmax())] if records else "review"
    conf = float(sec.max()) if records else 0.0

    overlay = base.copy()
    cov = weight_den > 0
    if cov.any():
        blended = color_num[cov] / weight_den[cov][..., None]
        overlay[cov] = (1.0 - sort_alpha) * base[cov] + sort_alpha * blended
    out = overlay
    out[talc_disp] = 0.68 * out[talc_disp] + 0.32 * np.array(TALC_RGB, np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return {
        "overlay": out, "verdict": verdict, "conf": conf,
        "proba": {classes[i]: float(sec[i]) for i in range(len(classes))},
        "talc_frac": talc_px / max(talc_px + matrix_px, 1),
        "n_ore": n_ore, "n_matrix": n_matrix, "n_tiles": n_tiles,
        "seconds": time.time() - t0, "factor": factor,
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
        "ore_source": ore_source,
    }


def analyze_panorama(path: str, cfg, jid: str) -> dict:
    """Public wrapper called by the API for `mode=="panorama"`."""
    cfg = copy.deepcopy(cfg)  # don't mutate the shared @lru_cache'd Config
    cfg.tiling.tile = 2048
    cfg.talc.detect_dark_frac = 0.15
    bundle = loader.load_classifier()
    if bundle is None:
        raise RuntimeError("classifier.pkl required for panorama sort")
    clf, feat, classes = bundle
    r = _run_panorama(path, clf, feat, classes, cfg)
    out = paths.images_dir() / f"{jid}.jpg"
    Image.fromarray(r["overlay"]).save(out, "JPEG", quality=88)
    return {
        "mode": "panorama",
        "verdict": {"ore_class": r["verdict"], "text": "",
                    "metrics": {"talc_frac": r["talc_frac"], "confidence": r["conf"],
                                "sort_proba": r["proba"],
                                "undetermined_fraction": r["undetermined_fraction"]}},
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": r["n_ore"], "n_tiles": r["n_tiles"], "talc_frac": r["talc_frac"],
        "low_conf_zones": r["low_conf_zones"], "ore_source": r["ore_source"],
    }
