"""Panorama product flow — tile a whole-section scan, classify ore-rich tiles,
aggregate an ore-area-weighted section verdict, and stitch a display overlay.

Ported from ``hakaton_nornikel/scripts/analyze_panorama.py::run_panorama``
(classical path only — no U-Net). Torch is never imported here, at module top
level or otherwise, so `import app.pipeline.panorama` works without torch
installed.
"""

from __future__ import annotations

import copy
import time

import cv2
import numpy as np
from PIL import Image

from app.shlif import load_config, phases  # noqa: F401 (load_config kept for parity)
from app.shlif.features import extract_features
from app.shlif.preprocess import preprocess
from app.shlif.segment import segment_phases
from app.shlif.talc import dark_gray_phase, detect_talc
from app.shlif.tiling import axis_core_bounds, iter_tiles, load_working_array, tile_blend_weight, tile_grid
from app.pipeline import loader, masks
from app.core import paths

SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
ORE_DENSITY_PCT = 92.0  # global brightness percentile that separates ore flecks from silicate


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


def _assemble_masks(path: str, cfg, arr: np.ndarray) -> dict:
    """Tile the section, segment + talc-detect each tile, and reassemble one
    continuous mask set for the whole working canvas — core-crop (no overlap
    double count, see `axis_core_bounds`) — so `verdict_from_masks` sees the
    same kind of input it gets from a single close-up pass."""
    H, W = arr.shape[:2]
    sulfide = np.zeros((H, W), bool)
    magnetite = np.zeros((H, W), bool)
    matrix = np.zeros((H, W), bool)
    talc = np.zeros((H, W), bool)
    dg = np.zeros((H, W), bool)
    tile_px = int(cfg.tiling.tile)
    step = max(1, tile_px - int(cfg.tiling.overlap))
    x_core_end = axis_core_bounds(W, tile_px, step)
    y_core_end = axis_core_bounds(H, tile_px, step)

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        # a tile's core always starts at its own (x, y) — only the end can be
        # pulled in earlier than the tile's full extent, per axis_core_bounds
        cx0, cy0 = tile.x, tile.y
        cx1, cy1 = x_core_end[tile.x], y_core_end[tile.y]
        lx1, ly1 = cx1 - tile.x, cy1 - tile.y

        if tile.empty:
            matrix[cy0:cy1, cx0:cx1] = True
            continue

        pre = preprocess(tile.rgb, cfg.preprocess)
        seg = segment_phases(pre, cfg.segment)
        tk = detect_talc(pre, seg.labels == phases.MATRIX, cfg.talc)
        dgm, _ = dark_gray_phase(tile.rgb, cfg.talc)

        sulfide[cy0:cy1, cx0:cx1] = seg.sulfide[:ly1, :lx1]
        magnetite[cy0:cy1, cx0:cx1] = seg.magnetite[:ly1, :lx1]
        matrix[cy0:cy1, cx0:cx1] = seg.labels[:ly1, :lx1] == phases.MATRIX
        talc[cy0:cy1, cx0:cx1] = tk[:ly1, :lx1]
        dg[cy0:cy1, cx0:cx1] = dgm[:ly1, :lx1] & (seg.labels[:ly1, :lx1] == phases.MATRIX)

    return {"sulfide": sulfide, "magnetite": magnetite, "matrix": matrix, "talc": talc, "dg": dg}


def _run_panorama(path, clf, feat_names, classes, cfg, arr: np.ndarray, min_ore: float = 0.04) -> dict:
    """Tile a panorama, classify ore-rich tiles for the `sort` card (ore-density
    weighted aggregation — unchanged mechanism, see design spec §4.2), and
    stitch the display overlay. The whole-image phase/talc masks and the
    `ore_class` verdict come from `_assemble_masks` + `verdict_from_masks`
    instead (design spec §4.1) — this function no longer decides ore_class."""
    Wt, Ht, factor = tile_grid(path, cfg.tiling)
    edit = masks.fit_max_side(arr, masks.EDIT_MAX_SIDE, cv2.INTER_AREA)
    dh, dw = edit.shape[:2]
    rx, ry = dw / Wt, dh / Ht
    ore_pct = float(getattr(cfg.tiling, "ore_density_pct", ORE_DENSITY_PCT))
    bright_thr = float(np.percentile(cv2.cvtColor(edit, cv2.COLOR_RGB2GRAY), ore_pct))

    base = edit.astype(np.float32)
    # Feathered stitch: accumulate weight*colour per tile and normalise, so
    # overlapping tiles blend seamlessly in the *display* overlay (no double-
    # darkened overlap band, no hard seam) — cosmetic only, unrelated to the
    # whole-canvas mask assembly above.
    color_num = np.zeros((dh, dw, 3), np.float32)
    weight_den = np.zeros((dh, dw), np.float32)
    talc_disp = np.zeros((dh, dw), bool)
    records = []
    n_tiles = n_ore = n_matrix = 0
    t0 = time.time()
    sort_alpha = 0.32

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        n_tiles += 1
        if tile.empty:
            continue
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        matrix = segment_phases(pre, cfg.segment).labels == phases.MATRIX
        talc = detect_talc(pre, matrix, cfg.talc)
        ore_px = int((~matrix).sum())
        ore_frac = ore_px / max(matrix.size, 1)

        dx0, dy0 = int(tile.x * rx), int(tile.y * ry)
        dx1, dy1 = min(int((tile.x + rgb.shape[1]) * rx), dw), min(int((tile.y + rgb.shape[0]) * ry), dh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue

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
    sort_proba = {classes[i]: float(sec[i]) for i in range(len(classes))}
    sort_top = classes[int(sec.argmax())] if records else classes[0]

    overlay = base.copy()
    cov = weight_den > 0
    if cov.any():
        blended = color_num[cov] / weight_den[cov][..., None]
        overlay[cov] = (1.0 - sort_alpha) * base[cov] + sort_alpha * blended
    out = overlay
    out[talc_disp] = 0.68 * out[talc_disp] + 0.32 * np.array(TALC_RGB, np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return {
        "overlay": out, "edit_rgb": edit, "sort": {"classes": sort_proba, "top": sort_top},
        "n_ore": n_ore, "n_matrix": n_matrix, "n_tiles": n_tiles,
        "seconds": time.time() - t0, "factor": factor,
    }


def analyze_panorama(path: str, cfg, jid: str) -> dict:
    """Public wrapper called by the API. Builds the whole-canvas phase/talc
    masks (design spec §4) and reuses `verdict_from_masks_dict` — the exact
    helper close-up uses — so the result has the same shape and the same
    meaning, computed over the whole image instead of per tile."""
    cfg = copy.deepcopy(cfg)  # don't mutate the shared @lru_cache'd Config
    cfg.tiling.tile = 2048
    cfg.talc.detect_dark_frac = 0.15
    bundle = loader.load_classifier()
    if bundle is None:
        raise RuntimeError("classifier.pkl required for panorama sort")
    clf, feat, classes = bundle

    arr = load_working_array(path, cfg.tiling)
    H, W = arr.shape[:2]

    assembled = _assemble_masks(path, cfg, arr)
    verdict = masks.verdict_from_masks_dict(
        assembled["sulfide"], assembled["magnetite"], assembled["matrix"], assembled["talc"], cfg)
    verdict["metrics"]["talc_share_est"] = float(assembled["dg"].mean())

    run = _run_panorama(path, clf, feat, classes, cfg, arr)
    Image.fromarray(run["overlay"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)

    edit = run["edit_rgb"]
    eh, ew = edit.shape[:2]
    sulfide_small = cv2.resize(assembled["sulfide"].astype(np.uint8), (ew, eh),
                               interpolation=cv2.INTER_NEAREST) > 0
    magnetite_small = cv2.resize(assembled["magnetite"].astype(np.uint8), (ew, eh),
                                 interpolation=cv2.INTER_NEAREST) > 0
    talc_small = cv2.resize(assembled["talc"].astype(np.uint8), (ew, eh),
                            interpolation=cv2.INTER_NEAREST) > 0
    phase_small = masks.phase_label_map(sulfide_small, magnetite_small)
    unc = masks.uncertainty_for_editor(edit, cfg)

    masks.persist_editor_artifacts(jid, {
        "phase_map": phase_small, "talc": talc_small,
        "superpixels": masks.build_superpixel_map(edit),
        "darkness": masks.build_darkness_map(edit),
        "confidence": unc["confidence"],
    })

    return {
        "mode": "panorama",
        "verdict": verdict,
        "sort": run["sort"],
        "text": verdict["text"],
        "size": [ew, eh],
        "native_size": [W, H],
        "low_conf_zones": unc["low_conf_zones"],
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": run["n_ore"], "n_tiles": run["n_tiles"],
    }
