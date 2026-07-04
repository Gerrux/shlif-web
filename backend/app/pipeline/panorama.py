"""Panorama product flow — tile a whole-section scan, classify ore-rich tiles,
aggregate an ore-area-weighted section verdict, and stitch a display overlay.

Ported from ``hakaton_nornikel/scripts/analyze_panorama.py::run_panorama``. The
ore/matrix gate routes through the trained U-Net (``ore_unet_mask``, see
``backend/app/shlif/ore_unet.py``) when its checkpoint and torch are available,
falling back to the classical segmenter otherwise; talc per tile similarly
comes from the trained talc U-Net when its weights are loadable, else the
classical ``detect_talc``. Torch is never imported at this module's top
level — only lazily, inside the U-Net loaders/mask functions when a U-Net
path actually runs — so `import app.pipeline.panorama` still works without
torch installed.

Note: `_assemble_masks` (the whole-canvas mask reconstruction that feeds the
reported verdict) still uses the classical segmenter only, not the U-Net gate
`_run_panorama` uses for its own matrix/talc decisions below. Wiring U-Net into
`_assemble_masks` too — mirroring how `shlif.analyze.analyze_image` combines an
`ore_mask` with the classical sulfide/magnetite split — is a reasonable
follow-up, but is new, undesigned work; deliberately left classical-only here
rather than improvised during this merge.
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
from app.shlif.ore_unet import ore_unet_mask
from app.shlif.talc_unet import talc_unet_mask
from app.shlif.tiling import axis_core_bounds, count_tiles, iter_tiles, load_working_array, tile_blend_weight, tile_grid
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import loader, masks, tiles
from app.core import paths

SORT_RGB = {"ordinary": (80, 190, 120), "hard": (225, 85, 80), "talcose": (95, 140, 235)}
TALC_RGB = (60, 120, 255)
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


def _assemble_masks(path: str, cfg, arr: np.ndarray, on_progress=None) -> dict:
    """Tile the section, segment + talc-detect each tile, and reassemble one
    continuous mask set for the whole working canvas — core-crop (no overlap
    double count, see `axis_core_bounds`) — so `verdict_from_masks` sees the
    same kind of input it gets from a single close-up pass. Classical
    segmentation only (see module docstring). `on_progress(progress, message)`,
    if given, is called once per tile, scaled into the 0.05-0.35 job-progress
    range (this is the first of panorama's two tile loops)."""
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
    total = max(1, count_tiles(path, cfg.tiling))
    n = 0

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        n += 1
        if on_progress:
            on_progress(0.05 + 0.30 * min(1.0, n / total), f"сборка масок ({n}/{total})")
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


def _run_panorama(path, clf, feat_names, classes, cfg, arr: np.ndarray, min_ore: float = 0.04,
                  on_progress=None) -> dict:
    """Tile a panorama, classify ore-rich tiles for the `sort` card (ore-density
    weighted aggregation — unchanged mechanism, see design spec §4.2), estimate
    per-tile ensemble uncertainty, and stitch the display overlay. Matrix
    segmentation uses the trained ore/matrix U-Net when available (IoU 0.975 vs
    classical 0.81), falling back to classical segmentation otherwise; talc
    similarly prefers the trained talc U-Net over the classical detector. The
    whole-image phase/talc masks and the `ore_class` verdict come from
    `_assemble_masks` + `verdict_from_masks` instead (design spec §4.1) — this
    function no longer decides ore_class. `on_progress(progress, message)`, if
    given, is called once per tile, scaled into the 0.35-0.85 job-progress
    range (this is the second of panorama's two tile loops, and the more
    expensive one — it runs a 5-perturbation ensemble per tile)."""
    unet = loader.load_talc_unet()
    ore_bundle = loader.load_ore_unet()
    ore_source = "unet" if ore_bundle is not None else "classical"

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
    low_conf_zones = []
    undet_weighted_sum = 0.0
    undet_px_total = 0
    n_tiles = n_ore = n_matrix = 0
    t0 = time.time()
    sort_alpha = 0.32
    total_tiles_est = max(1, count_tiles(path, cfg.tiling))

    for tile in iter_tiles(path, cfg.tiling, arr=arr):
        n_tiles += 1
        if on_progress:
            on_progress(0.35 + 0.50 * min(1.0, n_tiles / total_tiles_est),
                        f"сегментация тайлов ({n_tiles}/{total_tiles_est})")
        if tile.empty:
            continue
        rgb = tile.rgb
        pre = preprocess(rgb, cfg.preprocess)
        if ore_bundle is not None:
            ore_model, ore_device = ore_bundle
            matrix = ~ore_unet_mask(rgb, ore_model, ore_device)
        else:
            matrix = segment_phases(pre, cfg.segment).labels == phases.MATRIX
        if unet is not None:
            model, device = unet
            talc = talc_unet_mask(rgb, model, device, thr=None) & matrix
        else:
            talc = detect_talc(pre, matrix, cfg.talc)
        ore_px = int((~matrix).sum())
        ore_frac = ore_px / max(matrix.size, 1)

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
        "undetermined_fraction": undet_weighted_sum / max(undet_px_total, 1),
        "low_conf_zones": low_conf_zones,
        "ore_source": ore_source,
    }


def analyze_panorama(path: str, cfg, jid: str, on_progress=None) -> dict:
    """Public wrapper called by the API. Builds the whole-canvas phase/talc
    masks (design spec §4) and reuses `verdict_from_masks_dict` — the exact
    helper close-up uses — so the result has the same shape and the same
    meaning, computed over the whole image instead of per tile."""
    def report(p, msg):
        if on_progress:
            on_progress(p, msg)

    cfg = copy.deepcopy(cfg)  # don't mutate the shared @lru_cache'd Config
    cfg.tiling.tile = 2048
    cfg.talc.detect_dark_frac = 0.15
    bundle = loader.load_classifier()
    if bundle is None:
        raise RuntimeError("classifier.pkl required for panorama sort")
    clf, feat, classes = bundle

    report(0.05, "загрузка изображения")
    arr = load_working_array(path, cfg.tiling)
    H, W = arr.shape[:2]

    assembled = _assemble_masks(path, cfg, arr, on_progress=on_progress)
    report(0.35, "вердикт по фазам")
    verdict = masks.verdict_from_masks_dict(
        assembled["sulfide"], assembled["magnetite"], assembled["matrix"], assembled["talc"], cfg)
    verdict["metrics"]["talc_share_est"] = float(assembled["dg"].mean())

    run = _run_panorama(path, clf, feat, classes, cfg, arr, on_progress=on_progress)
    verdict["metrics"]["undetermined_fraction"] = run["undetermined_fraction"]
    report(0.85, "сохранение оверлея")
    Image.fromarray(run["overlay"]).save(paths.images_dir() / f"{jid}.jpg", "JPEG", quality=88)

    try:
        tiles.build_pyramid(arr, jid)
    except Exception as e:
        print(f"panorama tile pyramid failed for job {jid}: {e}")

    edit = run["edit_rgb"]
    eh, ew = edit.shape[:2]
    sulfide_small = cv2.resize(assembled["sulfide"].astype(np.uint8), (ew, eh),
                               interpolation=cv2.INTER_NEAREST) > 0
    magnetite_small = cv2.resize(assembled["magnetite"].astype(np.uint8), (ew, eh),
                                 interpolation=cv2.INTER_NEAREST) > 0
    talc_small = cv2.resize(assembled["talc"].astype(np.uint8), (ew, eh),
                            interpolation=cv2.INTER_NEAREST) > 0
    phase_small = masks.phase_label_map(sulfide_small, magnetite_small)
    # confidence MAP for the editor overlay only (single downscaled pass);
    # low_conf_zones/undetermined_fraction above use _run_panorama's finer,
    # per-tile aggregation instead of this call's own (coarser) values.
    report(0.88, "карта уверенности для редактора")
    unc = masks.uncertainty_for_editor(edit, cfg)

    report(0.93, "построение карт")
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
        "low_conf_zones": run["low_conf_zones"],
        "overlay_url": f"/api/images/{jid}.jpg",
        "n_ore": run["n_ore"], "n_tiles": run["n_tiles"],
        "ore_source": run["ore_source"],
    }
