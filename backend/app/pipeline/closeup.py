from __future__ import annotations
import cv2
import numpy as np
from app.shlif import analyze_image
from app.shlif.features import extract_features
from app.shlif.talc_unet import talc_unet_mask
from app.shlif.uncertainty import ensemble_uncertainty, find_low_conf_zones
from app.pipeline import masks, loader

_UNC_MAX_SIDE = 1024  # cap the ensemble-segmentation resolution — the fraction is scale-robust

def _sort_card(rgb, cfg):
    bundle = loader.load_classifier()
    if bundle is None:
        return None
    clf, feat, classes = bundle
    feats = extract_features(rgb, cfg)
    proba = clf.predict_proba(np.array([[feats[k] for k in feat]], float))[0]
    probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return {"classes": probs, "top": max(probs, key=lambda k: probs[k])}

def _uncertainty(rgb: np.ndarray, cfg) -> dict:
    """Ensemble-perturbation uncertainty, computed on a downscaled copy for speed
    and the confidence map resized back to the full frame."""
    h, w = rgb.shape[:2]
    s = min(1.0, _UNC_MAX_SIDE / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb
    u = ensemble_uncertainty(small, cfg)
    conf = cv2.resize(u["confidence"], (w, h), interpolation=cv2.INTER_LINEAR)
    return {"confidence": conf, "undetermined_fraction": u["undetermined_fraction"],
            "low_conf_zones": find_low_conf_zones(u)}


def analyze_closeup(rgb: np.ndarray, cfg) -> dict:
    """Uses the trained talc U-Net when its weights are loadable (GPU or CPU);
    falls back to the classical darkness-based talc seed when they aren't."""
    unet = loader.load_talc_unet()
    if unet is not None:
        model, device = unet
        talc_mask = talc_unet_mask(rgb, model, device, thr=None)
        res = analyze_image(rgb, cfg, talc_mask=talc_mask)
    else:
        res = analyze_image(rgb, cfg, detect_talc_flag=True)  # classical talc seed
    m = res.masks
    phase_map = masks.phase_label_map(m["sulfide"], m["magnetite"])

    unc = _uncertainty(rgb, cfg)
    metrics = dict(res.metrics)
    metrics["undetermined_fraction"] = unc["undetermined_fraction"]

    return {
        "verdict": {"ore_class": res.ore_class, "text": res.text, "metrics": metrics},
        "sort": _sort_card(rgb, cfg),
        "phase_map": phase_map,
        "talc": m["talc"].astype(bool),
        "superpixels": masks.build_superpixel_map(rgb),
        "darkness": masks.build_darkness_map(rgb),
        "confidence": unc["confidence"],
        "low_conf_zones": unc["low_conf_zones"],
        "text": res.text,
    }
