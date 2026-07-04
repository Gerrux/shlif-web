from __future__ import annotations
import numpy as np
from app.shlif import analyze_image
from app.shlif.features import extract_features
from app.pipeline import masks, loader

def _sort_card(rgb, cfg):
    bundle = loader.load_classifier()
    if bundle is None:
        return None
    clf, feat, classes = bundle
    feats = extract_features(rgb, cfg)
    proba = clf.predict_proba(np.array([[feats[k] for k in feat]], float))[0]
    probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return {"classes": probs, "top": max(probs, key=lambda k: probs[k])}

def analyze_closeup(rgb: np.ndarray, cfg) -> dict:
    """Classical/CPU path (GPU U-Net wiring is added later behind loader.gpu_available)."""
    res = analyze_image(rgb, cfg, detect_talc_flag=True)  # classical talc seed
    m = res.masks
    phase_map = masks.phase_label_map(m["sulfide"], m["magnetite"])
    return {
        "verdict": {"ore_class": res.ore_class, "text": res.text, "metrics": res.metrics},
        "sort": _sort_card(rgb, cfg),
        "phase_map": phase_map,
        "talc": m["talc"].astype(bool),
        "superpixels": masks.build_superpixel_map(rgb),
        "darkness": masks.build_darkness_map(rgb),
        "text": res.text,
    }
