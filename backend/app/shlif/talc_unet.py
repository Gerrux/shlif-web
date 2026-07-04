"""Trained talc-zone U-Net (``out/unet_talc.pt``) as a talc detector for the demo.

This is the SAME 1-class binary talc-segmentation model the talc annotator
(``annotate_talc.py``) uses as its U-Net suggestion layer, lifted out into a
shared, device-agnostic, in-memory helper so ``app.py`` can run it on close-ups
and panorama tiles for the actual talc verdict — not just as an annotation hint.

Trained by ``scripts/train_talc_unet.py``: resnet34 encoder, whole tile resized
to 512, ImageNet normalisation, single sigmoid head (Dice + BCE). Inference here
mirrors ``annotate_talc.unet_mask`` exactly — a single whole-image resize to 512
(matching the training scale), sigmoid, resize back, threshold. It deliberately
does **not** apply gray-world WB + CLAHE the way the ore U-Net's
``unet_ore_decision`` does: the talc model was trained on raw RGB, so WB/CLAHE
would shift the input distribution away from training.

``build_talc_unet`` guards on the checkpoint existing and returns ``None`` when it
is absent (or torch/smp cannot load), so machines without the weights or a GPU
fall back to the classical ``detect_talc`` cleanly — the local demo never breaks.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TALC_CKPT = "out/unet_talc.pt"
SZ = 512
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _use_amp(device) -> bool:
    """True when ``device`` names a CUDA device -- gates fp16 autocast; the
    CPU fallback path must stay plain fp32."""
    return str(device).startswith("cuda")


def resolve_threshold(prob: np.ndarray, lo: float = 0.05, hi: float = 0.30,
                      k95: float = 0.85, k99: float = 0.55) -> float:
    """Adaptive sigmoid threshold for a talc probability map, clamped to ``[lo, hi]``.

    ``max(p95*k95, p99*k99)`` keeps a weak-but-present talc signal from vanishing
    under a fixed 0.5 cut (the top percentiles survive) while the clamp stops noise
    from flooding (floor) or a strong map from over-shrinking (ceiling). Borrowed
    cascade — use where a non-empty-but-not-flooded mask matters more than a fixed
    operating point.
    """
    p95 = float(np.percentile(prob, 95))
    p99 = float(np.percentile(prob, 99))
    thr = max(p95 * k95, p99 * k99)
    return float(min(max(thr, lo), hi))


def build_talc_unet(ckpt: str = TALC_CKPT, device: str | None = None):
    """Load the trained talc U-Net → ``(model, device)``, or ``None`` if unavailable.

    ``device`` defaults to ``"cuda"`` when available else ``"cpu"``. The annotator
    hardcodes cuda; here we stay CPU-safe so the demo runs anywhere the checkpoint
    is present. Returns ``None`` when the checkpoint file is missing or torch/smp
    fail to import/load — the caller then keeps the classical ``detect_talc``.
    """
    if not Path(ckpt).exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        return model.to(dev).eval(), dev
    except Exception:
        return None


def talc_unet_mask(rgb: np.ndarray, model, device: str, thr: float | None = 0.5) -> np.ndarray:
    """Binary talc mask (bool HxW) from the trained U-Net: ``sigmoid >= thr``.

    Mirrors ``annotate_talc.unet_mask``: the whole image is resized to 512,
    ImageNet-normalised, run through the sigmoid head, resized back to native
    resolution and thresholded. No WB/CLAHE — the talc model was trained on raw
    RGB (unlike the ore U-Net's ``wb_clahe`` path). ``thr`` is a 0..1 fraction, or
    ``None`` to pick it adaptively from the map (:func:`resolve_threshold`).
    """
    import torch

    H, W = rgb.shape[:2]
    im = cv2.resize(rgb, (SZ, SZ)).astype(np.float32) / 255.0
    im = (im - _MEAN) / _STD
    x = torch.from_numpy(im.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    use_amp = _use_amp(device)
    with torch.inference_mode():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pr = torch.sigmoid(model(x))[0, 0]
        else:
            pr = torch.sigmoid(model(x))[0, 0]
        pr = pr.float().cpu().numpy()
    pr = cv2.resize(pr, (W, H))
    t = resolve_threshold(pr) if thr is None else float(thr)
    return pr >= t
