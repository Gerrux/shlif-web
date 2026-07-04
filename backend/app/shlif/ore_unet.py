"""Trained ore/matrix U-Net (``models/unet_ore.pt``) for the panorama ore gate.

Binary ore-vs-matrix segmentation (IoU 0.975 vs the classical multi-Otsu+Lab
segmenter's 0.81 on LumenStone), illumination-invariant by construction --
trained on gray-world-WB + CLAHE-normalised tiles, so an over/under-exposed
capture maps to the same decision (unlike ``segment_phases``'s per-image-
relative Otsu split -- see ``uncertainty.py`` for how that instability is
flagged for the finer magnetite/sulfide split this model does NOT make).

Ported from ``hakaton_nornikel/scripts/sam2_prelabel.py::build_unet`` /
``unet_ore_decision``. Guarded import: returns ``None`` when the checkpoint
or torch/segmentation_models_pytorch are unavailable, so CPU-only/no-model
machines fall back to the classical ``segment_phases`` cleanly.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ORE_CKPT = "unet_ore.pt"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_ore_unet(ckpt: str = ORE_CKPT, device: str | None = None):
    """Load the trained ore/matrix U-Net -> ``(model, device)``, or ``None``.

    Returns ``None`` when the checkpoint file is missing or torch/smp fail to
    import or load -- the caller then keeps the classical ``segment_phases``.
    """
    if not Path(ckpt).exists():
        return None
    try:
        import segmentation_models_pytorch as smp
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=2)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        return model.to(dev).eval(), dev
    except Exception:
        return None


def ore_unet_mask(rgb: np.ndarray, model, device: str, tile: int = 512,
                   batch_size: int = 32) -> np.ndarray:
    """Bool (H, W): True = ore (sulfide+magnetite), tiled U-Net inference.

    Applies gray-world WB + CLAHE per sub-tile before the ImageNet
    normalisation -- IDENTICAL to training (``wb_clahe``). This MUST stay on
    for this checkpoint (unlike the talc U-Net, which trained on raw RGB).

    All under-tile crops are stacked into as few forward passes as
    ``batch_size`` allows (default 32 -- a typical 2048px panorama tile at
    tile=512 is 16 crops, comfortably one batch), instead of one model call
    per crop.
    """
    import torch

    from .preprocess import wb_clahe

    H, W = rgb.shape[:2]
    ore = np.zeros((H, W), bool)

    coords, dims, crops = [], [], []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            crop = rgb[y:y + tile, x:x + tile]
            ch, cw = crop.shape[:2]
            cp = cv2.copyMakeBorder(crop, 0, tile - ch, 0, tile - cw, cv2.BORDER_REFLECT)
            cp = wb_clahe(cp)
            t = ((cp.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)
            coords.append((y, x))
            dims.append((ch, cw))
            crops.append(t)

    if not crops:
        return ore

    batch = torch.from_numpy(np.stack(crops)).to(device)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            chunk = batch[start:start + batch_size]
            preds = model(chunk).argmax(1).cpu().numpy()
            for i, p in enumerate(preds):
                (y, x), (ch, cw) = coords[start + i], dims[start + i]
                ore[y:y + ch, x:x + cw] = (p[:ch, :cw] != 0)
    return ore
