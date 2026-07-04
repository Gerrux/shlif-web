"""Illumination / colour normalisation.

The batches differ a lot in exposure (ч1 bright & olive, ч2 dark; panoramas
near-black). Normalising before thresholding is what lets one set of relative
thresholds work across all of them.
"""

from __future__ import annotations

import cv2
import numpy as np


def gray_world_white_balance(rgb: np.ndarray) -> np.ndarray:
    """Gray-world balance: scale each channel so its mean matches the global mean.

    Removes the olive/colour cast so neutral phases (magnetite) read as neutral.
    """
    img = rgb.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = means.mean()
    balanced = img * (gray / means)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def clahe_L(rgb: np.ndarray, clip: float = 0.01) -> np.ndarray:
    """CLAHE on the L channel of Lab — local contrast without colour shift."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clip_limit = max(1.0, clip * 255.0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def wb_clahe(rgb: np.ndarray, clahe_clip: float = 0.01) -> np.ndarray:
    """Fixed illumination normalisation (gray-world WB → CLAHE), applied IDENTICALLY
    in U-Net training and inference. This train/inference-matched normalisation is
    the dominant illumination-invariance lever for reflected-light OM (the model
    learns on normalised inputs, so different raw exposures map to the same answer).
    No config dependency, so training scripts can import it directly."""
    return clahe_L(gray_world_white_balance(rgb), clahe_clip)


def preprocess(rgb: np.ndarray, cfg) -> np.ndarray:
    """Apply the configured normalisation chain and return a uint8 RGB image."""
    out = rgb
    if cfg.get("white_balance", True):
        out = gray_world_white_balance(out)
    if cfg.get("denoise_median", 0):
        k = int(cfg["denoise_median"])
        if k >= 3 and k % 2 == 1:
            out = cv2.medianBlur(out, k)
    if cfg.get("clahe", True):
        out = clahe_L(out, float(cfg.get("clahe_clip", 0.01)))
    return out
