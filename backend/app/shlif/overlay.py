"""Colour overlays in the design-system "Шлиф" phase palette."""

from __future__ import annotations

import numpy as np

from . import phases


def colorize_phases(
    rgb: np.ndarray,
    seg,
    talc: np.ndarray | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend phase masks over the image: sulfide=brass, magnetite=steel, talc=blue."""
    out = rgb.astype(np.float32).copy()
    layers = [
        (seg.sulfide, phases.COLOR_SULFIDE),
        (seg.magnetite, phases.COLOR_MAGNETITE),
    ]
    if talc is not None:
        layers.append((talc, phases.COLOR_TALC))
    for mask, color in layers:
        if mask is None or not mask.any():
            continue
        c = np.array(color, dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)


def colorize_intergrowth(
    rgb: np.ndarray,
    normal_mask: np.ndarray,
    fine_mask: np.ndarray,
    talc: np.ndarray | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """The final verdict overlay: green=обычные, red=тонкие, blue=тальк."""
    out = rgb.astype(np.float32).copy()
    layers = [
        (normal_mask, phases.COLOR_NORMAL),
        (fine_mask, phases.COLOR_FINE),
    ]
    if talc is not None:
        layers.append((talc, phases.COLOR_TALC))
    for mask, color in layers:
        if mask is None or not mask.any():
            continue
        c = np.array(color, dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * c
    return np.clip(out, 0, 255).astype(np.uint8)
