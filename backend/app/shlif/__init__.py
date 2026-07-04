"""Шлиф — automatic ore classification from reflected-light optical microscopy.

Public API:
    from shlif import load_config, analyze_image, segment_phases
"""

from __future__ import annotations

from . import metrics, phases
from .analyze import Analysis, analyze_image
from .config import Config, load_config
from .imageio import (
    annotated_talc_pairs,
    list_class_images,
    list_panoramas,
    load_rgb,
)
from .overlay import colorize_intergrowth, colorize_phases
from .segment import SegResult, segment_phases
from .talc import detect_talc, talc_fraction, talc_mask_from_contours
from .tiling import Tile, iter_tiles

__all__ = [
    "Analysis",
    "Config",
    "SegResult",
    "Tile",
    "analyze_image",
    "annotated_talc_pairs",
    "colorize_intergrowth",
    "colorize_phases",
    "detect_talc",
    "iter_tiles",
    "list_class_images",
    "list_panoramas",
    "load_config",
    "load_rgb",
    "metrics",
    "phases",
    "segment_phases",
    "talc_fraction",
    "talc_mask_from_contours",
]
