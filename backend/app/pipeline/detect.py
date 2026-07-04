"""Automatic close-up/panorama routing by image size — no manual mode toggle.

A close-up is a single field of view (a few MP); a panorama is a stitched
whole-section scan (100+ MP in this dataset). There is a wide, empty gap
between the two in practice (see the design spec), so a single pixel-count
threshold cleanly separates them.
"""

from __future__ import annotations

from app.shlif.imageio import image_size


def detect_mode(width: int, height: int, cfg) -> str:
    """"closeup" (single pass) or "panorama" (tiled) from raw pixel count."""
    return "panorama" if width * height > int(cfg.tiling.direct_max_pixels) else "closeup"


def detect_mode_from_path(path: str, cfg) -> str:
    """Read just the image header (no full decode) and classify it."""
    w, h = image_size(path)
    return detect_mode(w, h, cfg)
