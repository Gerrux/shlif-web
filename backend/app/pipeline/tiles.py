"""Zoomable tile pyramid for the panorama viewer — built once from the
already-decoded working array (`arr` in `analyze_panorama`), before it's
discarded. Independent of the mask/verdict pipeline: takes a plain RGB
array and a job id, nothing more."""

from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image

from app.core import paths

TILE_SIZE = 256
JPEG_QUALITY = 82


def build_pyramid(arr: np.ndarray, jid: str) -> None:
    """Slice `arr` into a zoomable tile pyramid on disk:
    `data/tiles/{jid}/{level}/{col}_{row}.jpg` + `data/tiles/{jid}/manifest.json`.
    Level 0 is the lowest-resolution level (whole image fits in ~1 tile);
    `maxLevel` is `arr`'s own resolution — OpenSeadragon's own level
    numbering, so the frontend needs no translation."""
    h, w = arr.shape[:2]
    levels = [arr]
    while max(levels[-1].shape[:2]) > TILE_SIZE:
        prev = levels[-1]
        ph, pw = prev.shape[:2]
        nh, nw = max(1, (ph + 1) // 2), max(1, (pw + 1) // 2)
        levels.append(cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA))
    levels.reverse()  # levels[0] = smallest, levels[-1] = full resolution
    max_level = len(levels) - 1

    out_dir = paths.tiles_dir(jid)
    for level, level_arr in enumerate(levels):
        lh, lw = level_arr.shape[:2]
        level_dir = out_dir / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        for row, y in enumerate(range(0, lh, TILE_SIZE)):
            for col, x in enumerate(range(0, lw, TILE_SIZE)):
                tile = level_arr[y:y + TILE_SIZE, x:x + TILE_SIZE]
                Image.fromarray(tile).save(level_dir / f"{col}_{row}.jpg", "JPEG", quality=JPEG_QUALITY)

    (out_dir / "manifest.json").write_text(json.dumps({
        "width": w, "height": h, "tileSize": TILE_SIZE, "maxLevel": max_level,
    }))
