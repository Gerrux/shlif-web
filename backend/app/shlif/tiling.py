"""Memory-safe tiling for gigapixel panoramas.

JPEG has no random access, so we decode the whole image once at a *working
scale* that fits ``max_pixels`` (via draft mode), then iterate overlapping
native-scale tiles over that array. Empty (matrix-only) tiles are skipped so the
time budget is spent only where there is ore.

For a true no-full-decode path, swap :func:`load_rgb` for pyvips region reads —
the tile loop below is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .imageio import decode_factor, image_size, load_rgb


@dataclass
class Tile:
    rgb: np.ndarray   # tile pixels (working scale)
    x: int            # left in working-scale coords
    y: int            # top in working-scale coords
    factor: int       # working-scale reduction vs native (1 = native)
    empty: bool       # flagged as matrix-only and skipped by the caller


def _edge_ramp(n: int, m: int) -> np.ndarray:
    """1-D window: ramps (0,1] over ``m`` px at each end, flat 1 in the middle.
    Never reaches 0, so a stitch can normalise by the summed weight safely."""
    x = np.ones(n, np.float32)
    m = min(m, n // 2)
    if m > 0:
        ramp = (np.arange(m, dtype=np.float32) + 1.0) / (m + 1.0)
        x[:m] = ramp
        x[-m:] = ramp[::-1]
    return x


def tile_blend_weight(h: int, w: int, margin_frac: float = 0.12) -> np.ndarray:
    """2-D linear feather weight (``h×w``): 1 in the centre, ramping to ~0 over a
    ``margin_frac`` margin at each edge. Used to blend overlapping tiles into the
    panorama overlay seamlessly (borrowed feather pattern) — accumulate
    ``weight*colour`` and divide by summed weight."""
    return np.outer(_edge_ramp(h, int(round(h * margin_frac))),
                    _edge_ramp(w, int(round(w * margin_frac))))


def _is_empty(rgb: np.ndarray, bright_frac: float) -> bool:
    v = rgb.max(axis=2)
    thr = max(40, int(v.mean() + 2.0 * v.std()))
    return float((v > thr).mean()) < bright_frac


def load_working_array(path: str | Path, cfg) -> np.ndarray:
    """Decode the image once at the tiling working scale (memory-safe draft
    decode above ``cfg.max_pixels``). Shared by `iter_tiles` and any caller
    that also needs the full working-scale canvas (e.g. a display copy), so
    a gigapixel file is only ever decoded once per job."""
    return load_rgb(path, max_pixels=int(cfg.max_pixels))


def iter_tiles(path: str | Path, cfg, arr: np.ndarray | None = None) -> Iterator[Tile]:
    """Yield overlapping tiles across a (possibly gigapixel) image.

    ``cfg`` is the ``tiling`` config block. Empty tiles are yielded with
    ``empty=True`` (and no heavy work done) unless ``skip_empty`` is false.
    Pass a pre-loaded ``arr`` (from :func:`load_working_array`) to avoid
    decoding the image twice when the caller also needs the full canvas.
    """
    w, h = image_size(path)
    factor = decode_factor(w, h, int(cfg.max_pixels))
    if arr is None:
        arr = load_working_array(path, cfg)
    H, W = arr.shape[:2]

    tile = int(cfg.tile)
    step = max(1, tile - int(cfg.overlap))
    skip_empty = bool(cfg.skip_empty)
    bright_frac = float(cfg.empty_bright_frac)

    for y in range(0, max(1, H - 1), step):
        for x in range(0, max(1, W - 1), step):
            sub = arr[y : y + tile, x : x + tile]
            if sub.shape[0] < 8 or sub.shape[1] < 8:
                continue
            empty = skip_empty and _is_empty(sub, bright_frac)
            yield Tile(rgb=sub, x=x, y=y, factor=factor, empty=empty)


def tile_grid(path: str | Path, cfg) -> tuple[int, int, int]:
    """(working_width, working_height, factor) — for allocating a stitch canvas."""
    w, h = image_size(path)
    factor = decode_factor(w, h, int(cfg.max_pixels))
    return w // factor, h // factor, factor


def tile_core_bounds(x: int, y: int, tw: int, th: int, step: int, W: int, H: int) -> tuple[int, int, int, int]:
    """The non-overlapping "core" region a tile contributes when reassembling
    one continuous canvas from overlapping tiles: ``[x, x+step)`` on each
    axis, except the last tile in a row/column, which extends all the way to
    the true canvas edge (the stride does not have to evenly divide the
    canvas). Consecutive tiles' cores are exactly contiguous (the next tile
    always starts at ``x+step``), so summing every tile's core covers the
    canvas once, with no gap and no overlap."""
    cx1 = W if x + tw >= W else min(x + step, W)
    cy1 = H if y + th >= H else min(y + step, H)
    return x, y, cx1, cy1
