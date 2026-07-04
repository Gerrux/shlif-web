"""Core-crop reconstruction: each tile contributes only its non-overlapping
stride when reassembling one continuous canvas from overlapping tiles, so
summing every tile's core covers the canvas exactly once (no gap, no
double count from the overlap band)."""
import numpy as np

from app.shlif.tiling import tile_core_bounds


def test_tile_core_bounds_middle_tile_is_step_sized():
    x, y, x1, y1 = tile_core_bounds(x=256, y=0, tw=320, th=320, step=192, W=2000, H=2000)
    assert (x, y, x1, y1) == (256, 0, 448, 192)


def test_tile_core_bounds_last_tile_extends_to_true_edge():
    # this tile's pixel data reaches the canvas edge (x + tw >= W) -> no next
    # tile exists to claim the remainder, so its core must cover it
    x, y, x1, y1 = tile_core_bounds(x=1800, y=0, tw=200, th=320, step=192, W=2000, H=2000)
    assert (x1, y1) == (2000, 192)


def test_full_grid_reconstruction_has_no_gap_or_overlap():
    W, H, tile, overlap = 2000, 1500, 320, 64
    step = tile - overlap
    canvas = np.zeros((H, W), np.int32)
    for y in range(0, max(1, H - 1), step):
        for x in range(0, max(1, W - 1), step):
            tw, th = min(tile, W - x), min(tile, H - y)
            if tw < 8 or th < 8:
                continue
            cx0, cy0, cx1, cy1 = tile_core_bounds(x, y, tw, th, step, W, H)
            canvas[cy0:cy1, cx0:cx1] += 1
    assert (canvas == 1).all()  # every pixel covered exactly once
