"""Core-crop reconstruction: each tile contributes only its non-overlapping
core when reassembling one continuous canvas from overlapping tiles, so
summing every tile's core covers the canvas exactly once (no gap, no
double count from the overlap band).

axis_core_bounds derives each tile's core end from the *actual* sequence of
tile starts iter_tiles will yield along that axis (including its tw/th < 8
tail-tile skip) -- not from re-deriving "is this the last tile" locally
from a single tile's own clipped width, which cannot disambiguate a
genuinely last tile from an earlier tile whose data also happens to reach
the edge (this happens whenever `tile` is more than one `step` larger than
the canvas remainder -- a common case with this project's tile/overlap
ratio, not a rare corner case)."""
import numpy as np

from app.shlif.tiling import axis_core_bounds, axis_tile_starts


def test_axis_tile_starts_matches_naive_loop_with_tail_filter():
    # naive replica of iter_tiles' 1-D loop + tail-tile skip, for comparison
    size, tile, step = 2000, 320, 256
    expected = []
    for x in range(0, max(1, size - 1), step):
        if min(tile, size - x) < 8:
            continue
        expected.append(x)
    assert axis_tile_starts(size, tile, step) == expected


def test_axis_core_bounds_last_tile_extends_to_true_edge():
    bounds = axis_core_bounds(2000, 320, 256)
    last_start = max(bounds)
    assert bounds[last_start] == 2000


def test_axis_core_bounds_handles_multiple_tail_tiles_reaching_the_edge():
    # W=1800 with tile=320/step=256: BOTH x=1536 (tw=264) and x=1792 (tw=8)
    # independently have their raw pixel data reach the true edge (x+tw>=W)
    # -- exactly the case a per-tile-local "is this last?" check cannot
    # disambiguate. axis_core_bounds must still give exactly-contiguous,
    # non-overlapping cores.
    W, tile, overlap = 1800, 320, 64
    step = tile - overlap
    bounds = axis_core_bounds(W, tile, step)
    starts = sorted(bounds)
    for i, s in enumerate(starts):
        expected_end = starts[i + 1] if i + 1 < len(starts) else W
        assert bounds[s] == expected_end


def test_full_grid_reconstruction_has_no_gap_or_overlap():
    # sweep several (W, H) pairs, including ones where the stride does not
    # evenly divide the canvas and ones where multiple tail tiles reach the
    # edge (e.g. W=1800) -- the property must hold for every size, not one
    # hand-picked pair.
    tile, overlap = 320, 64
    step = tile - overlap
    for W, H in [(2000, 1500), (1800, 1500), (1801, 1499), (2049, 2049), (640, 640)]:
        x_bounds = axis_core_bounds(W, tile, step)
        y_bounds = axis_core_bounds(H, tile, step)
        canvas = np.zeros((H, W), np.int32)
        for y, y1 in y_bounds.items():
            for x, x1 in x_bounds.items():
                canvas[y:y1, x:x1] += 1
        assert (canvas == 1).all(), f"gap/overlap for W={W}, H={H}"


def test_production_tile_overlap_config_has_no_gap_or_overlap_across_many_widths():
    # the real config (default.yaml): tile=1024, overlap=128 -- sweep a wide
    # range of widths so we don't rely on one dimension happening to avoid
    # the defect this reconstruction must rule out for every image size.
    tile, overlap = 1024, 128
    step = tile - overlap
    for W in range(2000, 6000, 137):  # arbitrary irregular stride, broad coverage
        bounds = axis_core_bounds(W, tile, step)
        canvas = np.zeros(W, np.int32)
        for x, x1 in bounds.items():
            canvas[x:x1] += 1
        assert (canvas == 1).all(), f"gap/overlap for W={W}"
