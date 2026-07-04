"""Linear feather weight for seamless tile stitching (borrowed pattern).

A 2D weight that is 1 in the tile centre and ramps down over a margin at each
edge, so overlapping tiles blend smoothly into the panorama overlay instead of
hard-seaming (or double-darkening the overlap band)."""
import pytest

from app.shlif import tiling


def test_tile_blend_weight_center_high_edges_low():
    wgt = tiling.tile_blend_weight(100, 200, margin_frac=0.2)
    assert wgt.shape == (100, 200)
    assert wgt.max() == pytest.approx(1.0)
    assert wgt[50, 100] == pytest.approx(1.0)     # centre = 1
    assert wgt[0, 100] < 0.5                       # top edge faded
    assert wgt[50, 0] < 0.5                        # left edge faded
    assert wgt[0, 0] < wgt[50, 100]                # corner below centre
    assert (wgt > 0).all()                         # strictly positive → safe to normalise


def test_tile_blend_weight_handles_tiny_tiles():
    wgt = tiling.tile_blend_weight(3, 3, margin_frac=0.12)
    assert wgt.shape == (3, 3)
    assert (wgt > 0).all()
