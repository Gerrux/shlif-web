"""_assemble_masks tiles a section, segments/talc-detects each tile, and
reassembles one continuous mask via core-crop — every pixel must land in
exactly one phase (no gap from the last-tile edge, no double count from
the overlap band)."""
import copy

import numpy as np
from PIL import Image

from app.pipeline import loader, panorama
from app.shlif.imageio import load_rgb


def _synthetic_section():
    rng = np.random.default_rng(3)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)  # dark matrix
    # Bright sulfide blob straddling the x=448 core boundary that
    # tile=512/overlap=64 produces (step=448, so core ends fall at
    # 0/448/896/...) — sized at 200x200 (not the originally-drafted 300x300)
    # because a bigger/higher-contrast patch pushes `_is_empty`'s adaptive
    # threshold (mean + 2*std) above 255 on this bimodal image, which
    # misflags the whole tile as empty (a pre-existing quirk of
    # `tiling._is_empty`, unrelated to `_assemble_masks` and out of scope
    # here) and would hide the very seam behaviour this test checks.
    img[100:300, 350:550] = 220
    img[600:900, 1400:1900] = 120   # mid-grey magnetite blob, straddles tile seams below
    return img


def test_assemble_masks_partitions_every_pixel_exactly_once(tmp_path):
    img = _synthetic_section()
    p = tmp_path / "section.jpg"
    Image.fromarray(img).save(p, "JPEG", quality=95)

    cfg = copy.deepcopy(loader.get_config())
    cfg.tiling.tile = 512
    cfg.tiling.overlap = 64  # forces multiple tiles over the 1200x2400 image

    arr = load_rgb(str(p), max_pixels=int(cfg.tiling.max_pixels))
    assembled = panorama._assemble_masks(str(p), cfg, arr)

    total = (assembled["sulfide"].astype(np.int32) + assembled["magnetite"].astype(np.int32)
             + assembled["matrix"].astype(np.int32))
    assert total.shape == arr.shape[:2]
    assert (total == 1).all()  # exactly one phase per pixel — no gap, no double-write

    # the seeded bright blob (which straddles a tile boundary at this tile size)
    # must still be picked up as sulfide, not lost at the seam
    assert assembled["sulfide"][100:300, 350:550].mean() > 0.5
