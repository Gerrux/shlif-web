"""_assemble_masks tiles a section, segments/talc-detects each tile, and
reassembles one continuous mask via core-crop — every pixel must land in
exactly one phase (no gap from the last-tile edge, no double count from
the overlap band)."""
import copy

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

from app.pipeline import loader, panorama
from app.pipeline.masks import EDIT_MAX_SIDE, fit_max_side
from app.shlif.imageio import load_rgb
from app.shlif.preprocess import preprocess
from app.shlif.segment import compute_levels


def _synthetic_section():
    rng = np.random.default_rng(3)
    img = rng.integers(8, 30, (1200, 2400, 3)).astype(np.uint8)  # dark matrix
    # Bright sulfide blob straddling the x=448 core boundary that
    # tile=512/overlap=64 produces (step=448, so core ends fall at
    # 0/448/896/...) -- sized/positioned (not the naive [100:400,100:400])
    # because a bigger/higher-contrast patch pushes `tiling._is_empty`'s
    # adaptive threshold (mean + 2*std) above 255 on this bimodal image,
    # which misflags the whole tile as empty (a pre-existing quirk of
    # `_is_empty`, unrelated to `_assemble_masks` and out of scope here) and
    # would hide the very seam behaviour this test checks.
    img[100:300, 350:550] = 220
    # Mid-grey magnetite blob straddling the y=896 core boundary instead (x
    # safely inside the [896,1344) x-core so only the y-seam is exercised
    # here). Value 60, not the naive 120: 120 converts to Lab L high enough
    # that segment_phases classifies it as sulfide, not magnetite.
    #
    # IMPORTANT: segment_phases runs PER TILE inside _assemble_masks (each
    # tile gets its own independent 3-class Otsu split), not once over the
    # whole image -- verified directly against the real per-tile path, not
    # just a whole-image segment_phases() call (an earlier attempt at this
    # fixture was wrongly validated that way and passed only by accident).
    # A tile containing just dark background + ONE brighter blob is
    # effectively bimodal, and 3-class Otsu on a bimodal population reliably
    # puts the blob in the brightest ("sulfide") band regardless of its
    # absolute value -- there's no genuine "middle" population for it to
    # land in. Getting a real magnetite (middle-band) classification requires
    # a truly trimodal histogram within that same tile: dark matrix + this
    # mid-grey blob + something distinctly brighter still. The small sulfide
    # anchor below supplies that third population (placed in the y=[896,960)
    # overlap band shared by both tiles this blob straddles, so one anchor
    # serves both). Verified empirically: with the anchor present, 60
    # classifies as ~100% magnetite in the blob region across both tiles,
    # and both tiles stay comfortably non-empty under tiling._is_empty
    # (bright_frac ~0.005-0.007, threshold is 0.002).
    img[800:1000, 1000:1200] = 60
    img[890:930, 890:930] = 220  # sulfide anchor -- gives the two tiles the
    # magnetite blob straddles a real trimodal histogram (see note above);
    # not itself asserted on, it only exists to make Otsu's split meaningful
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

    # the seeded bright blob (which straddles an x-tile boundary at this tile
    # size) must still be picked up as sulfide, not lost at the seam
    assert assembled["sulfide"][100:300, 350:550].mean() > 0.5

    # the seeded mid-grey blob (which straddles a y-tile boundary) must
    # still be picked up as magnetite, not lost at the seam
    assert assembled["magnetite"][800:1000, 1000:1200].mean() > 0.5


def _textured_gray(seed, base, amp, sigma, size):
    """Spatially-correlated (not white-noise) grey patch, standing in for real
    rock texture: grainy but locally smooth, unlike iid-per-pixel randint."""
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(0, 1, size).astype(np.float32), sigma=sigma)
    noise = noise / (np.abs(noise).max() + 1e-6) * amp
    gray = np.clip(base + noise, 0, 255)
    jitter = np.random.default_rng(seed + 100).normal(0, 1.5, size + (3,)).astype(np.float32)
    return np.clip(gray[..., None] + jitter, 0, 255).astype(np.uint8)


def test_assemble_masks_with_anchor_keeps_ore_free_tiles_matrix(tmp_path):
    """Regression for "everything paints as magnetite on the panorama": a
    section that is mostly textured matrix, with one genuine ore patch tucked
    in a corner, tiled at 2048px (analyze_panorama's forced tile size). Without
    an anchor, each matrix-only tile refits its own 3-class Otsu on nothing but
    its own texture noise and reads a big chunk of itself as magnetite/sulfide
    (see test_segment.py). Anchoring `levels` to a section-wide reference (as
    analyze_panorama now does) must keep those tiles matrix -- and this must
    hold with CLAHE left ON in cfg.preprocess (the default), since per-tile
    CLAHE independently re-stretches each tile's contrast and would otherwise
    defeat the anchor (see `panorama._segmentation_preprocess_cfg`)."""
    H, W = 2400, 3000
    img = _textured_gray(1, base=45, amp=15, sigma=4, size=(H, W))
    img[300:900, 300:1200] = _textured_gray(9, base=170, amp=15, sigma=4, size=(600, 900))
    img[300:500, 300:1200] = _textured_gray(10, base=225, amp=10, sigma=4, size=(200, 900))
    p = tmp_path / "mixed_section.jpg"
    Image.fromarray(img).save(p, "JPEG", quality=95)

    cfg = copy.deepcopy(loader.get_config())
    cfg.tiling.tile = 2048
    assert cfg.preprocess.clahe  # the scenario this guards against needs CLAHE on

    arr = load_rgb(str(p), max_pixels=int(cfg.tiling.max_pixels))
    anchor_rgb = fit_max_side(arr, EDIT_MAX_SIDE, cv2.INTER_AREA)
    seg_pre_cfg = panorama._segmentation_preprocess_cfg(cfg)
    anchor_L = rgb2lab(preprocess(anchor_rgb, seg_pre_cfg))[..., 0]
    levels = compute_levels(anchor_L, float(cfg.segment.bright_percentile))

    assembled = panorama._assemble_masks(str(p), cfg, arr, levels=levels)

    # far from the ore patch (a different 2048px tile, entirely matrix): must
    # stay matrix, not fabricate magnetite/sulfide out of the tile's own
    # texture noise
    far = assembled["magnetite"][2050:2350, 2200:2900] | assembled["sulfide"][2050:2350, 2200:2900]
    assert far.mean() < 0.05

    # the real ore patch itself must still be picked up
    assert assembled["magnetite"][300:900, 300:1200].mean() > 0.3
