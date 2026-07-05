"""segment_phases's magnetite gate must not reject true magnetite for being
cool-toned (negative Lab b) -- the `not_olive` assumption it used to encode
(olive/warm hue = matrix) is backwards for this material: LumenStone ground
truth and the project's own labelled images show olive = sulfide, and real
magnetite skews cool (negative b), not warm."""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

from app.shlif.config import load_config
from app.shlif.segment import compute_levels, segment_phases

CFG = load_config()


def test_cool_toned_mid_grey_patch_is_magnetite_not_matrix():
    rng = np.random.default_rng(0)
    img = rng.integers(15, 35, (256, 256, 3)).astype(np.uint8)  # dark matrix background
    img[40:110, 40:110] = (220, 220, 220)     # bright neutral block -> sulfide
    img[150:210, 150:210] = (100, 108, 116)   # cool-toned mid-grey -> should be magnetite
                                                # (L=45.2, a=-1.3, b=-5.5, chroma=5.6 --
                                                # b is below the old green_b_min=-4.0 floor,
                                                # which used to misclassify this as matrix)

    seg = segment_phases(img, CFG.segment)

    assert seg.magnetite[150:210, 150:210].mean() > 0.9
    assert seg.sulfide[40:110, 40:110].mean() > 0.9


def _textured_gray(seed, base, amp, sigma, size):
    """A spatially-correlated (not white-noise) grey patch -- real rock texture
    has grain but is locally smooth, unlike iid-per-pixel randint. White noise
    also breaks `_clean`'s pinhole-fill into an all-or-nothing mask, which
    would misrepresent this scenario."""
    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(0, 1, size).astype(np.float32), sigma=sigma)
    noise = noise / (np.abs(noise).max() + 1e-6) * amp
    gray = np.clip(base + noise, 0, 255)
    jitter = np.random.default_rng(seed + 100).normal(0, 1.5, size + (3,)).astype(np.float32)
    return np.clip(gray[..., None] + jitter, 0, 255).astype(np.uint8)


def test_matrix_only_tile_is_not_invented_into_magnetite_by_floating_otsu():
    """A panorama tile that is ENTIRELY matrix (no real sulfide/magnetite) has no
    genuine trimodal brightness distribution, but segment_phases's per-call
    3-class Otsu will still force a split out of the tile's own texture noise --
    and reads it as ore. Anchoring `levels` to a wider reference that actually
    contains the other two phases must suppress this (the panorama pipeline's
    fix for "everything paints as magnetite" on ore-free sections)."""
    matrix_tile = _textured_gray(1, base=45, amp=15, sigma=4, size=(512, 512))

    floating = segment_phases(matrix_tile, CFG.segment)
    assert floating.fractions["magnetite"] + floating.fractions["sulfide"] > 0.3  # reproduces the bug

    matrix_ref = _textured_gray(2, base=45, amp=15, sigma=4, size=(430, 600))
    magnetite_ref = _textured_gray(3, base=110, amp=10, sigma=4, size=(120, 600))
    sulfide_ref = _textured_gray(4, base=210, amp=10, sigma=4, size=(50, 600))
    wide_section = np.concatenate([sulfide_ref, magnetite_ref, matrix_ref], axis=0)
    levels = compute_levels(rgb2lab(wide_section)[..., 0], float(CFG.segment.bright_percentile))

    anchored = segment_phases(matrix_tile, CFG.segment, levels=levels)
    assert anchored.fractions["matrix"] > 0.95
