"""segment_phases's magnetite gate must not reject true magnetite for being
cool-toned (negative Lab b) -- the `not_olive` assumption it used to encode
(olive/warm hue = matrix) is backwards for this material: LumenStone ground
truth and the project's own labelled images show olive = sulfide, and real
magnetite skews cool (negative b), not warm."""
import numpy as np

from app.shlif.config import load_config
from app.shlif.segment import segment_phases

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
