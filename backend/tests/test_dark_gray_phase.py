"""Dark-gray (talc-like) area proxy — a graceful fallback for the talc share
where the darkness segmenter over/under-fires (borrowed heuristic).

Contract: it selects GREY (low saturation/chroma), MEDIUM-DARK (lower-middle of
the dynamic range, above the black bottom, below the bright-sulfide top), and
never the drawn annotation — full matching area counts, whether it's scattered
specks or one large contiguous mass (real talc can be either).
"""
import cv2
import numpy as np

from app.shlif import talc
from app.pipeline import loader

CFG = loader.get_config()


def test_dark_gray_phase_keeps_specks_drops_extremes_and_marks():
    h = w = 256
    rgb = np.full((h, w, 3), 200, np.uint8)          # bright grey matrix (excluded, the bulk)
    rgb[0:26] = 5                                     # ~10% black background (excluded)
    rgb[26:52] = 250                                  # ~10% bright sulfide (excluded)

    speck_pts = []
    for i in range(6):
        for j in range(6):
            y, x = 70 + i * 24, 20 + j * 36
            rgb[y:y + 6, x:x + 6] = 100              # scattered medium-dark grey specks (talc)
            speck_pts.append((y, x))
    cv2.line(rgb, (10, 240), (240, 240), (25, 55, 230), 5)  # blue annotation, in the matrix

    mask, frac = talc.dark_gray_phase(rgb, CFG.talc)

    assert not mask[0:26].any()          # black excluded
    assert not mask[26:52].any()         # bright sulfide excluded
    assert not mask[talc.blue_line_mask(rgb, CFG.talc)].any()  # annotation excluded
    # the specks are picked up
    hit = sum(mask[y:y + 6, x:x + 6].any() for y, x in speck_pts)
    assert hit >= len(speck_pts) * 0.8
    assert 0.0 < frac <= 0.12


def test_dark_gray_phase_counts_a_single_large_solid_blob_in_full():
    # a real talc-rich sample is often one big contiguous mass, not specks —
    # the share must track its true area, not collapse to 0.
    h = w = 256
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[0:38] = 10                        # ~15% black bottom of range
    rgb[38:140] = 90                      # ~40% solid medium-dark blob (the talc mass)
    rgb[140:] = 220                       # ~45% bright matrix
    mask, frac = talc.dark_gray_phase(rgb, CFG.talc)
    assert mask[38:140].all()
    assert not mask[0:38].any()
    assert not mask[140:].any()
    assert frac == 102 / 256  # rows 38:140, full credit
