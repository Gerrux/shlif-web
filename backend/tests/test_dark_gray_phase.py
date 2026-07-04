"""Dispersed dark-gray (talc-like) area proxy — a graceful fallback for the talc
share where the darkness segmenter over/under-fires (borrowed heuristic).

Contract: it selects GREY (low saturation/chroma), MEDIUM-DARK (lower-middle of
the dynamic range, above the black bottom, below the bright-sulfide top),
DISPERSED (a single solid component larger than the cap is dropped — that's
matrix/hole, not talc), and never the drawn annotation.
"""
import cv2
import numpy as np

from app.shlif import talc
from app.pipeline import loader

CFG = loader.get_config()


def test_dark_gray_phase_keeps_dispersed_specks_drops_extremes_and_marks():
    h = w = 256
    rgb = np.full((h, w, 3), 200, np.uint8)          # bright grey matrix (excluded, the bulk)
    rgb[0:26] = 5                                     # ~10% black background (excluded)
    rgb[26:52] = 250                                  # ~10% bright sulfide (excluded)

    speck_pts = []
    for i in range(6):
        for j in range(6):
            y, x = 70 + i * 24, 20 + j * 36
            rgb[y:y + 6, x:x + 6] = 100              # dispersed medium-dark grey specks (talc)
            speck_pts.append((y, x))
    cv2.line(rgb, (10, 240), (240, 240), (25, 55, 230), 5)  # blue annotation, in the matrix

    mask, frac = talc.dark_gray_phase(rgb, CFG.talc)

    assert not mask[0:26].any()          # black excluded
    assert not mask[26:52].any()         # bright sulfide excluded
    assert not mask[talc.blue_line_mask(rgb, CFG.talc)].any()  # annotation excluded
    # the dispersed specks are picked up
    hit = sum(mask[y:y + 6, x:x + 6].any() for y, x in speck_pts)
    assert hit >= len(speck_pts) * 0.8
    assert 0.0 < frac <= 0.12


def test_dark_gray_phase_drops_a_single_large_solid_blob():
    # one big medium-dark region is matrix/hole, not dispersed talc → dropped.
    h = w = 256
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[0:38] = 10                        # ~15% black bottom of range
    rgb[38:140] = 90                      # ~40% solid medium-dark blob
    rgb[140:] = 220                       # ~45% bright matrix
    mask, frac = talc.dark_gray_phase(rgb, CFG.talc)
    assert frac == 0.0
    assert not mask.any()
