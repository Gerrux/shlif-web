"""Ensemble-perturbation uncertainty for the classical phase segmentation.

Segment the image under a few soft photometric perturbations; where the phase
label is stable across the ensemble the pixel is confident, where it flips the
pixel is disputed. Yields a per-pixel confidence map + an undetermined_fraction —
an honesty/UX signal for the human-in-the-loop (borrowed idea)."""
import numpy as np

from app.shlif import uncertainty
from app.pipeline import loader

CFG = loader.get_config()


def test_high_contrast_image_is_mostly_confident():
    rgb = np.full((128, 128, 3), 10, np.uint8)   # dark matrix
    rgb[40:88, 40:88] = 245                        # unambiguous bright sulfide
    u = uncertainty.ensemble_uncertainty(rgb, CFG)
    assert u["confidence"].shape == (128, 128)
    assert 0.0 <= u["undetermined_fraction"] <= 1.0
    assert u["undetermined_fraction"] < 0.15       # the ensemble agrees almost everywhere
    assert u["confidence"].min() >= 0.0 and u["confidence"].max() <= 1.0


def test_ambiguous_grey_is_not_more_confident_than_clear():
    clear = np.full((128, 128, 3), 10, np.uint8)
    clear[40:88, 40:88] = 245
    rng = np.random.default_rng(0)
    ambiguous = rng.integers(95, 150, (128, 128, 3), dtype=np.uint8)  # mid-grey near thresholds
    u_clear = uncertainty.ensemble_uncertainty(clear, CFG)
    u_amb = uncertainty.ensemble_uncertainty(ambiguous, CFG)
    assert u_amb["undetermined_fraction"] >= u_clear["undetermined_fraction"]
