"""Ensemble-perturbation uncertainty for the classical phase segmentation.

Segment the image under a few soft photometric perturbations; where the phase
label is stable across the ensemble the pixel is confident, where it flips the
pixel is disputed. Yields a per-pixel confidence map + an undetermined_fraction —
an honesty/UX signal for the human-in-the-loop (borrowed idea)."""
import os
import threading
import time

import numpy as np
import pytest

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


def test_ensemble_uncertainty_reports_progress_per_perturbation():
    rgb = np.full((64, 64, 3), 10, np.uint8)
    rgb[16:48, 16:48] = 245
    calls = []
    uncertainty.ensemble_uncertainty(rgb, CFG, on_step=lambda i, total: calls.append((i, total)))

    total = len(uncertainty._PERTURBATIONS)
    assert calls == [(i, total) for i in range(1, total + 1)]


@pytest.mark.skipif(
    (os.cpu_count() or 1) <= 1,
    reason="needs >1 CPU to observe overlap",
)
def test_perturbations_run_concurrently(monkeypatch):
    lock = threading.Lock()
    current = [0]
    max_concurrent = [0]

    def fake_segment_phases(pre, cfg):
        with lock:
            current[0] += 1
            max_concurrent[0] = max(max_concurrent[0], current[0])
        time.sleep(0.05)
        with lock:
            current[0] -= 1
        class _R:
            labels = np.zeros(pre.shape[:2], np.uint8)
        return _R()

    monkeypatch.setattr(uncertainty, "segment_phases", fake_segment_phases)
    rgb = np.zeros((16, 16, 3), np.uint8)
    uncertainty.ensemble_phase_labels(rgb, CFG)

    assert max_concurrent[0] >= 2, (
        f"expected overlapping perturbation calls, max concurrent was {max_concurrent[0]}")


def test_matches_manual_sequential_reference():
    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[8:24, 8:24] = 220
    expected = np.stack([
        uncertainty.segment_phases(
            uncertainty.preprocess(uncertainty._perturb(rgb, gamma, gain), CFG.preprocess),
            CFG.segment,
        ).labels.astype(np.uint8)
        for gamma, gain in uncertainty._PERTURBATIONS
    ])
    actual = uncertainty.ensemble_phase_labels(rgb, CFG)
    assert np.array_equal(actual, expected)
