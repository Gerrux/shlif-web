"""Adaptive sigmoid threshold for a talc probability map (borrowed cascade):
max(p95*0.85, p99*0.55) clamped to [0.05, 0.30] — a weak-but-present talc signal
still yields a non-empty, non-flooded mask; pure noise clamps out cleanly."""
import numpy as np
import pytest

from app.shlif.talc_unet import resolve_threshold


def test_resolve_threshold_clamps_to_range():
    assert resolve_threshold(np.full((20, 20), 0.01, np.float32)) == pytest.approx(0.05)  # floor
    assert resolve_threshold(np.full((20, 20), 0.95, np.float32)) == pytest.approx(0.30)  # ceiling


def test_resolve_threshold_surfaces_a_sparse_peak():
    prob = np.full((100, 100), 0.08, np.float32)
    prob[:5, :] = 0.85                      # a strong ~5% talc region
    thr = resolve_threshold(prob)
    assert 0.05 <= thr <= 0.30
    mask = prob >= thr
    assert mask.any()                       # non-empty
    assert mask.mean() < 0.2                # not flooded
