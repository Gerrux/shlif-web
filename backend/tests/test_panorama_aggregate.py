"""Ore-density-weighted section aggregation (borrowed prior).

On disseminated grey whole-sections both the classical and U-Net segmenters read
every tile as ore-bearing, so an area-weighted verdict averages silicate fields
in. Weighting each tile by its bright-pixel density — the fraction above the
panorama's global 92nd brightness percentile — lets the actual sulfide-bearing
tiles drive the verdict and stops empty silicate fields from diluting it."""
import numpy as np
import pytest

from app.pipeline import panorama

CLASSES = ["ordinary", "hard", "talcose"]


def test_ore_density_counts_bright_fraction():
    g = np.zeros((10, 10), np.float32)
    g[:2, :] = 200.0                       # 20% bright
    assert panorama.ore_density(g, 100.0) == pytest.approx(0.2)


def test_aggregate_section_lets_dense_tile_win_over_faint_one():
    # a high-density ordinary tile + a near-empty (silicate) talcose tile:
    # the faint tile must not flip the verdict.
    records = [
        ({"ordinary": 0.9, "hard": 0.05, "talcose": 0.05}, 0.50),
        ({"ordinary": 0.1, "hard": 0.00, "talcose": 0.90}, 0.02),
    ]
    sec = panorama.aggregate_section(records, CLASSES)
    assert CLASSES[int(sec.argmax())] == "ordinary"


def test_aggregate_section_all_zero_weight_falls_back_to_uniform():
    records = [
        ({"ordinary": 0.8, "hard": 0.2, "talcose": 0.0}, 0.0),
        ({"ordinary": 0.2, "hard": 0.8, "talcose": 0.0}, 0.0),
    ]
    sec = panorama.aggregate_section(records, CLASSES)
    assert sec == pytest.approx([0.5, 0.5, 0.0])   # unweighted mean, not divide-by-zero


def test_aggregate_section_empty_is_zero_vector():
    sec = panorama.aggregate_section([], CLASSES)
    assert sec.tolist() == [0.0, 0.0, 0.0]
