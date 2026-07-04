"""Annotation robustness of the talc / feature pipeline.

Two borrowed techniques (see TODO in origin repo):
  * cyan strokes must count as annotation, not just pure blue (HSV range sync);
  * hand-drawn annotation must be inpainted out *before* texture features so the
    blue/cyan pixels never leak into GLCM/LBP (a classifier-leak guard).
"""
import cv2
import numpy as np
import pytest

from app.shlif import talc
from app.shlif.features import extract_features
from app.pipeline import loader

CFG = loader.get_config()


def _matrix(color=(120, 105, 90), size=(200, 200)):
    return np.full((*size, 3), color, np.uint8)


def test_blue_line_mask_detects_cyan_stroke():
    # cyan = low R, high G, high B — the pure-blue RGB rule (B-G large) misses it.
    rgb = _matrix()
    cv2.line(rgb, (0, 100), (199, 100), (20, 220, 220), 6)
    assert talc.blue_line_mask(rgb, CFG.talc).sum() > 0


def test_strip_annotation_removes_blue_and_cyan_strokes():
    rgb = _matrix()
    cv2.line(rgb, (0, 60), (199, 60), (30, 60, 220), 6)     # blue
    cv2.line(rgb, (0, 140), (199, 140), (20, 220, 220), 6)  # cyan
    assert talc.blue_line_mask(rgb, CFG.talc).sum() > 0
    cleaned = talc.strip_annotation(rgb, CFG.talc)
    assert talc.blue_line_mask(cleaned, CFG.talc).sum() == 0


def test_extract_features_strips_annotation_before_texture():
    # extract_features must inpaint annotation, so an annotated image yields the
    # same features as the pre-stripped one (no blue leaking into the texture).
    # Grey base (r==g==b) → no pixel is ever "blue", so strip is idempotent and the
    # comparison is deterministic; the drawn polyline is the only annotation.
    rng = np.random.default_rng(0)
    gray = rng.integers(40, 190, (500, 640, 1), dtype=np.uint8)
    base = np.repeat(gray, 3, axis=2)
    annotated = base.copy()
    pts = np.array([[100, 90], [500, 110], [470, 380], [130, 360]], np.int32)
    cv2.polylines(annotated, [pts], True, (25, 55, 230), 9)

    pre_stripped = talc.strip_annotation(annotated, CFG.talc)
    f_pre = extract_features(pre_stripped, CFG)   # already clean → internal strip is a no-op
    f_auto = extract_features(annotated, CFG)      # must strip the polyline itself

    assert set(f_auto) == set(f_pre)
    for k in f_auto:
        assert f_auto[k] == pytest.approx(f_pre[k], rel=1e-6, abs=1e-6)
