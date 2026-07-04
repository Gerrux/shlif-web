"""Auto-detect closeup vs panorama by pixel count — replaces the manual
mode toggle. Threshold picked from the real dataset: closeups top out at
~26 MP, panoramas start at ~126.5 MP (see docs/superpowers/specs/
2026-07-04-panorama-closeup-unification-design.md §1)."""
from app.pipeline import detect, loader

CFG = loader.get_config()


def test_detect_mode_closeup_below_threshold():
    assert detect.detect_mode(5000, 4000, CFG) == "closeup"  # 20 MP


def test_detect_mode_panorama_above_threshold():
    assert detect.detect_mode(13330, 9489, CFG) == "panorama"  # 126.5 MP, real sample size


def test_detect_mode_boundary_is_inclusive_of_threshold():
    thr = int(CFG.tiling.direct_max_pixels)
    assert detect.detect_mode(thr, 1, CFG) == "closeup"       # exactly at threshold
    assert detect.detect_mode(thr + 1, 1, CFG) == "panorama"  # one pixel over
