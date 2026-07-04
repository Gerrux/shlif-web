"""count_tiles gives the panorama progress bar a cheap upfront total-tile estimate,
without decoding any tile pixels."""
import numpy as np
from PIL import Image

from app.shlif import tiling
from app.shlif.config import Config


def test_count_tiles_matches_iter_tiles_exactly(tmp_path):
    img = np.zeros((300, 300, 3), np.uint8)
    p = tmp_path / "t.png"
    Image.fromarray(img).save(p, "PNG")
    cfg = Config({"tile": 128, "overlap": 32, "max_pixels": 1_000_000,
                  "skip_empty": False, "empty_bright_frac": 0.002})

    counted = tiling.count_tiles(str(p), cfg)
    actual = sum(1 for _ in tiling.iter_tiles(str(p), cfg))

    assert counted == actual == 16


def test_count_tiles_exact_when_tail_tile_would_be_skipped(tmp_path):
    img = np.zeros((295, 295, 3), np.uint8)
    p = tmp_path / "t.png"
    Image.fromarray(img).save(p, "PNG")
    cfg = Config({"tile": 128, "overlap": 32, "max_pixels": 1_000_000,
                  "skip_empty": False, "empty_bright_frac": 0.002})

    counted = tiling.count_tiles(str(p), cfg)
    actual = sum(1 for _ in tiling.iter_tiles(str(p), cfg))

    assert counted == actual == 9
