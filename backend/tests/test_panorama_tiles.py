import json
import numpy as np
from PIL import Image
from app.pipeline import tiles


def test_build_pyramid_writes_expected_levels_and_manifest(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)

    arr = (np.random.default_rng(0).integers(0, 255, (600, 1000, 3))).astype(np.uint8)
    tiles.build_pyramid(arr, "jobA")

    out_dir = tmp_path / "tiles" / "jobA"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest == {"width": 1000, "height": 600, "tileSize": 256, "maxLevel": 2}

    # level 0 (150x250): fits in a single tile
    assert sorted(p.name for p in (out_dir / "0").iterdir()) == ["0_0.jpg"]
    assert Image.open(out_dir / "0" / "0_0.jpg").size == (250, 150)

    # level 1 (300x500): 2x2 tiles, edges cropped to 244/44
    assert sorted(p.name for p in (out_dir / "1").iterdir()) == ["0_0.jpg", "0_1.jpg", "1_0.jpg", "1_1.jpg"]
    assert Image.open(out_dir / "1" / "1_1.jpg").size == (244, 44)  # last col x last row

    # level 2 (maxLevel, full resolution 600x1000): 4 cols x 3 rows
    level2 = out_dir / "2"
    assert len(list(level2.iterdir())) == 12
    assert Image.open(level2 / "3_2.jpg").size == (232, 88)  # last col (x=768..1000) x last row (y=512..600)

    # reconstructing maxLevel's tiles must match the source array's dimensions
    recon = np.zeros((600, 1000, 3), np.uint8)
    for row in range(3):
        for col in range(4):
            tile = np.asarray(Image.open(level2 / f"{col}_{row}.jpg"))
            th, tw = tile.shape[:2]
            recon[row * 256: row * 256 + th, col * 256: col * 256 + tw] = tile
    assert recon.shape == arr.shape


def test_build_pyramid_single_level_when_already_small(tmp_path, monkeypatch):
    from app.core import paths as core_paths
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)

    arr = (np.random.default_rng(1).integers(0, 255, (100, 150, 3))).astype(np.uint8)
    tiles.build_pyramid(arr, "jobB")

    out_dir = tmp_path / "tiles" / "jobB"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest == {"width": 150, "height": 100, "tileSize": 256, "maxLevel": 0}
    assert sorted(p.name for p in (out_dir / "0").iterdir()) == ["0_0.jpg"]
