import json
from fastapi.testclient import TestClient
from main import app
from app.core import paths as core_paths


def test_get_manifest_and_tile_returns_200(tmp_path, monkeypatch):
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    out = core_paths.tiles_dir("jobA")
    (out / "manifest.json").write_text(json.dumps({"width": 10, "height": 10, "tileSize": 256, "maxLevel": 0}))
    (out / "0").mkdir()
    (out / "0" / "0_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal fake JPEG bytes

    c = TestClient(app)
    r = c.get("/api/tiles/jobA/manifest.json")
    assert r.status_code == 200
    assert r.json() == {"width": 10, "height": 10, "tileSize": 256, "maxLevel": 0}

    r = c.get("/api/tiles/jobA/0/0_0.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"


def test_get_manifest_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(core_paths.settings, "data_dir", tmp_path)
    c = TestClient(app)
    assert c.get("/api/tiles/missingjob/manifest.json").status_code == 404
    assert c.get("/api/tiles/missingjob/0/0_0.jpg").status_code == 404
