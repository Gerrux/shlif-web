from fastapi.testclient import TestClient
from main import app

def test_health_ok():
    c = TestClient(app)
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["gpu"] is False  # no torch in test env
    assert set(body["models"]) == {"classifier", "unet_ore", "unet_talc"}
