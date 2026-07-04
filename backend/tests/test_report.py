"""PDF protocol export — Cyrillic passport-style report for a finished job."""
import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.pipeline import report
from main import app

CLOSEUP_RESULT = {
    "mode": "closeup",
    "verdict": {
        "ore_class": "ordinary",
        "text": "Классифицирована как рядовая руда.",
        "metrics": {
            "sulfide_frac": 0.21, "magnetite_frac": 0.05, "matrix_frac": 0.74,
            "talc_frac": 0.03, "talc_share_est": 0.04, "fine_share": 0.30,
            "confidence": 0.71, "undetermined_fraction": 0.08,
        },
    },
    "low_conf_zones": [{"area": 500, "phase_a": "сульфид", "phase_b": "магнетит",
                        "bbox": [0, 0, 10, 10]}],
}


def test_build_report_pdf_is_a_pdf_with_cyrillic():
    pdf = report.build_report_pdf("abc123", "closeup", CLOSEUP_RESULT, None)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 1500


def test_build_report_pdf_handles_panorama_and_missing_metrics():
    pano = {"mode": "panorama",
            "verdict": {"ore_class": "review", "text": "",
                        "metrics": {"talc_frac": 0.01, "confidence": 0.4}}}
    pdf = report.build_report_pdf("pano1", "panorama", pano, None)
    assert pdf[:5] == b"%PDF-"


def _png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


def _poll(c, jid):
    for _ in range(100):
        r = c.get(f"/api/jobs/{jid}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_report_endpoint_returns_pdf(tiny_rgb):
    c = TestClient(app)
    up = c.post("/api/analyze", data={"mode": "closeup"},
                files={"image": ("t.png", _png_bytes(tiny_rgb), "image/png")})
    jid = up.json()["job_id"]
    assert _poll(c, jid)["status"] == "done"
    r = c.get(f"/api/report/{jid}.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_report_endpoint_404_for_unknown_job():
    c = TestClient(app)
    assert c.get("/api/report/does-not-exist.pdf").status_code == 404
