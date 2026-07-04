"""analyze_closeup reports progress through on_progress across its pipeline stages."""
import numpy as np

from app.pipeline import closeup, loader

CFG = loader.get_config()


def test_analyze_closeup_reports_progress():
    rgb = np.full((256, 256, 3), 10, np.uint8)
    rgb[80:176, 80:176] = 245
    calls = []
    closeup.analyze_closeup(rgb, CFG, on_progress=lambda p, msg: calls.append((p, msg)))

    assert len(calls) >= 5
    progresses = [p for p, _ in calls]
    assert progresses == sorted(progresses)
    assert all(0.0 <= p <= 1.0 for p in progresses)

    messages = " ".join(msg for _, msg in calls if msg)
    assert "сегментация" in messages
    assert "неопределённост" in messages
    assert "карт" in messages


def test_analyze_closeup_works_without_on_progress():
    rgb = np.full((64, 64, 3), 10, np.uint8)
    r = closeup.analyze_closeup(rgb, CFG)
    assert r["verdict"]["ore_class"] in {"ordinary", "hard", "talcose", "review"}
