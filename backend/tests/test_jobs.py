import sqlite3
import time
from app.jobs.store import JobStore
from app.jobs.runner import JobRunner

def test_job_lifecycle_success(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")
    assert store.get(jid).status == "queued"
    runner.submit(jid, lambda: {"ore_class": "ordinary"})
    for _ in range(50):
        if store.get(jid).status == "done": break
        time.sleep(0.05)
    rec = store.get(jid)
    assert rec.status == "done" and rec.result == {"ore_class": "ordinary"}

def test_job_lifecycle_error(tmp_path):
    store = JobStore(tmp_path / "t.db")
    runner = JobRunner(store)
    jid = store.create("closeup")
    def boom(): raise ValueError("nope")
    runner.submit(jid, boom)
    for _ in range(50):
        if store.get(jid).status == "error": break
        time.sleep(0.05)
    rec = store.get(jid)
    assert rec.status == "error" and "nope" in rec.message

def test_log_correction_inserts_row(tmp_path):
    db_path = tmp_path / "t.db"
    store = JobStore(db_path)
    jid = store.create("closeup")
    store.log_correction(jid, "talc", 42)

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT job_id, layer, n_pixels FROM corrections WHERE job_id=?", (jid,)
        ).fetchone()
    finally:
        conn.close()

    assert row == (jid, "talc", 42)
