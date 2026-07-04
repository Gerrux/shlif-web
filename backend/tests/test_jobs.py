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
