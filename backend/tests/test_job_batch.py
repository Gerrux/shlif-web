import sqlite3
from app.jobs.store import JobStore


def test_create_stores_batch_id_and_filename(tmp_path):
    store = JobStore(tmp_path / "t.db")
    jid = store.create("closeup", batch_id="batch-1", filename="a.png")
    rec = store.get(jid)
    assert rec.batch_id == "batch-1"
    assert rec.filename == "a.png"
    assert rec.created_at is not None


def test_create_without_batch_id_leaves_it_null(tmp_path):
    store = JobStore(tmp_path / "t.db")
    jid = store.create("closeup")
    rec = store.get(jid)
    assert rec.batch_id is None
    assert rec.filename is None


def test_list_by_batch_returns_jobs_in_creation_order(tmp_path):
    store = JobStore(tmp_path / "t.db")
    j1 = store.create("closeup", batch_id="batch-1", filename="a.png")
    j2 = store.create("closeup", batch_id="batch-1", filename="b.png")
    store.create("closeup", batch_id="other-batch", filename="c.png")
    recs = store.list_by_batch("batch-1")
    assert [r.id for r in recs] == [j1, j2]
    assert [r.filename for r in recs] == ["a.png", "b.png"]


def test_list_by_batch_empty_for_unknown_batch(tmp_path):
    store = JobStore(tmp_path / "t.db")
    assert store.list_by_batch("nope") == []


def test_legacy_db_without_batch_columns_migrates_cleanly(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""CREATE TABLE jobs(
        id TEXT PRIMARY KEY, mode TEXT, status TEXT, progress REAL,
        message TEXT, result TEXT)""")
    conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                 ("old1", "closeup", "done", 1.0, None, None))
    conn.commit()
    conn.close()

    store = JobStore(db_path)
    rec = store.get("old1")
    assert rec.batch_id is None
    assert rec.filename is None

    new_jid = store.create("closeup", batch_id="b1", filename="x.jpg")
    assert store.get(new_jid).batch_id == "b1"
