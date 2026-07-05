from __future__ import annotations
import json, sqlite3, threading, time, uuid
from contextlib import contextmanager
from pathlib import Path
from app.schemas.jobs import JobRecord

class JobStore:
    def __init__(self, db_path: Path):
        self._path = str(db_path)
        self._lock = threading.Lock()
        with self._tx() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, mode TEXT, status TEXT, progress REAL,
                message TEXT, result TEXT, batch_id TEXT, filename TEXT, created_at TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS corrections(
                id TEXT PRIMARY KEY, job_id TEXT, layer TEXT, n_pixels INTEGER, ts TEXT)""")
            existing = {row[1] for row in c.execute("PRAGMA table_info(jobs)").fetchall()}
            for col in ("batch_id", "filename", "created_at"):
                if col not in existing:
                    c.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")

    def _conn(self):
        return sqlite3.connect(self._path, timeout=30, check_same_thread=False)

    @contextmanager
    def _tx(self):
        c = self._conn()
        try:
            with c:            # commit on success / rollback on exception
                yield c
        finally:
            c.close()          # deterministic close (the sqlite3 CM does not close)

    def create(self, mode: str, batch_id: str | None = None, filename: str | None = None) -> str:
        jid = uuid.uuid4().hex
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock, self._tx() as c:
            c.execute(
                "INSERT INTO jobs(id,mode,status,progress,message,result,batch_id,filename,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (jid, mode, "queued", 0.0, None, None, batch_id, filename, created_at))
        return jid

    def get(self, jid: str) -> JobRecord | None:
        with self._tx() as c:
            row = c.execute(
                "SELECT id,mode,status,progress,message,result,batch_id,filename,created_at "
                "FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row: return None
        return JobRecord(id=row[0], mode=row[1], status=row[2], progress=row[3],
                         message=row[4], result=json.loads(row[5]) if row[5] else None,
                         batch_id=row[6], filename=row[7], created_at=row[8])

    def list_by_batch(self, batch_id: str) -> list[JobRecord]:
        with self._tx() as c:
            rows = c.execute(
                "SELECT id,mode,status,progress,message,result,batch_id,filename,created_at "
                "FROM jobs WHERE batch_id=? ORDER BY created_at, rowid", (batch_id,)).fetchall()
        return [JobRecord(id=r[0], mode=r[1], status=r[2], progress=r[3], message=r[4],
                          result=json.loads(r[5]) if r[5] else None,
                          batch_id=r[6], filename=r[7], created_at=r[8]) for r in rows]

    def set_status(self, jid, status, progress=None, message=None):
        with self._lock, self._tx() as c:
            if progress is None:
                c.execute("UPDATE jobs SET status=?,message=? WHERE id=?", (status, message, jid))
            else:
                c.execute("UPDATE jobs SET status=?,progress=?,message=? WHERE id=?",
                          (status, progress, message, jid))

    def set_result(self, jid, result: dict):
        with self._lock, self._tx() as c:
            c.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), jid))

    def log_correction(self, job_id, layer, n_pixels):
        import time as _t
        with self._lock, self._tx() as c:
            c.execute("INSERT INTO corrections VALUES(?,?,?,?,?)",
                      (uuid.uuid4().hex, job_id, layer, int(n_pixels),
                       _t.strftime("%Y-%m-%dT%H:%M:%S")))
