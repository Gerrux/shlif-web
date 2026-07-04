from __future__ import annotations
import json, sqlite3, threading, uuid
from pathlib import Path
from app.schemas.jobs import JobRecord

class JobStore:
    def __init__(self, db_path: Path):
        self._path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, mode TEXT, status TEXT, progress REAL,
                message TEXT, result TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS corrections(
                id TEXT PRIMARY KEY, job_id TEXT, layer TEXT, n_pixels INTEGER, ts TEXT)""")

    def _conn(self):
        return sqlite3.connect(self._path, timeout=30, check_same_thread=False)

    def create(self, mode: str) -> str:
        jid = uuid.uuid4().hex
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?)",
                      (jid, mode, "queued", 0.0, None, None))
        return jid

    def get(self, jid: str) -> JobRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT id,mode,status,progress,message,result FROM jobs WHERE id=?",
                            (jid,)).fetchone()
        if not row: return None
        return JobRecord(id=row[0], mode=row[1], status=row[2], progress=row[3],
                         message=row[4], result=json.loads(row[5]) if row[5] else None)

    def set_status(self, jid, status, progress=None, message=None):
        with self._lock, self._conn() as c:
            if progress is None:
                c.execute("UPDATE jobs SET status=?,message=? WHERE id=?", (status, message, jid))
            else:
                c.execute("UPDATE jobs SET status=?,progress=?,message=? WHERE id=?",
                          (status, progress, message, jid))

    def set_result(self, jid, result: dict):
        with self._lock, self._conn() as c:
            c.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), jid))

    def log_correction(self, job_id, layer, n_pixels):
        import time as _t
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO corrections VALUES(?,?,?,?,?)",
                      (uuid.uuid4().hex, job_id, layer, int(n_pixels),
                       _t.strftime("%Y-%m-%dT%H:%M:%S")))
