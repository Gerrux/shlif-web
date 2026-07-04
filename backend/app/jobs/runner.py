from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from app.jobs.store import JobStore

class JobRunner:
    def __init__(self, store: JobStore, max_workers: int = 1):
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, jid: str, fn: Callable[[Callable[[float, str | None], None]], dict]) -> None:
        self._store.set_status(jid, "running", progress=0.05)
        self._pool.submit(self._run, jid, fn)

    def _run(self, jid: str, fn: Callable[[Callable[[float, str | None], None]], dict]) -> None:
        def report(progress: float, message: str | None = None) -> None:
            self._store.set_status(jid, "running", progress=progress, message=message)
        try:
            result = fn(report)
            self._store.set_result(jid, result)
            self._store.set_status(jid, "done", progress=1.0)
        except Exception as e:  # noqa: BLE001 — surfaced to the client as status=error
            self._store.set_status(jid, "error", message=str(e))
