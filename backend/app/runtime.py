from __future__ import annotations
import os
from app.core import paths

class Runtime:
    """Holds the app-scoped job store + runner (single granian worker)."""
    def __init__(self) -> None:
        from app.jobs.store import JobStore
        from app.jobs.runner import JobRunner
        self.store = JobStore(paths.db_path())
        workers = int(os.environ.get("SHLIF_JOB_WORKERS", "4"))
        self.runner = JobRunner(self.store, max_workers=workers)

_runtime: Runtime | None = None

def get_runtime() -> Runtime:
    """Return the process-wide Runtime, creating it on first use (call-time, not import-time)."""
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime
