"""Runtime wires JobRunner's worker count from SHLIF_JOB_WORKERS (default 4) so a batch
upload of many images actually runs concurrently in the background instead of queueing
strictly one at a time behind a long-running panorama analysis."""
from app.runtime import Runtime


def test_runner_uses_default_worker_count_without_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.paths.db_path", lambda: tmp_path / "t.db")
    monkeypatch.delenv("SHLIF_JOB_WORKERS", raising=False)
    rt = Runtime()
    assert rt.runner._pool._max_workers == 4


def test_runner_honors_shlif_job_workers_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.paths.db_path", lambda: tmp_path / "t.db")
    monkeypatch.setenv("SHLIF_JOB_WORKERS", "6")
    rt = Runtime()
    assert rt.runner._pool._max_workers == 6
