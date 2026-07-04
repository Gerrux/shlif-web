"""Runtime wires JobRunner with max_workers=2 so a second job (or a
health-check) isn't serialised behind a long-running panorama analysis --
the priority is single-panorama latency, but this is a free, low-risk fix
for job-level concurrency on top of that."""
from app.runtime import Runtime


def test_runner_allows_two_concurrent_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.paths.db_path", lambda: tmp_path / "t.db")
    rt = Runtime()
    assert rt.runner._pool._max_workers == 2
