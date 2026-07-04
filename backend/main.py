from __future__ import annotations
from fastapi import FastAPI
from app.core import paths

# app-scoped singletons (single granian worker; threadpool inside)
class _Runtime:
    def __init__(self):
        from app.jobs.store import JobStore
        from app.jobs.runner import JobRunner
        self.store = JobStore(paths.db_path())
        self.runner = JobRunner(self.store)

def create_app() -> FastAPI:
    import app as app_pkg
    app_pkg.runtime = _Runtime()          # published for routers: `from app import runtime`
    from app.api import health, analyze, jobs, masks
    api = FastAPI(title="Шлиф-Web API")
    for r in (health.router, analyze.router, jobs.router, masks.router):
        api.include_router(r, prefix="/api")
    return api

app = create_app()
