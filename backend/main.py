from __future__ import annotations
from fastapi import FastAPI
from app.runtime import get_runtime

def create_app() -> FastAPI:
    get_runtime()  # initialize the job store/runner singleton at startup
    from app.api import health, analyze, jobs, masks, report
    api = FastAPI(title="Шлиф-Web API")
    for r in (health.router, analyze.router, jobs.router, masks.router, report.router):
        api.include_router(r, prefix="/api")
    return api

app = create_app()
