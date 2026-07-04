from __future__ import annotations
from fastapi import FastAPI
from app.api import health

def create_app() -> FastAPI:
    app = FastAPI(title="Шлиф-Web API")
    app.include_router(health.router, prefix="/api")
    return app

app = create_app()
