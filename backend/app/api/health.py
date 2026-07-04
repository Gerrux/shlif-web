from __future__ import annotations
from fastapi import APIRouter
from app.pipeline import loader

router = APIRouter()

@router.get("/health")
def health() -> dict:
    return {"status": "ok", "gpu": loader.gpu_available(), "models": loader.model_status()}
