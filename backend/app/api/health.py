from __future__ import annotations
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health() -> dict:
    # Task 4 replaces the stub bodies with real detection via pipeline.loader.
    return {
        "status": "ok",
        "gpu": False,
        "models": {"classifier": False, "unet_ore": False, "unet_talc": False},
    }
